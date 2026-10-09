"""Local Bareun candidate construction. No policy, scorer, or paid model calls."""
from collections import Counter
import json
from pathlib import Path
import random
import time

from ..common import read_json, sha_text, write_json
from ..env.analysis import ParagraphAnalyzer
from ..ko import render
from ..train.teacher_bulk import atomic_new
from .local import load_environment
from .config import PHASE, require_v2
from .data import source_pools
from .operators import Fusion, delete_link, fuse


class PriorityParagraphs(ParagraphAnalyzer):
    def profile(self, text, neighbors=()):
        key = sha_text(json.dumps([text, sha_text(json.dumps(neighbors, ensure_ascii=False))], ensure_ascii=False))
        if not (self.cache_dir / (key + '.json')).exists():
            status = self.config['paths']['repo'] / 'verak/v3/outputs/phase7_sft/continuation_status.json'
            while read_json(status)['current']['stage'] not in {'training_korean', 'waiting_for_global',
                                                                'completed_awaiting_final_audit'}:
                time.sleep(15)  # Yield the shared analyzer service throughout v1 evaluation.
        return super().profile(text, neighbors)


def make_row(example, source, score, changed, record, op, split, tokenizer):
    view = render(changed.structure(), compact=True)
    tokens = len(tokenizer.encode(view, add_special_tokens=False))
    if tokens > 3000:
        raise ValueError('corrupted_compact_view_exceeds_3000')
    episode_id = f'v2:{op}:{split}:{example.source_line}:v1'
    record['record_id'] = episode_id + ':R1'
    return {'schema_version': 'verak_v3_ops_v2', 'method_version': 'v2', 'episode_id': episode_id,
        'source_id': example.id, 'split': split, 'question': example.question,
        'question_hash': example.question_hash, 'genre': example.genre, 'level': 'L3',
        'operator': op, 'source_hash': example.essay_hash, 'source_text': source.text,
        'corrupted_text': changed.text, 'corrupted_hash': sha_text(changed.text),
        'source_layout': source.snapshot(), 'corrupted_layout': changed.snapshot(), 'records': [record],
        'compact_view': view, 'compact_tokens': tokens, 'q_source': score['mean'], 'q_corrupted': None,
        'preexisting_spell_spans': [], 'supervision_private': ['source_text', 'source_layout', 'records', 'q_source']}


def build(config, *, operator, limit=None):
    require_v2(config)
    if operator not in {'G_DEL_LINK', 'L_FUSE'}:
        raise ValueError('Unknown v2 operator')
    load_environment(config)
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')['plans'][operator]
    pools, _ = source_pools(config)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    analysis = PriorityParagraphs(config, cache_dir=root / 'bareun_paragraphs')
    stats, processed = {}, 0
    for split in ('agent_train', 'agent_dev'):
        sources = {e.id: (e, source, score) for e, source, score in pools[split]}
        counts = Counter()
        for entry in plan[split]:
            if limit is not None and processed >= limit:
                break
            source_id = entry['source_id']
            path = root / 'candidates' / operator / split / (source_id.replace(':', '_') + '.json')
            processed += 1
            if path.exists():
                old = read_json(path)
                if old['source_hash'] != entry['source_hash']:
                    raise ValueError('Saved candidate source changed')
                counts['reused'] += 1
                continue
            example, source, score = sources[source_id]
            failures, result = Counter(), None
            if operator == 'G_DEL_LINK':
                labels_path = root / 'labels' / (source_id.replace(':', '_') + '.json')
                if not labels_path.exists():
                    counts['awaiting_label'] += 1
                    continue
                labels = read_json(labels_path)
                if labels['source_hash'] != example.essay_hash:
                    raise ValueError('Stale Sol sentence labels')
                proposals = [(r['sid'], r['label']) for r in labels['labels'] if r['label'] != 'none']
            else:
                proposals = [Fusion(w['paragraph'], w['position'], tuple(w['sids']), tuple(w['source_classes']))
                             for w in entry['fusion_windows']]
            random.Random(config[PHASE]['seed'] + example.source_line).shuffle(proposals)
            for index, proposal in enumerate(proposals):
                try:
                    changed, record = (delete_link(source, *proposal) if operator == 'G_DEL_LINK'
                                       else fuse(source, proposal, analysis, variant=index % 2))
                    result = make_row(example, source, score, changed, record, operator, split, tokenizer)
                    break
                except ValueError as exc:
                    failures[str(exc)] += 1
            if result is None:
                failure = {'source_id': source_id, 'source_hash': example.essay_hash, 'operator': operator,
                           'reason': 'no_valid_surface_candidate', 'failures': dict(failures)}
                write_json(root / 'candidate_failures' / operator / split / path.name, failure)
                counts['failed'] += 1
            else:
                result['mechanical_rejections_before_selection'] = dict(failures)
                atomic_new(path, result)
                counts['created'] += 1
            if processed % 10 == 0:
                print(json.dumps({'operator': operator, 'processed': processed, 'split': split,
                    'counts': dict(counts), 'new_paid_calls': 0}, ensure_ascii=False), flush=True)
        stats[split] = dict(counts)
    result = {'operator': operator, 'counts': stats, 'paid_calls': 0, 'scorer_calls': 0,
              'gpu_used': False, 'new_local_bareun_profiles': len(analysis.calls), 'cached_profiles': analysis.hits}
    write_json(root / ('local_build_' + operator + ('_probe' if limit else '') + '.json'), result)
    return result
