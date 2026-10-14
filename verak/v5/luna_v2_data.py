"""Read train/valid only; keep private feedback out of every policy observation."""
from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re
import time

from verak.v3.common import extract_question_essay
from verak.v4.common import REPO, file_sha, sha_text, write_json, read_json
from verak.v4.data import read_rows, genre_of, normalized_source

ROOT = REPO / 'verak/v5/outputs/luna_upper_bound_v2'
SPEC = REPO / 'imple/FEAK-Agent_Luna시험_v2.md'
SEED = 20261014
CONDITIONS = ['A', 'A+', 'B', 'C', 'D']


def paragraphs(text):
    # Only original newline boundaries, never infer paragraphs from sentences.
    return [x.strip() for x in re.split(r'\n+', text.strip()) if x.strip()]


def metadata(row):
    return {k: v for k, v in row.items() if k not in {'question', 'text', 'original_text', 'feedback'}}


def balanced(pool, n):
    """Round robin through joint length/score tertiles, unseen questions first."""
    rng = random.Random(SEED)
    groups = defaultdict(list)
    ranks = {}
    for field in ('characters', 'human_mean'):
        ranks[field] = {r['source_id']: min(2, 3*i//len(pool)) for i, r in enumerate(sorted(pool, key=lambda x: (x[field], x['source_id'])))}
    for r in pool:
        groups[(ranks['characters'][r['source_id']], ranks['human_mean'][r['source_id']])].append(r)
    for values in groups.values():
        rng.shuffle(values)
        values.sort(key=lambda r: r['question_seen_in_training'], reverse=True)
    out, hashes = [], set()
    while len(out) < n:
        progress = False
        for key in sorted(groups):
            while groups[key]:
                r = groups[key].pop()
                if r['normalized_source_hash'] not in hashes:
                    out.append(r); hashes.add(r['normalized_source_hash']); progress = True
                    break
            if len(out) == n:
                break
        if not progress:
            raise ValueError('Insufficient distinct eligible sources')
    return out


def prepare():
    started = time.monotonic()
    target = ROOT / 'selection.json'
    if target.exists():
        return read_json(target)
    ROOT.mkdir(parents=True, exist_ok=True)
    paths = {s: REPO / 'data/data_jsonl' / (s+'.jsonl') for s in ('train', 'valid')}
    train_hashes, train_questions, blocked_ids = set(), set(), set()
    artifacts = [REPO / p for p in (
        'verak/v4/outputs/train44/data/manifest.json',
        'verak/v5/outputs/impl1_fast/data_impl1/sources_manifest.json',
        'verak/v5/outputs/impl1_fast/data_impl1/manifest.json',
        'verak/v4/outputs/prep/design.json')]
    artifact_hashes = {}
    def absorb(obj):
        if isinstance(obj, dict):
            for k, v in obj.items():
                if k == 'source_id' and isinstance(v, str): blocked_ids.add(v)
                if re.fullmatch(r'(train|valid):\d+', k): blocked_ids.add(k)
                if k in {'normalized_source_hash', 'normalized_space_hash', 'normalized_text_hash'} and isinstance(v, str): train_hashes.add(v)
                absorb(v)
        elif isinstance(obj, list):
            for v in obj: absorb(v)
    for p in artifacts:
        if p.exists():
            artifact_hashes[str(p)] = file_sha(p); absorb(read_json(p))
    pools = {'train': defaultdict(list), 'valid': defaultdict(list)}
    counts = Counter()
    for split, path in paths.items():
        for line, raw in read_rows(path):
            question, text = extract_question_essay(raw)
            norm, qhash = normalized_source(text), sha_text(question)
            if split == 'train':
                train_hashes.add(norm); train_questions.add(qhash)
            counts[split+'_rows'] += 1
            genre = genre_of(raw)
            ps = paragraphs(text)
            if len(ps) < 2:
                counts[split+'_single_paragraph'] += 1; continue
            if split == 'valid' and (norm in train_hashes or split+':'+str(line) in blocked_ids):
                counts['valid_excluded_training_overlap'] += 1; continue
            scores = raw.get('grader_1_scores', []) + raw.get('grader_2_scores', [])
            if not genre or len(scores) != 16 or not all(isinstance(x, (int, float)) for x in scores):
                counts[split+'_invalid_metadata'] += 1; continue
            r = {'source_id': f'{split}:{line}', 'genre': genre, 'question': question,
                 'original_text': text, 'text': '\n\n'.join(ps), 'paragraph_count': len(ps),
                 'boundary_kind': 'blank_lines' if len(re.split(r'\n\s*\n', text.strip())) > 1 else 'single_newlines',
                 'feedback': raw.get('assistant', ''), 'feedback_provenance': 'stored_assistant_field_authorship_unverified',
                 'question_hash': qhash, 'normalized_source_hash': norm, 'original_sha256': sha_text(text),
                 'human_mean': sum(scores)/16, 'characters': len(text),
                 'question_seen_in_training': qhash in train_questions}
            pools[split][genre].append(r)
    chosen = {}
    for split, quotas in [('train', {'explanatory': 4, 'argumentative': 3, 'emotional': 3}),
                          ('valid', {'explanatory': 17, 'argumentative': 17, 'emotional': 16})]:
        chosen[split] = []
        for genre, n in quotas.items():
            pool = pools[split][genre]
            # Enforce unseen-question preference globally before stratifying.
            unseen = [r for r in pool if not r['question_seen_in_training']]
            selected = balanced(unseen, n) if len(unseen) >= n else unseen + balanced([r for r in pool if r['question_seen_in_training']], n-len(unseen))
            chosen[split].extend(selected)
        for r in chosen[split]: write_json(ROOT/'sources'/(r['source_id'].replace(':', '_')+'.json'), r)
    if {r['normalized_source_hash'] for r in chosen['train']} & {r['normalized_source_hash'] for r in chosen['valid']}:
        raise ValueError('Preparation/evaluation overlap')
    value = {'seed': SEED, 'spec_sha256': file_sha(SPEC), 'test_texts_opened': False,
             'raw_inputs': {s: {'path': str(p), 'sha256': file_sha(p)} for s, p in paths.items()},
             'exclusion_manifests': artifact_hashes, 'counts': dict(counts),
             'eligible': {s: {g: len(rows) for g, rows in pools[s].items()} for s in pools},
             'unseen_eligible_valid': sum(not r['question_seen_in_training'] for rows in pools['valid'].values() for r in rows),
             'prep10': [metadata(r) for r in chosen['train']], 'eval50': [metadata(r) for r in chosen['valid']],
             'wall_seconds': time.monotonic()-started, 'status': 'selected_not_evaluated'}
    write_json(target, value)
    write_json(ROOT/'future_training_exclusions.json', {'reason': 'Luna v2 reserved evaluation cohort', 'sources': value['eval50']})
    return value


def initialize_contract():
    p = ROOT/'api_contract.json'
    contract = {'user_approved': True, 'total_cap_usd': 5.0,
                'luna': {'model': 'gpt-6-luna', 'cap_usd': 5.0, 'endpoint': 'chat_completions',
                         'reasoning_effort': 'none', 'temperature': 0, 'input_per_million': .10,
                         'output_per_million': .50, 'cache_read_per_million': .01, 'cache_write_per_million': .125},
                'claude': {'model': 'claude-sonnet-5-5', 'cap_usd': 5.0, 'input_per_million': 2.0,
                           'output_per_million': 10.0, 'cache_read_per_million': .10, 'cache_write_per_million': 2.50,
                           'generation_options': {'output_config': {'effort': 'medium'}, 'thinking': {'type': 'between_tools'}}},
                'temperature_exception': 'User approved preserving Sonnet 5.5 and omitting unsupported temperature=0.'}
    if p.exists() and read_json(p) != contract: raise ValueError('API contract changed')
    write_json(p, contract)


if __name__ == '__main__':
    value = prepare(); initialize_contract()
    print(json.dumps({k: value[k] for k in ('counts', 'eligible', 'unseen_eligible_valid', 'wall_seconds')}, ensure_ascii=False))
