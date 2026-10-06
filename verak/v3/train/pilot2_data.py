"""Versioned content-scope corpus and independently judged conjunction deletions."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import json
import random

from ..common import file_sha, read_json, sha_text, write_json
from ..phase2 import read_jsonl, write_jsonl
from ..ko import render
from ..corrupt.document import BareunBank
from ..corrupt.builder import load_sources
from ..corrupt.instance_policy import candidates as local_candidates
from ..corrupt.missing_conjunction import candidates as drop_candidates
from ..corrupt.operators import apply, restore_record, exact_restoration_satisfies
from ..corrupt.api import QC_PROMPT, QCResponse
from ..corrupt.qc import qc_payload, validate_judgments

PHASE = 'phase7_pilot2'


def inventory(rows):
    records = [r for row in rows for r in row['records']]
    local = Counter(r['level'] for r in records if r['level'] != 'GLOBAL')
    return {'essays': len(rows), 'sources': len({r['source_id'] for r in rows}),
        'levels': dict(Counter(r['level'] for r in rows)),
        'genres': dict(Counter(r['genre'] for r in rows)),
        'operators': dict(Counter(r['op'] for r in records)),
        'local_counts': dict(local),
        'local_shares': {k: local[k]/sum(local.values()) if local else 0. for k in ('WORD', 'SENTENCE', 'TEXT')}}


def without_deletions(rows):
    return [r for r in rows if all(x['op'] != 'G_DELETE_SUPPORT' for x in r['records'])]


def bank_for(config):
    return BareunBank(config, cache_dir=config['paths'][PHASE+'_output']/'bareun_units',
        read_cache_dirs=tuple(config['paths'][p]/'bareun_units' for p in
            ('phase7_pilot_output', 'phase6_output', 'phase5_output', 'phase3b_output', 'phase3_output')))


def build(config):
    from transformers import AutoTokenizer
    root = config['paths'][PHASE+'_output']
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    bank = bank_for(config)
    seed = config[PHASE]['seed']
    for split, count in [('agent_train', config[PHASE]['train_candidates']), ('agent_dev', config[PHASE]['dev_candidates'])]:
        path = root/f'candidates_{split}.jsonl'
        if path.exists() and len(read_jsonl(path)) == count:
            continue
        if path.exists():
            raise ValueError('Partial build exists: inspect checkpoint before rebuilding')
        sources = load_sources(config, split, bank)
        random.Random(seed).shuffle(sources)
        rows, attempts, failures = [], Counter(), Counter()
        for example, source, score in sources:
            rng = random.Random(seed + example.source_line)
            proposals = list(drop_candidates(source))
            rng.shuffle(proposals)
            if not proposals:
                continue
            desired = 'L1' if len(rows) % 2 == 0 else 'L2'
            accepted = None
            for proposal in proposals:
                attempts['L_CONJ_DROP'] += 1
                try:
                    changed, record = apply(source, proposal, bank)
                    records = [record]
                    if desired == 'L2':
                        # One WORD and one TEXT edit on separate sentences keeps the
                        # three levels equally represented within each L2 example.
                        for options in [('L_CONN', 'L_POLARITY', 'L_SPACING'), ('L_REGISTER',)]:
                            choices = list(options)
                            rng.shuffle(choices)
                            added = False
                            used = {sid for r in records for sid in r['sids']}
                            for op in choices:
                                variants = [p for p in local_candidates(changed, op) if not used.intersection(p.sids)]
                                rng.shuffle(variants)
                                for other in variants[:8]:
                                    attempts[op] += 1
                                    try:
                                        next_doc, next_record = apply(changed, other, bank)
                                    except ValueError as exc:
                                        failures[f'{op}: {exc}'] += 1
                                        continue
                                    changed = next_doc
                                    records.append(next_record)
                                    added = True
                                    break
                                if added:
                                    break
                            if not added:
                                raise ValueError('No distinct verified WORD/TEXT site for L2')
                    view = render(changed.structure(), compact=True)
                    tokens = len(tokenizer.encode(view, add_special_tokens=False))
                    if tokens > config['view_token_budget']:
                        raise ValueError('compact_view_exceeds_3000')
                    restored = changed
                    for r in reversed(records):
                        restored = restore_record(restored, r, bank)
                    if restored.text != source.text or not all(exact_restoration_satisfies(source, r) for r in records):
                        raise ValueError('Roundtrip recovery failed')
                    id = f'phase7drop:{split}:{example.source_line}:v1'
                    for i, r in enumerate(records, 1):
                        r['record_id'] = f'{id}:R{i}'
                        r['params']['seed'] = seed
                    accepted = {'schema_version': 'phase7_conj_drop_v1', 'episode_id': id,
                        'source_id': example.id, 'split': split, 'question': example.question,
                        'question_hash': example.question_hash, 'genre': example.genre,
                        'level': desired, 'seed': seed, 'source_text': source.text,
                        'corrupted_text': changed.text, 'source_hash': example.essay_hash,
                        'corrupted_hash': sha_text(changed.text), 'source_layout': source.snapshot(),
                        'corrupted_layout': changed.snapshot(), 'records': records,
                        'compact_view': view, 'compact_tokens': tokens,
                        'q_source': score['mean'], 'q_corrupted': None,
                        'preexisting_spell_spans': [], 'spelling_detection': 'not_run_optional'}
                    break
                except ValueError as exc:
                    failures[str(exc)] += 1
            if accepted:
                rows.append(accepted)
                if len(rows) % 20 == 0:
                    print(f'Candidates {split}: {len(rows)}/{count}', flush=True)
            if len(rows) == count:
                break
        write_jsonl(path, rows)
        write_json(root/f'build_{split}.json', {'seed': seed, 'requested': count,
            'inventory': inventory(rows), 'attempts': dict(attempts), 'failures': dict(failures),
            'candidate_sha256': file_sha(path), 'source_rules': 'Section 4.3, frozen scored sources',
            'composition': 'alternating L1 (drop only), L2 (drop + WORD + TEXT on distinct sentences)',
            'lexicon': 'unchanged Phase 2c closed set; 실제로 is not in that set'})
        if len(rows) != count:
            raise ValueError(f'Only {len(rows)} eligible {split} sources, requested {count}')


def bounded_map(items, function, *, workers=4):
    """Stop dispatch after an error; preserve completed work in each function."""
    errors = []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        iterator, pending = iter(items), {}
        def fill():
            while len(pending) < workers and not errors:
                item = next(iterator, None)
                if item is None:
                    break
                pending[pool.submit(function, item)] = item
        fill()
        while pending:
            done, _ = wait(pending, timeout=30, return_when=FIRST_COMPLETED)
            for future in done:
                item = pending.pop(future)
                try:
                    future.result()
                except Exception as exc:
                    errors.append({'item': item.get('episode_id') if isinstance(item, dict) else item,
                                   'type': type(exc).__name__, 'message': str(exc)})
            fill()
    return errors


def judge(config, api):
    root = config['paths'][PHASE+'_output']
    rows = [r for split in ('agent_train', 'agent_dev') for r in read_jsonl(root/f'candidates_{split}.jsonl')]
    if len(rows) != 380:
        raise ValueError('Freeze the complete 300/80 candidate pool first')
    def one(row):
        path = root/'qc'/(row['episode_id'].replace(':', '_')+'.json')
        if path.exists():
            return read_json(path)
        response = api.request([{'role': 'system', 'content': QC_PROMPT},
            {'role': 'user', 'content': json.dumps(qc_payload(row), ensure_ascii=False, sort_keys=True)}],
            stage='conj_drop_qc', item_id=row['episode_id'], effort='high', max_output=4096,
            schema=QCResponse.model_json_schema())
        parsed = QCResponse.model_validate_json(response['raw']).model_dump()
        validate_judgments(row, parsed)
        result = {'episode_id': row['episode_id'], 'judgments': parsed['judgments'],
            'kept': all(j['damage_real'] and j['original_is_fix'] for j in parsed['judgments']),
            'candidate_hash': sha_text(json.dumps(row, ensure_ascii=False, sort_keys=True)),
            'phase_call': response['phase_call'], 'cost': response['cost'], 'label': 'LLM-verified'}
        write_json(path, result)
        print(json.dumps({'qc': row['episode_id'], 'kept': result['kept'], 'budget': api.accounting()}, ensure_ascii=False), flush=True)
        return result
    errors = bounded_map(rows, one)
    write_json(root/'qc_status.json', {'errors': errors, 'budget': api.accounting()})
    if errors:
        raise RuntimeError('QC stopped; see qc_status.json')


def finalize(config):
    from ..score.kanana import KananaScorer
    root = config['paths'][PHASE+'_output']
    target = config['paths']['metadata']/'corrupt_scope_v2'
    manifest = {}
    scorer = None
    try:
        for split in ('agent_train', 'agent_dev'):
            previous = config['paths']['previous_corrupt']/f'{split}.jsonl'
            old = read_jsonl(previous)
            retained = without_deletions(old)
            candidates = read_jsonl(root/f'candidates_{split}.jsonl')
            kept, judgments = [], []
            for row in candidates:
                qc = read_json(root/'qc'/(row['episode_id'].replace(':', '_')+'.json'))
                if qc['candidate_hash'] != sha_text(json.dumps(row, ensure_ascii=False, sort_keys=True)):
                    raise ValueError('QC candidate mismatch')
                judgments.append((row, qc))
                if not qc['kept']:
                    continue
                score_path = root/'corrupted_scores'/(row['episode_id'].replace(':', '_')+'.json')
                if score_path.exists():
                    score = read_json(score_path)
                else:
                    if scorer is None:
                        scorer = KananaScorer(config)
                    score = scorer.score(row['question'], row['corrupted_text']).to_dict()
                    write_json(score_path, score)
                kept.append({**row, 'q_corrupted': score['mean'], 'corrupted_score': score,
                             'instance_qc': qc})
            final = retained + kept
            write_jsonl(target/f'{split}.jsonl', final)
            write_jsonl(root/f'kept_new_{split}.jsonl', kept)
            by_op = {}
            for row, qc in judgments:
                ops = {r['record_id']: r['op'] for r in row['records']}
                for j in qc['judgments']:
                    out = by_op.setdefault(ops[j['record_id']], Counter())
                    out['judged'] += 1
                    out['damage_real'] += j['damage_real']
                    out['original_is_fix'] += j['original_is_fix']
                    out['both'] += j['damage_real'] and j['original_is_fix']
            manifest[split] = {'before': inventory(old), 'after_removal': inventory(retained),
                'candidates': inventory(candidates), 'kept_new': inventory(kept), 'final': inventory(final),
                'removed_ids': [r['episode_id'] for r in old if r not in retained],
                'qc_by_operator': {k: dict(v) for k, v in by_op.items()},
                'previous_path': str(previous), 'previous_sha256': file_sha(previous),
                'final_path': str(target/f'{split}.jsonl'), 'final_sha256': file_sha(target/f'{split}.jsonl')}
            print(f'Corpus {split}: {len(old)} -> {len(retained)} + {len(kept)} = {len(final)}', flush=True)
    finally:
        if scorer is not None:
            scorer.close()
    write_json(root/'corpus_manifest.json', manifest)
    return manifest


def audit(config):
    """Source/split/provenance audit using the historical hash index, never train/test."""
    from ..view_data import load_episode_examples
    from ..data_policy import assert_data_tree_clean
    root = config['paths'][PHASE+'_output']
    manifest = read_json(root/'corpus_manifest.json')
    questions, results = {}, {}
    for split in ('agent_train', 'agent_dev'):
        path = config['paths']['active_corrupt']/f'{split}.jsonl'
        rows = read_jsonl(path)
        examples = {e.id: e for e in load_episode_examples(config, split)}
        previous = read_jsonl(config['paths']['previous_corrupt']/f'{split}.jsonl')
        retained = without_deletions(previous)
        if rows[:len(retained)] != retained:
            raise ValueError('Previously retained rows were changed')
        if file_sha(config['paths']['previous_corrupt']/f'{split}.jsonl') != manifest[split]['previous_sha256']:
            raise ValueError('Previous corpus changed')
        if len({r['episode_id'] for r in rows}) != len(rows):
            raise ValueError('Duplicate active episode ID')
        for row in rows:
            source = examples[row['source_id']]
            if row['source_text'] != source.text or row['question'] != source.question or row['question_hash'] != source.question_hash:
                raise ValueError('Source provenance mismatch')
            if row['split'] != split or any(r['op'] == 'G_DELETE_SUPPORT' for r in row['records']):
                raise ValueError('Wrong split or removed operator')
            if row['corrupted_hash'] != sha_text(row['corrupted_text']) or row['compact_tokens'] > 3000:
                raise ValueError('Invalid corruption hash/view budget')
            score = row['corrupted_score']
            if len(score['expected']) != 8 or row['q_corrupted'] != score['mean'] or not 1 <= score['mean'] <= 9:
                raise ValueError('Missing expected-score corruption Q')
            for record in row['records']:
                if record['op'] == 'G_OFFTOPIC' and record['params']['donor']['source_id'] not in examples:
                    raise ValueError('Wrong-split donor')
            if row['episode_id'].startswith('phase7drop:'):
                qc = row['instance_qc']
                validate_judgments(row, qc)
                if not qc['kept'] or not all(j['damage_real'] and j['original_is_fix'] for j in qc['judgments']):
                    raise ValueError('QC-rejected new essay in corpus')
        questions[split] = {r['question_hash'] for r in rows}
        results[split] = {'essays': len(rows), 'sha256': file_sha(path), 'prior_rows_unchanged': len(retained),
                          'new_passed_rows': len(rows)-len(retained)}
    if questions['agent_train'] & questions['agent_dev']:
        raise ValueError('Question leakage')
    deny = set(read_json(config['paths']['metadata']/'audit_index.json')['train']['essay_hashes'])
    assert_data_tree_clean(config['paths']['active_corrupt'], deny)
    result = {'passed': True, 'splits': results, 'question_overlap': 0, 'training_essay_matches': 0,
              'raw_train_test_opened': False, 'new_api_calls': 0}
    write_json(root/'data_validation.json', result)
    return result
