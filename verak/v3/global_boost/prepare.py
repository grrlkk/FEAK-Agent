"""Exact Phase-3b source eligibility, proposal generation and restoration checks."""
from collections import Counter
import json
import random

from ..agent.runner import system_prompt
from ..common import file_sha, read_json, sha_text, write_json
from ..corrupt.builder import source_coupled_changes
from ..corrupt.document import BareunBank, source_document
from ..corrupt.instance_policy import candidates
from ..corrupt.operators import apply, exact_restoration_satisfies, restore_record
from ..corrupt.sources import select_sources
from ..ko import render
from ..phase2 import read_jsonl
from ..train.teacher_bulk import atomic_new
from ..v2_ops.data import excluded_sources
from .config import PHASE, OPERATORS


def safe_id(value):
    return value.replace(':', '_')


def candidate_path(root, operator, source_id):
    return root / 'candidates' / operator / (safe_id(source_id) + '.json')


def source_inventory(config):
    examples, source_policy = select_sources(config, 'agent_train')
    scores_path = config['paths']['phase3_output'] / 'sources_agent_train.json'
    scores = read_json(scores_path)['rows']
    active_paths = [config['paths']['active_corrupt'] / f'{split}.jsonl'
                    for split in ('agent_train', 'agent_dev')]
    active = [row for path in active_paths for row in read_jsonl(path)]
    active_ids, active_hashes = {r['source_id'] for r in active}, {r['source_hash'] for r in active}
    excluded, evaluation = excluded_sources(config)
    excluded_hashes = {r['source_hash'] for r in excluded.values()}
    counts, pool = Counter(), []
    for example in examples:
        counts['phase3b_train_source_pool'] += 1
        if example.id in active_ids or example.essay_hash in active_hashes:
            counts['active_corpus_source_excluded'] += 1
            continue
        if example.id in excluded or example.essay_hash in excluded_hashes:
            counts['evaluation_source_excluded'] += 1
            continue
        score = scores[example.id]
        if score['essay_hash'] != example.essay_hash:
            raise ValueError('Frozen Phase3 source hash changed')
        if not score['eligible']:
            counts['failed_source_score_excluded'] += 1
            continue
        pool.append((example, score['score']))
    counts['unused_eligible_sources'] = len(pool)
    return pool, {'counts': dict(counts), 'source_policy': source_policy,
        'source_scores': {'path': str(scores_path), 'sha256': file_sha(scores_path)},
        'active_corpora': {str(p): file_sha(p) for p in active_paths},
        'evaluation': evaluation, 'excluded_evaluation_sources': excluded,
        'active_source_ids': sorted(active_ids), 'active_source_hashes': sorted(active_hashes)}


def make_candidate(config, example, source, score, operator, bank, tokenizer):
    episode_id = f'global_boost:{operator}:agent_train:{example.source_line}:v1'
    seed = config[PHASE]['seed'] + example.source_line * 1000 + OPERATORS.index(operator)
    proposals = candidates(source, operator, donors=(), vague_cache={}, question_hash=example.question_hash)
    random.Random(seed).shuffle(proposals)
    failures = Counter()
    # Phase3b uses the first eight shuffled proposals per composition attempt.
    # There is one pre-QC composition per source/operator in this expansion.
    for proposal in proposals[:8]:
        try:
            changed, record = apply(source, proposal, bank)
            restored = restore_record(changed, record, bank)
            if restored.text != source.text or not exact_restoration_satisfies(restored, record):
                raise ValueError('Exact restoration failed')
            view = render(changed.structure(), compact=True)
            tokens = len(tokenizer.encode(view, add_special_tokens=False))
            if tokens > 3000:
                raise ValueError('corrupted_compact_view_exceeds_3000')
            record['record_id'] = episode_id + ':R1'
            record['params']['seed'] = seed
            record['coupled_changes'] = source_coupled_changes(record['coupled_changes'], source.structure())
            return {'schema_version': 'phase3b_global_boost_v1', 'method_version': 'v1',
                'episode_id': episode_id, 'source_id': example.id, 'split': 'agent_train',
                'question': example.question, 'question_hash': example.question_hash,
                'genre': example.genre, 'level': 'L3', 'operator': operator, 'seed': seed,
                'source_text': source.text, 'source_hash': example.essay_hash,
                'corrupted_text': changed.text, 'corrupted_hash': sha_text(changed.text),
                'source_layout': source.snapshot(), 'corrupted_layout': changed.snapshot(),
                'records': [record], 'compact_view': view, 'compact_tokens': tokens,
                'q_source': score['mean'], 'q_corrupted': None,
                'preexisting_spell_spans': [], 'spelling_detection': 'not_run_optional',
                'supervision_private': ['source_text', 'source_layout', 'records', 'q_source'],
                'generation_rejections': dict(failures)}, None
        except ValueError as exc:
            failures[str(exc)] += 1
    return None, {'source_id': example.id, 'operator': operator, 'proposals': len(proposals),
                  'reason': 'no_mechanically_eligible_candidate', 'failures': dict(failures)}


def prepare(config):
    from transformers import AutoTokenizer
    root = config['paths'][PHASE + '_output']
    root.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    bank = BareunBank(config, cache_dir=root / 'bareun_units')
    pool, audit = source_inventory(config)
    random.Random(config[PHASE]['seed']).shuffle(pool)
    chosen, failures = {op: [] for op in OPERATORS}, {op: [] for op in OPERATORS}
    documents = {}
    for operator in OPERATORS:
        for example, score in pool:
            source = documents.setdefault(example.id, source_document(config, example, bank))
            row, error = make_candidate(config, example, source, score, operator, bank, tokenizer)
            if error:
                failures[operator].append(error)
                continue
            path = candidate_path(root, operator, example.id)
            if path.exists():
                if read_json(path) != row:
                    raise ValueError('Immutable boost candidate changed')
            else:
                atomic_new(path, row)
            chosen[operator].append({'episode_id': row['episode_id'], 'source_id': example.id,
                'source_hash': example.essay_hash, 'path': str(path), 'sha256': file_sha(path)})
            if len(chosen[operator]) == config[PHASE]['per_operator']:
                break
    plan = {'phase': PHASE, 'version': 'v1', 'budget_usd': 12,
        'source_policy': 'Exact Phase3b genre q75/500–2500-character and view-eligible agent_train sources; exclude active corpus sources by ID and hash.',
        'operators': list(OPERATORS), 'requested_per_operator': config[PHASE]['per_operator'],
        'records_per_essay': 1, 'candidates': chosen, 'mechanical_exclusions': failures,
        'counts': {op: len(chosen[op]) for op in OPERATORS}, 'source_inventory': audit,
        'source_overlap_across_operators': sorted({r['source_id'] for r in chosen[OPERATORS[0]]}
                                                & {r['source_id'] for r in chosen[OPERATORS[1]]}),
        'shortfall': {op: config[PHASE]['per_operator'] - len(chosen[op]) for op in OPERATORS},
        'prompt_sha256': {r: sha_text(system_prompt(r)) for r in ('global', 'korean')},
        'contract_sha256': sha_text(json.dumps({k: config[k] for k in ('env', 'reward', 'policy', 'similarity')}, sort_keys=True)),
        'gpu_used': False, 'training': False, 'new_bareun_calls': 0}
    target = root / 'source_plan.json'
    if target.exists() and read_json(target) != plan:
        raise ValueError('Frozen GLOBAL boost source plan changed')
    write_json(target, plan)
    return plan


def corpus(config):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    result = {}
    for entries in plan['candidates'].values():
        for entry in entries:
            if file_sha(entry['path']) != entry['sha256']:
                raise ValueError('Candidate bytes changed after preparation')
            result[entry['episode_id']] = read_json(entry['path'])
    return result
