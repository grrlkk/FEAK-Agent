"""Source-diverse Phase3b practices with proof of new structural positions."""
from collections import Counter
from copy import deepcopy
import json
import random
from pathlib import Path

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
from .expansion import batch_configs, batch_config, root_for
from .prepare import corpus, safe_id


def position_payload(source_id, source_hash, record):
    p = record['params']
    if record['op'] == 'G_PARA_SWAP':
        position = {'paragraph_indices': sorted(p['paragraph_indices'])}
    elif record['op'] == 'G_SENT_MOVE':
        position = {'sentence_id': record['sids'][0], **{k: p[k] for k in
            ('from_paragraph', 'from_position', 'to_paragraph', 'to_position')}}
    else:
        raise ValueError('Only structural boost operators have this position contract')
    return {'source_id': source_id, 'source_hash': source_hash, 'operator': record['op'], 'position': position}


def position_signature(source_id, source_hash, record):
    return sha_text(json.dumps(position_payload(source_id, source_hash, record), sort_keys=True, ensure_ascii=False))


def holdouts(config):
    path = config['paths']['repo'] / 'verak/v3/outputs/phase7_sft/data/manifest.json'
    roles = read_json(path)['contract']['split']['roles']
    return set().union(*(set(r['validation_sources']) for r in roles.values())), path


def prior_index(config):
    root = root_for(config)
    path = root / 'v4/prior_positions.json'
    if path.exists():
        return read_json(path)
    base = config['paths']['repo'] / 'verak/v3'
    paths = set(base.glob('data/corrupt*/agent_train.jsonl'))
    paths.update((base / 'outputs/phase3b').rglob('*agent_train.jsonl'))
    signatures, inputs = {}, {}
    for p in sorted(paths):
        inputs[str(p)] = file_sha(p)
        for row in read_jsonl(p):
            for record in row.get('records', []):
                if record['op'] in OPERATORS:
                    key = position_signature(row['source_id'], row['source_hash'], record)
                    signatures.setdefault(key, {'source_id': row['source_id'], 'operator': record['op'],
                        'origin': str(p), 'episode_id': row['episode_id']})
    for cfg in batch_configs(config):
        for row in corpus(cfg).values():
            for record in row['records']:
                key = position_signature(row['source_id'], row['source_hash'], record)
                signatures.setdefault(key, {'source_id': row['source_id'], 'operator': record['op'],
                    'origin': str(cfg['paths'][PHASE+'_output']), 'episode_id': row['episode_id']})
    value = {'schema_version': 1, 'signatures': signatures, 'corpus_sha256': inputs,
        'definition': 'SHA256 of sorted ensure_ascii=False JSON: source_id,source_hash,operator,position; seed/span omitted.',
        'paid_calls': 0, 'gpu_calls': 0}
    atomic_new(path, value)
    return value


def prepare(config):
    from transformers import AutoTokenizer
    root = root_for(config)
    target = root / 'v4/plan.json'
    if target.exists():
        return read_json(target)
    if not (root / 'v4/legacy_drain.json').exists():
        raise RuntimeError('Drain legacy concentrated collection before freezing the new pool')
    index = prior_index(config)
    index_path = root / 'v4/prior_positions.json'
    seen = set(index['signatures'])
    held, held_path = holdouts(config)
    examples, policy = select_sources(config, 'agent_train')
    scores_path = config['paths']['phase3_output'] / 'sources_agent_train.json'
    scores = read_json(scores_path)['rows']
    excluded, evaluation = excluded_sources(config)
    excluded_hashes = {r['source_hash'] for r in excluded.values()}
    active = read_jsonl(config['paths']['active_corrupt'] / 'agent_train.jsonl')
    active_ids = {r['source_id'] for r in active}
    historical = [row for cfg in batch_configs(config) for row in corpus(cfg).values()]
    used = Counter((r['source_id'], r['operator']) for r in historical)
    old_capped = {op: sum(min(4,n) for (source,o),n in used.items() if o == op) for op in OPERATORS}
    needed = {op: max(0,400-old_capped[op]) for op in OPERATORS}
    pool, excluded_counts = [], Counter()
    for ex in examples:
        if ex.id in held:
            excluded_counts['sft_validation_source'] += 1
        elif ex.id in excluded or ex.essay_hash in excluded_hashes:
            excluded_counts['evaluation_source'] += 1
        elif not scores[ex.id]['eligible']:
            excluded_counts['frozen_source_score_failure'] += 1
        else:
            if scores[ex.id]['essay_hash'] != ex.essay_hash:
                raise ValueError('Source score identity changed')
            pool.append(ex)
    random.Random(107).shuffle(pool)
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    bank = BareunBank(config, cache_dir=root / 'bareun_units')
    documents, new_rows, failures = {}, {op: [] for op in OPERATORS}, Counter()
    for op in OPERATORS:
        # A previously unused source for this operator always precedes repeats.
        ordered = sorted(pool, key=lambda e: used[(e.id, op)])
        for ex in ordered:
            if len(new_rows[op]) >= needed[op]:
                break
            if used[(ex.id, op)] >= 4:
                continue
            if ex.id not in documents:
                documents[ex.id] = source_document(config, ex, bank)
            source = documents[ex.id]
            proposals = candidates(source, op, donors=(), vague_cache={}, question_hash=ex.question_hash)
            random.Random(107+ex.source_line*1000+OPERATORS.index(op)).shuffle(proposals)
            # Same Phase3b closed generator and per-composition proposal limit.
            proposals = [p for p in proposals if position_signature(ex.id, ex.essay_hash,
                {'op': op, 'sids': p.sids, 'params': p.params}) not in seen]
            for proposal in proposals[:8]:
                try:
                    changed, record = apply(source, proposal, bank)
                    restored = restore_record(changed, record, bank)
                    if restored.text != source.text or not exact_restoration_satisfies(restored, record):
                        raise ValueError('exact_inverse_failed')
                    view = render(changed.structure(), compact=True)
                    tokens = len(tokenizer.encode(view, add_special_tokens=False))
                    if tokens > 3000:
                        raise ValueError('compact_view_above_3000')
                    ordinal = len(new_rows[op])+1
                    eid = f'global_boost:batch_v4_{(ordinal-1)//25+1:03}:{op}:agent_train:{ex.source_line}:v1'
                    record['record_id'] = eid+':R1'
                    record['params']['seed'] = 107+ex.source_line*1000
                    record['coupled_changes'] = source_coupled_changes(record['coupled_changes'], source.structure())
                    signature = position_signature(ex.id, ex.essay_hash, record)
                    provenance = {'schema_version': 1, 'pool': 'phase3b_agent_train',
                        'active_corpus_source': ex.id in active_ids, 'source_id': ex.id,
                        'source_hash': ex.essay_hash, 'source_question_hash': ex.question_hash,
                        'position_signature': signature, 'position_payload': position_payload(ex.id, ex.essay_hash, record),
                        'prior_position_index_path': str(index_path), 'prior_position_index_sha256': file_sha(index_path),
                        'sft_holdout_manifest_sha256': file_sha(held_path)}
                    row = {'schema_version': 'phase3b_global_boost_v4prep', 'method_version': 'v1',
                        'episode_id': eid, 'source_id': ex.id, 'split': 'agent_train', 'question': ex.question,
                        'question_hash': ex.question_hash, 'source_question_hash': ex.question_hash,
                        'genre': ex.genre, 'level': 'L3', 'operator': op, 'seed': 107+ex.source_line*1000,
                        'source_text': source.text, 'source_hash': ex.essay_hash, 'source_layout': source.snapshot(),
                        'corrupted_text': changed.text, 'corrupted_hash': sha_text(changed.text),
                        'corrupted_layout': changed.snapshot(), 'records': [record], 'compact_view': view,
                        'compact_tokens': tokens, 'q_source': scores[ex.id]['score']['mean'], 'q_corrupted': None,
                        'preexisting_spell_spans': [], 'spelling_detection': 'not_run_optional',
                        'supervision_private': ['source_text','source_layout','records','q_source','source_provenance'],
                        'source_provenance': provenance, 'generation_rejections': {}}
                    batch = root / 'batches' / eid.split(':')[1]
                    path = batch / 'candidates' / op / (safe_id(eid)+'.json')
                    atomic_new(path, row)
                    new_rows[op].append({'episode_id': eid, 'source_id': ex.id, 'source_hash': ex.essay_hash,
                        'path': str(path), 'sha256': file_sha(path)})
                    seen.add(signature)
                    used[(ex.id,op)] += 1
                    break
                except ValueError as error:
                    failures[str(error)] += 1
            else:
                failures[op+':no_new_mechanical_position'] += 1
    parent = read_json(root / 'source_plan.json')
    output_batches = sorted({Path(e['path']).parents[2] for values in new_rows.values() for e in values})
    inventory = {'counts': {'phase3b_train_source_pool': len(examples), 'eligible_after_holdouts': len(pool),
        'unused_eligible_sources': len(pool), **excluded_counts}, 'source_scores': str(scores_path),
        'source_scores_sha256': file_sha(scores_path), 'evaluation': evaluation, 'sft_holdout_sources': sorted(held)}
    for batch in output_batches:
        entries = {op: [v for v in new_rows[op] if Path(v['path']).parents[2] == batch] for op in OPERATORS}
        counts = {op: len(entries[op]) for op in OPERATORS}
        plan = {'phase': PHASE, 'version': 'v1', 'task_version': 'v4_prep', 'batch': batch.name,
            'budget_usd': 40, 'shared_budget_root': str(root), 'records_per_essay': 1,
            'source_policy': 'Phase3b eligible agent_train, active sources allowed, new structural positions, SFT/evaluation sources excluded.',
            'source_inventory': inventory, 'operators': list(OPERATORS), 'candidates': entries, 'counts': counts,
            'shortfall': {op:0 for op in OPERATORS}, 'source_overlap_across_operators': sorted(
                {e['source_id'] for e in entries[OPERATORS[0]]} & {e['source_id'] for e in entries[OPERATORS[1]]}),
            'prompt_sha256': parent['prompt_sha256'], 'contract_sha256': parent['contract_sha256'],
            'gpu_used': False, 'training': False}
        atomic_new(batch / 'source_plan.json', plan)
    value = {'task_version': 'v4_prep', 'budget_usd_cumulative': 40, 'per_source_operator_cap': 4,
        'usable_practice_quota_per_operator': 400, 'old_raw_counts': dict(Counter(r['operator'] for r in historical)),
        'old_source_capped_counts': old_capped, 'new_counts': {op:len(v) for op,v in new_rows.items()},
        'new_distinct_sources': {op:len({r['source_id'] for r in v}) for op,v in new_rows.items()},
        'all_distinct_sources': {op:len({r['source_id'] for r in historical if r['operator']==op}
            | {r['source_id'] for r in new_rows[op]}) for op in OPERATORS},
        'requested_new': needed, 'source_policy': policy, 'source_inventory': inventory,
        'source_holdout_manifest': str(held_path), 'source_holdout_manifest_sha256': file_sha(held_path),
        'prior_position_index_path': str(index_path), 'prior_position_index_sha256': file_sha(index_path),
        'batch_roots': [str(b) for b in output_batches], 'mechanical_exclusions': dict(failures),
        'new_bareun_calls': 0, 'gpu_used': False, 'paid_calls': 0,
        'quota_interpretation': 'Archived historical excess is excluded from the 400/operator diversified pool count; final max4 ranking uses GPU R only.'}
    atomic_new(target, value)
    return value
