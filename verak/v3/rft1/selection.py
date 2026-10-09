"""Explicit round-1 role eligibility, teacher merge, source holdout and reweighting."""
from collections import Counter
from copy import deepcopy
import gzip
import json
import math
from pathlib import Path

from ..common import file_sha, read_json, sha_text, write_json
from ..train.formatting import formatted_turns, lengths
from ..train.sft_composition import structural_attempt
from ..train.teacher_bulk import atomic_new
from .config import PHASE, ROLES
from .rollout import episode_path, prepare


def role_reward(row, role):
    return (row.get('reward') or {}).get(role) or (row.get('global_only_reward') if role == 'global' else None)


def structural_action(action):
    # A rejected malformed argument container must not crash post-run selection.
    # Preserve the raw action/target in the export; this is only its category.
    value = action if isinstance(action.get('args'), dict) else {**action, 'args': {}}
    return structural_attempt(value)


def eligible(row, role, corpus):
    reward = role_reward(row, role)
    actions = row.get('actions_by_role', {}).get(role, [])
    no_global = role == 'global' and not any(r['level'] == 'GLOBAL' for r in corpus['records'])
    if reward is None or (not no_global and reward['R'] < .8):
        return False, 'role_R_below_0.80_or_missing'
    if row.get('termination', {}).get(role) != 'STOP':
        return False, 'role_did_not_STOP'
    if sum(not a.get('valid', False) for a in actions) > 1:
        return False, 'more_than_one_rejected_action'
    if not actions or actions[-1]['action'] != 'STOP' or not actions[-1].get('valid'):
        return False, 'no_valid_terminal_STOP'
    if role == 'korean' and not row.get('completed'):
        return False, 'incomplete_korean'
    if no_global:
        if (row['steps']['global'] > 2 or reward['R_over'] != 0 or any(structural_action(a) for a in actions)):
            return False, 'no_GLOBAL_STOP_rule'
    return True, 'eligible'


def stop_only(row, role):
    own = row['actions_by_role'][role]
    return bool(own) and all(a['action'] == 'STOP' for a in own)


def operator_evidence(row, role, corpus):
    reward = role_reward(row, role)
    actual = {r['record_id']: r['main'] for r in reward['per_record']}
    records = [r for r in corpus['records'] if (r['level'] == 'GLOBAL') == (role == 'global')]
    evidence = {}
    for op in sorted({r['op'] for r in records}):
        own = [r for r in records if r['op'] == op]
        if any(r['record_id'] not in actual for r in own):
            raise ValueError('Selected role lacks record recovery evidence')
        evidence[op] = {'records': len(own), 'fully_recovered_records': sum(actual[r['record_id']] >= 1-1e-12 for r in own),
                        'all_fully_recovered': all(actual[r['record_id']] >= 1-1e-12 for r in own)}
    return evidence


def rebalance(entries, role):
    """Cap STOP-only; duplicate weak GLOBAL operators only below200 recoveries."""
    values = deepcopy(entries)
    removed = []
    if role == 'global':
        stops = sorted((e for e in values if e['STOP_only']),
                       key=lambda e: (-e['R'], sha_text('83:' + e['episode_id']), e['episode_id']))
        active = [e for e in values if not e['STOP_only']]
        allowed = math.floor(len(active) * .35 / .65)
        removed = [e['episode_id'] for e in stops[allowed:]]
        values = active + stops[:allowed]
    wanted = {'G_PARA_SWAP', 'G_SENT_MOVE'} if role == 'global' else {'L_CONJ'}
    if role == 'global':
        wanted = {op for op in wanted if sum(e['operators'].get(op, {}).get('all_fully_recovered', False)
                                             for e in values) < 200}
    for e in values:
        e['duplicated_for'] = sorted(op for op in wanted if e['operators'].get(op, {}).get('all_fully_recovered'))
        e['weight'] = 2 if e['duplicated_for'] else 1
    return sorted(values, key=lambda e: e['episode_id']), removed


def counts(entries):
    n, weighted = len(entries), sum(e.get('weight', 1) for e in entries)
    stops = sum(e['STOP_only'] for e in entries)
    operators = {}
    for op in sorted({o for e in entries for o in e['operators']}):
        own = [e for e in entries if op in e['operators']]
        operators[op] = {'unique_trajectories': len(own), 'weighted_trajectories': sum(e.get('weight', 1) for e in own),
            'fully_recovered_unique': sum(e['operators'][op]['all_fully_recovered'] for e in own),
            'fully_recovered_weighted': sum(e.get('weight', 1) for e in own if e['operators'][op]['all_fully_recovered']),
            'by_source': {source: {'unique_trajectories': len(group),
                'weighted_trajectories': sum(e.get('weight', 1) for e in group),
                'fully_recovered_unique': sum(e['operators'][op]['all_fully_recovered'] for e in group)}
                for source in sorted({e['origin'] for e in own})
                for group in [[e for e in own if e['origin'] == source]]}}
    return {'unique_trajectories': n, 'weighted_trajectories': weighted, 'STOP_only': stops,
        'STOP_only_share_unique': stops/n if n else None,
        'STOP_only_share_weighted': sum(e.get('weight',1) for e in entries if e['STOP_only'])/weighted if weighted else None,
        'origins': dict(Counter(e['origin'] for e in entries)), 'operators': operators}


def extra_global(config, active_corpus, held):
    """Only the frozen, GPU-only v1 addition available before training may merge."""
    root = config['paths']['repo'] / 'verak/v3/outputs/data_boost'
    merge_path = root / 'rft1_extra_merge.json'
    if not merge_path.exists():
        return [], {'included': False, 'reason': 'no_extra_merge_authorization'}
    merge = read_json(merge_path)
    if not merge['include_extra_global']:
        return [], {'included': False, 'reason': merge['reason'], 'merge_sha256': file_sha(merge_path)}
    path = Path(merge['selection_path'])
    if file_sha(path) != merge['selection_sha256']:
        raise ValueError('GPU-rescored extra selection changed after the RFT boundary')
    selection = read_json(path)
    if (selection.get('version') != 'v1' or selection.get('role') != 'global'
            or selection.get('score_source') != 'gpu_reference'):
        raise ValueError('Only GPU-reference v1 GLOBAL additions may enter RFT1')
    complete = read_json(root / 'gpu_rescore/global_complete.json')
    if (complete['slot'] != 'pre_rft_training' or complete['fingerprint'] != selection['fingerprint']
            or complete['manifest_sha256'] != selection['manifest_sha256']):
        raise ValueError('Extra selection lacks the pre-training GPU reference evidence')
    entries, excluded = [], Counter()
    active_sources = {row['source_id'] for row in active_corpus.values()}
    for eid, selected in selection['selected'].items():
        candidate = selected['candidate']
        if (eid in active_corpus or candidate['source_id'] in active_sources
                or candidate.get('split') != 'agent_train'
                or any(r['level'] != 'GLOBAL' or r['op'] not in {'G_PARA_SWAP', 'G_SENT_MOVE'}
                       for r in candidate['records']) or len(candidate['records']) != 1):
            raise ValueError('Extra teacher data violates unused-source, role, split or operator constraints')
        trajectory_path = Path(selected['path'])
        if file_sha(trajectory_path) != selected['sha256']:
            raise ValueError('GPU-rescored extra trajectory changed')
        row = read_json(trajectory_path)
        if row.get('score_source') != 'gpu_reference':
            raise ValueError('CPU-scored extra reward cannot enter RFT training')
        if candidate['source_id'] in held:
            excluded['held_out_source'] += 1
            continue
        keep, reason = eligible(row, 'global', candidate)
        if not keep:
            excluded[reason] += 1
            continue
        entries.append({'episode_id': eid, 'source_id': candidate['source_id'], 'role': 'global',
            'origin': 'extra_teacher', 'sample': selected['attempt'], 'path': str(trajectory_path),
            'sha256': selected['sha256'], 'R': role_reward(row, 'global')['R'],
            'STOP_only': stop_only(row, 'global'), 'operators': operator_evidence(row, 'global', candidate),
            'partition': 'train', 'score_source': 'gpu_reference',
            'global_only_reward': row.get('global_only_reward') if not row.get('reward') else None})
    return entries, {'included': bool(entries), 'selected': len(selection['selected']), 'merged': len(entries),
        'excluded': dict(excluded), 'selection_sha256': file_sha(path), 'merge_sha256': file_sha(merge_path),
        'gpu_complete_sha256': file_sha(root / 'gpu_rescore/global_complete.json'),
        'selection_changes': selection.get('selection_changes'),
        'interpretation': 'Gains combine student rollouts and extra teacher data; no v2 data was merged.'}


def select(config):
    root = config['paths'][PHASE + '_output']
    design, corpus = prepare(config)
    if not read_json(root / 'rollout_status.json')['sampling_complete']:
        raise RuntimeError('Selection requires all 5,720 saved attempts')
    sft = read_json(config['paths']['phase7_sft_output'] / 'data/manifest.json')
    held = set().union(*(set(sft['contract']['split']['roles'][role]['validation_sources']) for role in ROLES))
    teacher_path = config['paths']['repo'] / 'verak/v3/outputs/teacher_bulk_two_stage/best_role_trajectories.json'
    teacher = read_json(teacher_path)
    partial_path = config['paths']['phase7_teacher_output'] / 'completed_global_evaluation.json'
    partial = read_json(partial_path)['results']
    best = {r: {} for r in ROLES}
    reasons = {r: Counter() for r in ROLES}
    slots, rollout_best = [], {r: {} for r in ROLES}

    def entry_for(row, path, role, eid, origin, sample):
        return {'episode_id': eid, 'source_id': corpus[eid]['source_id'], 'role': role, 'origin': origin,
            'score_source': 'gpu_reference',
            'sample': sample, 'path': str(path), 'sha256': file_sha(path), 'R': role_reward(row, role)['R'],
            'STOP_only': stop_only(row, role), 'operators': operator_evidence(row, role, corpus[eid]),
            'partition': 'validation' if corpus[eid]['source_id'] in held else 'train',
            'global_only_reward': row.get('global_only_reward') if not row.get('reward') else None}

    for eid in design['order']:
        for sample in range(1, 5):
            path = episode_path(root, sample, eid)
            row = read_json(path)
            if row['rft1']['design_sha256'] != file_sha(root / 'rollout_design.json'):
                raise ValueError('Rollout provenance changed')
            slot = {'episode_id': eid, 'sample': sample, 'completed': row['completed'], 'eligibility': {}}
            for role in ROLES:
                keep, reason = eligible(row, role, corpus[eid])
                reasons[role]['rollout:' + reason] += 1
                slot['eligibility'][role] = reason
                if keep:
                    entry = entry_for(row, path, role, eid, 'rft_rollout', sample)
                    old = rollout_best[role].get(eid)
                    if old is None or (entry['R'], -sample) > (old['R'], -old['sample']):
                        rollout_best[role][eid] = entry
            slots.append(slot)
    best = deepcopy(rollout_best)
    for role in ROLES:
        for eid, original in teacher[role].items():
            path = Path(original['path'])
            if file_sha(path) != original['sha256']:
                raise ValueError('Accepted teacher selection changed')
            row = read_json(path)
            if role == 'global' and role_reward(row, role) is None:
                row['global_only_reward'] = partial.get('luna_low:' + eid, {}).get('global_only_reward')
            keep, reason = eligible(row, role, corpus[eid])
            reasons[role]['teacher:' + reason] += 1
            if keep:
                entry = entry_for(row, path, role, eid, 'sft_teacher', original['attempt'])
                old = best[role].get(eid)
                if old is None or entry['R'] >= old['R']:  # Teacher wins a reward tie.
                    best[role][eid] = entry
    additions, addition_provenance = extra_global(config, corpus, held)
    for entry in additions:
        if entry['episode_id'] in best['global']:
            raise ValueError('An extra teacher practice collides with the original corpus')
        best['global'][entry['episode_id']] = entry
    selections, summaries, dropped = {}, {}, {}
    for role in ROLES:
        merged = list(best[role].values())
        train, removed = rebalance([e for e in merged if e['partition'] == 'train'], role)
        validation = [{**e, 'weight': 1, 'duplicated_for': []} for e in merged if e['partition'] == 'validation']
        selections[role] = {'train': train, 'validation': sorted(validation, key=lambda e: e['episode_id'])}
        dropped[role] = removed
        summaries[role] = {'rollout_best': counts(list(rollout_best[role].values())), 'merged': counts(merged),
                           'train': counts(train), 'validation': counts(validation)}
        if not train:
            raise RuntimeError('No eligible training trajectories for ' + role)
        if role == 'global' and (summaries[role]['train']['STOP_only_share_unique'] > .35 or
                                summaries[role]['train']['STOP_only_share_weighted'] > .35):
            raise AssertionError('STOP-only cap violated')
    result = {'rollout_design_sha256': file_sha(root / 'rollout_design.json'),
        'teacher_selection_sha256': file_sha(teacher_path), 'sft_manifest_sha256': file_sha(config['paths']['phase7_sft_output'] / 'data/manifest.json'),
        'selection_rule': 'role R>=.80 except existing no-GLOBAL STOP-within2/no-structural/R_over0 rule; all valid terminal STOP and <=1 rejected executed action',
        'no_GLOBAL_exception': 'explicitly confirmed by user on 2026-10-09; applies to teacher and rollout selections',
        'teacher_gate': 'explicit current RFT criteria also applied to accepted teacher selections',
        'merge_tie': 'teacher; rollout ties lower sample ID', 'holdout_sources': sorted(held),
        'rebalancing': 'cap unique train GLOBAL STOP-only at35%; weight2 for full G_PARA_SWAP/G_SENT_MOVE only when merged train full-recovery count is below200 per operator; full L_CONJ remains weight2; validation unchanged',
        'extra_teacher': addition_provenance,
        'selections': selections, 'summaries': summaries, 'eligibility_reasons': reasons,
        'STOP_only_removed_ids': dropped, 'rollout_slots': slots}
    destination = root / 'selection.json'
    if destination.exists():
        if read_json(destination) != result:
            raise ValueError('Frozen RFT selection changed')
    else:
        atomic_new(destination, result)
    return result


def export(config):
    from transformers import AutoTokenizer
    root = config['paths'][PHASE + '_output']
    selection = select(config)
    output = root / 'data'
    output.mkdir(exist_ok=True)
    target = output / 'manifest.json'
    selection_sha = file_sha(root / 'selection.json')
    if target.exists():
        result = read_json(target)
        if result['selection_sha256'] != selection_sha:
            raise ValueError('RFT export selection changed')
        for parts in result['roles'].values():
            for entry in parts.values():
                if file_sha(entry['path']) != entry['sha256']:
                    raise ValueError('RFT export changed')
        return result
    tokenizer = AutoTokenizer.from_pretrained(str(config['paths']['policy_base']), local_files_only=True)
    result = {'selection_sha256': selection_sha, 'roles': {}, 'loss': 'inference-exact current action target only',
              'context': 8192, 'packing': False, 'holdout_sources': selection['holdout_sources']}
    for role in ROLES:
        result['roles'][role] = {}
        for part, entries in selection['selections'][role].items():
            path = output / f'{role}.{part}.jsonl.gz'
            sizes, targets, checks = [], [], 0
            with gzip.open(path, 'wt', encoding='utf-8') as stream:
                for entry in entries:
                    if file_sha(entry['path']) != entry['sha256']:
                        raise ValueError('Selected trajectory changed')
                    row = read_json(entry['path'])
                    samples = list(formatted_turns(row, role, tokenizer, 8192))
                    calls = [c for c in row['calls'] if c['role'] == role]
                    for sample, call in zip(samples, calls):
                        if sample['messages'][:-1] != call['messages'] or sample['messages'][-1]['content'] != call['raw']:
                            raise ValueError('RFT action context differs from actual sampling context')
                        n = sample['prompt_tokens']
                        if any(x != -100 for x in sample['labels'][:n]) or sample['labels'][n:] != sample['input_ids'][n:]:
                            raise ValueError('Action-only masking failed')
                        checks += 1
                    if not samples or len(samples) != len(calls):
                        raise ValueError('Every selected model call must have a target')
                    for duplicate in range(entry['weight']):
                        for sample in samples:
                            sample.update(source_id=entry['source_id'], source_path=entry['path'],
                                source_sha256=entry['sha256'], origin=entry['origin'], sample=entry['sample'],
                                score_source=entry['score_source'],
                                duplicate_index=duplicate, partition=part)
                            stream.write(json.dumps(sample, ensure_ascii=False) + '\n')
                            sizes.append(sample['total_tokens']); targets.append(sample['loss_tokens'])
            result['roles'][role][part] = {'path': str(path), 'sha256': file_sha(path),
                'trajectories': len(entries), 'weighted_trajectories': sum(e['weight'] for e in entries),
                'sources': len({e['source_id'] for e in entries}), 'turns': len(sizes),
                'tokens': sum(sizes), 'target_tokens': sum(targets), 'lengths': lengths(sizes),
                'mask_context_checks': checks}
    atomic_new(target, result)
    return result
