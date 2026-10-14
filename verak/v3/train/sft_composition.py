"""CPU-only diagnostics over immutable SFT selections and saved evaluations."""
from collections import Counter
import json
from pathlib import Path

from ..common import file_sha, load_config, read_json, write_json

OPERATORS = {
    'global': ('G_PARA_SWAP', 'G_SENT_MOVE', 'G_OFFTOPIC'),
    'korean': ('L_CONJ', 'L_CONN', 'L_REGISTER', 'L_SUBJ_INSERT', 'L_SPACING', 'L_POLARITY'),
}


def structural_attempt(action):
    """MOVE or sentence insertion/deletion, including rejected attempts."""
    if action.get('action') == 'MOVE':
        return True
    args = action.get('args') or {}
    target = args.get('target')
    return bool(action.get('action') == 'EDIT' and isinstance(target, str) and
                (target.startswith(('before:', 'after:')) or
                 (':' not in target and args.get('new_text') == '')))


def trajectory_details(row, role, records, reward):
    actions = row.get('actions_by_role', {}).get(role, [])
    attempts = [a for a in actions if structural_attempt(a)]
    accepted = [a for a in attempts if a.get('valid')]
    terminal_stop = row.get('termination', {}).get(role) == 'STOP'
    stop_only = terminal_stop and bool(actions) and all(a['action'] == 'STOP' for a in actions)
    recovery = {r['record_id']: r for r in (reward or {}).get('per_record', [])}
    operators = {}
    for op in OPERATORS[role]:
        own = [r for r in records if r['op'] == op]
        if not own:
            continue
        if any(r['record_id'] not in recovery for r in own):
            raise ValueError(f'Missing saved {role} recovery evidence for {op}')
        values = [recovery[r['record_id']]['main'] for r in own]
        if any(not 0 <= v <= 1 for v in values):
            raise ValueError('Recovery evidence must be finite and bounded')
        full = sum(v >= 1 - 1e-12 for v in values)
        operators[op] = {'records': len(values), 'fully_recovered_records': full,
            'partially_recovered_records': sum(0 < v < 1 - 1e-12 for v in values),
            'at_least_one_fully_recovered': bool(full), 'all_fully_recovered': full == len(values),
            'any_positive_recovery': any(v > 0 for v in values)}
    return {'episode_id': row['corpus_episode_id'], 'source_id': row['source_id'],
        'episode_completed': row['completed'], 'has_global_records': any(r['level'] == 'GLOBAL' for r in records),
        'structural_attempts': len(attempts), 'accepted_structural_actions': len(accepted),
        'STOP_only': stop_only, 'single_valid_STOP_only': stop_only and len(actions) == 1 and actions[0]['valid'],
        'STOP_without_structural_attempt': terminal_stop and not attempts,
        'operators': operators}


def composition_summary(details, role):
    result = {'trajectories': len(details),
        'with_structural_attempt': sum(d['structural_attempts'] > 0 for d in details),
        'with_accepted_structural_action': sum(d['accepted_structural_actions'] > 0 for d in details),
        'STOP_only': sum(d['STOP_only'] for d in details),
        'single_valid_STOP_only': sum(d['single_valid_STOP_only'] for d in details),
        'STOP_without_structural_attempt': sum(d['STOP_without_structural_attempt'] for d in details),
        'with_global_records': sum(d['has_global_records'] for d in details),
        'STOP_only_with_global_records': sum(d['STOP_only'] and d['has_global_records'] for d in details),
        'incomplete_episode_with_selected_role': sum(not d['episode_completed'] for d in details),
        'operators': {}}
    result['other_than_structural_or_STOP_only'] = sum(
        not d['structural_attempts'] and not d['STOP_only'] for d in details)
    for op in OPERATORS[role]:
        own = [d['operators'][op] for d in details if op in d['operators']]
        result['operators'][op] = {'selected_trajectories_with_operator': len(own),
            **{key: sum(d[key] for d in own) for key in ('records', 'fully_recovered_records',
                'partially_recovered_records', 'at_least_one_fully_recovered', 'all_fully_recovered',
                'any_positive_recovery')}}
    return result


def export_composition(root, corpus_path, *, partial_path=None):
    """Read saved stage-specific rewards; do not regenerate scores or exports."""
    root = Path(root)
    path = root / 'data/manifest.json'
    manifest = read_json(path)
    if file_sha(corpus_path) != manifest['contract']['corpus_sha256']:
        raise ValueError('The corpus used for the frozen export changed')
    with Path(corpus_path).open(encoding='utf-8') as stream:
        corpus = {r['episode_id']: r for r in map(json.loads, stream)}
    partial = read_json(partial_path)['results'] if partial_path and Path(partial_path).exists() else {}
    result = {'manifest_sha256': file_sha(path), 'api_calls': 0, 'gpu_used': False,
        'definitions': {
            'structural_action': 'MOVE or sentence insertion/deletion; attempts and accepted actions reported separately',
            'STOP_only': 'role terminates STOP and every attempted action is STOP; rejected STOP retries retained',
            'full_recovery': 'saved per-record main recovery equals 1 (tolerance 1e-12)',
            'global_recovery_state': 'after GLOBAL, before KOREAN',
            'korean_recovery_state': 'after KOREAN; original local records only',
            'trajectory_operator_counts': 'each selected trajectory counted once per operator; any/all record recovery separate',
        }, 'roles': {}, 'trajectories': [],
        'partial_reward_evidence': {'path': str(partial_path), 'sha256': file_sha(partial_path)} if partial else None}
    for role in OPERATORS:
        own = []
        split = manifest['contract']['split']['roles'][role]
        partition = {i: p for p in ('train', 'validation') for i in split[p + '_ids']}
        for key, entry in manifest['source_trajectories'].items():
            if not key.startswith(role + ':'):
                continue
            episode_id = key[len(role) + 1:]
            source = Path(entry['path'])
            if file_sha(source) != entry['sha256']:
                raise ValueError('Selected teacher trajectory changed')
            row = read_json(source)
            if row['corpus_episode_id'] != episode_id or row['source_id'] != corpus[episode_id]['source_id']:
                raise ValueError('Selected teacher identity mismatch')
            reward = (row.get('reward') or {}).get(role)
            if reward is None and role == 'global':
                reward = row.get('global_only_reward') or partial.get('luna_low:' + episode_id, {}).get('global_only_reward')
            if reward is None:
                raise ValueError('Selected role has no saved reward evidence')
            detail = trajectory_details(row, role, corpus[episode_id]['records'], reward)
            detail.update(role=role, partition=partition[episode_id], source_path=str(source),
                          source_sha256=entry['sha256'], attempt=entry['attempt'])
            own.append(detail)
        if len(own) != sum(manifest['roles'][role][p]['trajectories'] for p in ('train', 'validation')):
            raise ValueError('Composition does not cover the frozen export')
        result['roles'][role] = {'all': composition_summary(own, role),
            **{p: composition_summary([d for d in own if d['partition'] == p], role) for p in ('train', 'validation')}}
        result['trajectories'].extend(own)
    write_json(root / 'export_composition.json', result)
    return result


def global_inaction(rows, corpus, intended_ids):
    """GLOBAL-record denominator includes failures; missing decisions stay unknown."""
    eligible = {i for i in intended_ids if any(r['level'] == 'GLOBAL' for r in corpus[i]['records'])}
    observed = {r['corpus_episode_id']: r for r in rows if r['corpus_episode_id'] in eligible}
    if len(observed) != sum(r['corpus_episode_id'] in eligible for r in rows):
        raise ValueError('Duplicate evaluation episode in GLOBAL stopping diagnostic')
    resolved = {i: r for i, r in observed.items() if r.get('termination', {}).get('global') is not None}
    no_attempt, no_accepted = [], []
    for i, row in resolved.items():
        if row['termination']['global'] != 'STOP':
            continue
        actions = row.get('actions_by_role', {}).get('global', [])
        if not any(structural_attempt(a) for a in actions):
            no_attempt.append(i)
        if not any(structural_attempt(a) and a.get('valid') for a in actions):
            no_accepted.append(i)
    n, unknown = len(eligible), len(eligible - resolved.keys())
    result = {'intended_with_global_records': n, 'attempted_with_global_records': len(observed),
        'global_stage_finished': len(resolved), 'unknown': unknown,
        'global_terminations': dict(Counter(r['termination']['global'] for r in resolved.values())),
        'unknown_ids': sorted(eligible - resolved.keys())}
    for key, ids in [('STOP_without_structural_attempt', no_attempt),
                     ('STOP_without_accepted_structural_action', no_accepted)]:
        result[key] = {'n': len(ids), 'denominator': n, 'ids': sorted(ids),
            'rate': len(ids) / n if n and not unknown else None,
            'lower_bound': len(ids) / n if n else None,
            'upper_bound': (len(ids) + unknown) / n if n else None}
    return result


def main():
    config = load_config()
    root = config['paths']['repo'] / 'verak/v3/outputs/phase7_sft'
    def corpus(split):
        with (config['paths']['active_corrupt'] / (split + '.jsonl')).open(encoding='utf-8') as stream:
            return {r['episode_id']: r for r in map(json.loads, stream)}
    result = export_composition(root, config['paths']['active_corrupt'] / 'agent_train.jsonl',
        partial_path=config['paths']['phase7_teacher_output'] / 'completed_global_evaluation.json')
    design = read_json(root / 'evaluation_design.json')
    dev = corpus('agent_dev')
    evaluations = {}
    for condition in design['contract']['conditions']:
        rows = {r['corpus_episode_id']: r for p in (root / 'evaluation' / condition / 'episodes').glob('*.json')
                if (r := read_json(p)).get('cohort') == 'dev'}
        if condition == 'luna_low':
            for i in design['contract']['dev_ids']:
                if i in design['reuse']:
                    rows[i] = read_json(design['reuse'][i]['path'])
        evaluations[condition] = global_inaction(list(rows.values()), dev, design['contract']['dev_ids'])
    write_json(root / 'global_inaction_progress.json', evaluations)
    print(json.dumps({'composition': result['roles'], 'global_inaction': evaluations}, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
