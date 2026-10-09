"""Read-only component report and GLOBAL-only best-attempt selection index."""
from collections import Counter
from statistics import mean

from ..common import file_sha, read_json, write_json
from ..train.teacher_comparison import absolute_selection
from .config import PHASE, OPERATORS
from .paid import BoostAPI
from .prepare import corpus
from .qc import summarize
from .teacher import attempt_path, prepare_teacher


def report(config):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    qc = summarize(config)
    design, rows = prepare_teacher(config)
    metrics, selected = {}, {}
    for operator in OPERATORS:
        ids = [i for i, r in rows.items() if r['operator'] == operator]
        values, recovery = [], []
        attempts, complete, measured, full = 0, 0, 0, 0
        unknown, end = [], {'global': Counter(), 'korean': Counter()}
        per_essay = {i: [] for i in ids}
        for episode_id in ids:
            for attempt in (1, 2):
                path = attempt_path(root, attempt, episode_id)
                measured_path = root / f'attempt_{attempt}/measured' / path.name
                if not path.exists():
                    unknown.append({'episode_id': episode_id, 'attempt': attempt, 'reason': 'not_attempted'})
                    continue
                raw = read_json(path)
                attempts += 1
                complete += bool(raw['generation_completed'])
                for role in end:
                    end[role][raw['termination'].get(role, 'not_completed')] += 1
                if not measured_path.exists():
                    unknown.append({'episode_id': episode_id, 'attempt': attempt, 'reason': 'not_measured'})
                    continue
                result = read_json(measured_path)
                if result['raw_generation_sha256'] != file_sha(path):
                    raise ValueError('Measured result no longer matches the immutable teacher trace')
                reward = result['reward']['global'] if result.get('reward') else result.get('global_only_reward')
                if reward is None:
                    unknown.append({'episode_id': episode_id, 'attempt': attempt, 'reason': 'global_incomplete'})
                    continue
                measured += 1
                values.append(reward)
                rec = reward['per_record'][0]['main']
                recovery.append(rec)
                per_essay[episode_id].append(rec)
                full += rec == 1
                keep = absolute_selection(result, rows[episode_id])
                if keep['global']:
                    entry = {'attempt': attempt, 'path': str(measured_path), 'sha256': file_sha(measured_path),
                        'trajectory_path': str(path), 'trajectory_sha256': file_sha(path),
                        'R': reward['R'], 'R_rec': reward['R_rec'], 'R_over': reward['R_over'],
                        'source_id': rows[episode_id]['source_id'], 'operator': operator,
                        'rule': keep['global_rule'], 'selected_role': 'global'}
                    old = selected.get(episode_id)
                    if old is None or (entry['R'], -attempt) > (old['R'], -old['attempt']):
                        selected[episode_id] = entry
        complete_pairs = [values for values in per_essay.values() if len(values) == 2]
        observed_pairs = [values for values in per_essay.values() if values]
        metrics[operator] = {'requested': 400, 'generated': plan['counts'][operator],
            'supply_shortfall': plan['shortfall'][operator], 'qc': qc[operator],
            'teacher_planned': 2*len(ids), 'teacher_saved': attempts, 'generation_completed': complete,
            'global_reward_measured': measured, 'completion_rate': complete/attempts if attempts else None,
            'teacher_main_recovery': mean(recovery) if recovery else None,
            'fully_recovered_attempts': full, 'full_recovery_rate': full/measured if measured else None,
            'at_least_one_of_two': {'fully_observed_essays': len(complete_pairs),
                'recovered': sum(max(v) == 1 for v in complete_pairs),
                'share': mean(max(v) == 1 for v in complete_pairs) if complete_pairs else None},
            'observed_any_sample': {'essays': len(observed_pairs),
                'recovered': sum(max(v) == 1 for v in observed_pairs)},
            'R': mean(v['R'] for v in values) if values else None,
            'R_over': mean(v['R_over'] for v in values) if values else None,
            'termination': {role: dict(counts) for role, counts in end.items()},
            'selected_global': sum(row['operator'] == operator for row in selected.values()),
            'selected_korean': 0, 'unknown': unknown}
    api = BoostAPI(config, 0)
    account = api.accounting()
    if account['confirmed_usd'] + account['reserved_usd'] > 12. + 1e-8:
        raise ValueError('Aggregate GLOBAL boost budget exceeded')
    all_inputs = corpus(config)
    result = {'phase': PHASE, 'operators': metrics, 'selected_global': len(selected),
        'selected_korean': 0, 'selected': selected, 'api': account,
        'distinct_new_sources': len({r['source_id'] for r in all_inputs.values()}),
        'source_overlap_across_operators': len(plan['source_overlap_across_operators']),
        'source_inventory': plan['source_inventory']['counts'],
        'selection_rule': 'Existing SFT absolute_selection, best GLOBAL R per practice; no additional RFT-only STOP/rejection gate.',
        'new_teacher_roles': ['global', 'korean'], 'selected_roles': ['global'],
        'method': 'Unchanged v1 two-stage prompts/actions/recovery; hidden reward deferred and CPU NF4 scored.',
        'gpu_used': False, 'training': False}
    result['completion'] = {
        'teacher_requested': sum(v['teacher_planned'] for v in metrics.values()),
        'teacher_saved': sum(v['teacher_saved'] for v in metrics.values()),
        'rewards_measured': sum(v['global_reward_measured'] for v in metrics.values()),
        'pending_api': account['pending'],
        'stop_reason': read_json(root / 'teacher_status.json').get('stop_reason')
            if (root / 'teacher_status.json').exists() else 'teacher_not_started',
    }
    cpu_audit = root.parent / 'cpu_scorer/calibration.json'
    if cpu_audit.exists():
        audit = read_json(cpu_audit)
        result['cpu_score_audit'] = {k: v for k, v in audit.items() if k not in {'comparisons', 'decisions'}}
    write_json(root / 'best_global_trajectories.json', {'global': selected, 'korean': {}})
    write_json(root / 'component_metrics.json', result)
    lines = ['# GLOBAL data boost', '',
        f"Phase3b source pool {result['source_inventory']['phase3b_train_source_pool']}; "
        f"unused eligible {result['source_inventory']['unused_eligible_sources']}; "
        f"mechanically eligible distinct sources {result['distinct_new_sources']}.", '',
        'The 400-per-operator targets exceed unused-source supply under the preserved Phase3b source and mechanical rules. '
        'The two operators reuse the same new sources, each practice has exactly one GLOBAL record. No active corpus was changed.', '',
        '|Operator|Generated|QC pass/judged|Teacher saved/planned|GLOBAL recovery|R_over|Selected GLOBAL|',
        '|---|---:|---:|---:|---:|---:|---:|']
    def number(value):
        return 'NA' if value is None else f'{value:.4f}'
    for operator, row in metrics.items():
        lines.append(f"|{operator}|{row['generated']}|{row['qc']['passed']}/{row['qc']['judged']}|"
            f"{row['teacher_saved']}/{row['teacher_planned']}|{number(row['teacher_main_recovery'])}|"
            f"{number(row['R_over'])}|{row['selected_global']}|")
    lines += ['', f"Confirmed API cost ${account['confirmed_usd']:.6f}; retained reservation "
        f"${account['reserved_usd']:.6f}; cap $12. Selected KOREAN: 0. GPU calls: 0. Training: none.", '',
        result['selection_rule'], '', 'Teacher prompt and tool behavior use v1. Scoring runs after generation because CHECK is disabled. '
        'CPU scoring has separate provenance; numerical comparison with saved GPU values is in cpu_scorer/calibration.json.', '']
    (root / 'component_report.md').write_text('\n'.join(lines), encoding='utf-8')
    return {k: v for k, v in result.items() if k != 'selected'}
