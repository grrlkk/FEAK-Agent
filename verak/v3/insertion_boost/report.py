"""Bounded insertion component report for the root-owned V3_DATA_BOOST.md."""
from collections import Counter
from pathlib import Path
from statistics import mean

from ..common import file_sha, read_json, write_json
from ..train.teacher_comparison import absolute_selection
from ..v2_ops.config import PHASE
from ..v2_ops.report import accounting, table
from ..v2_ops.teacher import attempt_path
from .cpu_score import shared_root
from .data import summary


def average(values):
    return mean(values) if values else None


def report(config, *, final=False):
    root = config['paths'][PHASE + '_output']
    qc = summary(config)
    approval_path = shared_root(config) / 'cpu_scorer/selection_approval.json'
    approval = read_json(approval_path) if approval_path.exists() else {'canonical_for_selection': False}
    if final and (not approval.get('canonical_for_selection') or approval.get('score_source') != 'gpu_reference'):
        raise ValueError('Final selection requires GPU reference provenance and the completed 200-essay audit')
    design_path = root / 'teacher_design.json'
    design = read_json(design_path) if design_path.exists() else None
    selected, slots, recoveries, rewards, examples, failures = {}, [], [], [], [], []
    generation, termination, actions = Counter(), Counter(), Counter()
    scored_devices, best_recovery, recovery_over = Counter(), {}, []
    if design:
        for attempt in (1, 2):
            for eid in design['orders'][str(attempt)]:
                path = attempt_path(root, attempt, eid)
                slot = {'episode_id': eid, 'attempt': attempt, 'status': 'not_attempted'}
                slots.append(slot)
                if not path.exists():
                    continue
                row = read_json(path)
                if row['v2']['design_sha256'] != file_sha(design_path):
                    raise ValueError('Teacher provenance changed')
                candidate = read_json(design['corpus'][eid]['path'])
                generation['saved'] += 1
                generation['completed'] += row['completed']
                generation['global_stage_saved'] += row.get('stage1_layout') is not None
                slot['status'] = 'completed' if row['completed'] else 'runtime_failure'
                slot['path'] = str(path)
                termination.update([row['termination'].get('global', 'runtime_failure')])
                actions.update(a['action'] for a in row['actions_by_role'].get('global', []))
                recovery_path = root / f'attempt_{attempt}/recovery' / path.name
                stage = None
                if recovery_path.exists():
                    saved = read_json(recovery_path)
                    if saved['episode_sha256'] != file_sha(path):
                        raise ValueError('Recovery provenance changed')
                    stage = saved['stages'].get('global')
                    if stage:
                        value = stage['per_record'][0]['main']
                        recoveries.append(value)
                        best_recovery[eid] = max(value, best_recovery.get(eid, 0))
                        if 'R_over' in stage:
                            recovery_over.append(stage['R_over'])
                scored = root / f'scored/attempt_{attempt}' / path.name
                reward = None
                if scored.exists():
                    enriched = read_json(scored)
                    if enriched['raw_episode_sha256'] != file_sha(path):
                        raise ValueError('Scored trajectory provenance changed')
                    if enriched.get('score_source') != 'gpu_reference' or any(
                           s.get('execution_device') != 'gpu_reference' or
                           s.get('scorer_fingerprint') != approval.get('fingerprint') for s in enriched['quality_scores']):
                        generation['superseded_scored_attempts'] += 1
                    else:
                        reward = enriched['global_only_reward']
                        rewards.append(reward)
                        scored_devices.update(s['execution_device'] for s in enriched['quality_scores'])
                        keep = absolute_selection(enriched, candidate)
                        if keep['korean']:
                            raise AssertionError('No new KOREAN selection is authorized')
                        if keep['global']:
                            item = {'attempt': attempt, 'path': str(scored), 'sha256': file_sha(scored),
                                'R': reward['R'], 'operator': 'G_DEL_LINK', 'source_id': candidate['source_id']}
                            previous = selected.get(eid)
                            if previous is None or (item['R'], -attempt) > (previous['R'], -previous['attempt']):
                                selected[eid] = item
                inserted = []
                for action in row['actions_by_role'].get('global', []):
                    if not action.get('valid'):
                        continue
                    if action['action'] == 'INSERT':
                        inserted.append(action['args']['text'])
                    elif action['action'] == 'EDIT' and action['args'].get('target', '').startswith(('before:', 'after:')):
                        inserted.append(action['args']['new_text'])
                example = {'episode_id': eid, 'attempt': attempt, 'source_id': candidate['source_id'],
                        'deleted_original': ' / '.join(candidate['records'][0]['original_text'].values()),
                        'inserted': inserted, 'GLOBAL_main_recovery': stage['per_record'][0]['main'] if stage else None,
                        'GLOBAL_R_over': reward['R_over'] if reward else stage.get('R_over') if stage else None,
                        'GLOBAL_R': reward['R'] if reward else None, 'path': str(path),
                        'insertion_outcome': 'accepted_insertion' if inserted else 'no_accepted_insertion'}
                (examples if inserted else failures).append(example)
    examples.sort(key=lambda r: (-(r['GLOBAL_main_recovery'] or 0), r['episode_id'], r['attempt']))
    diverse, seen = [], set()
    for row in examples:
        if row['source_id'] not in seen:
            seen.add(row['source_id'])
            diverse.append(row)
    accepted_example_count = len(diverse)
    for row in failures:
        if row['source_id'] not in seen:
            seen.add(row['source_id'])
            diverse.append(row)
    chosen_examples = diverse[:5]
    budget = accounting(root)
    if budget['confirmed_usd']+budget['reserved_usd'] > 6+1e-9:
        raise AssertionError('New insertion $6 cap exceeded')
    if final and budget['pending']:
        raise ValueError('Cannot finalize with API requests in flight')
    calibration_path = Path(approval.get('calibration_path', shared_root(config) / 'cpu_scorer/audit_200/calibration.json'))
    calibration = read_json(calibration_path) if calibration_path.exists() else None
    if final and (calibration is None or calibration.get('status') != 'complete' or
            calibration.get('unique_source_essays', 0) < 200 or file_sha(calibration_path) != approval['calibration_sha256']):
        raise ValueError('Final report requires the exact completed >=200-source consistency audit')
    numeric = ({k: v for k, v in calibration.items() if k not in ('comparisons', 'decisions')}
               if calibration else {'status': 'pending'})
    gpu_path = root / 'gpu_selection.json'
    gpu_selection = read_json(gpu_path) if gpu_path.exists() else None
    if final and (gpu_selection is None or gpu_selection['selected']['global'] != selected):
        raise ValueError('Final report differs from the GPU-only selection result')
    metrics = {'status': 'final' if final else 'progress', 'qc': qc, 'budget': budget,
        'planned_teacher_attempts': len(slots), 'generation': dict(generation),
        'not_attempted': sum(s['status'] == 'not_attempted' for s in slots),
        'global_termination': dict(termination), 'global_actions': dict(actions),
        'teacher_recovery': {'judged': len(recoveries), 'main_mean': average(recoveries),
            'full': sum(r == 1 for r in recoveries), 'partial': sum(r == .5 for r in recoveries),
            'none': sum(r == 0 for r in recoveries), 'best_of_two_essays': len(best_recovery),
            'best_of_two_main_mean': average(list(best_recovery.values())),
            'best_of_two_full': sum(r == 1 for r in best_recovery.values())},
        'global_rewards': {'available': len(rewards), 'mean_R': average([r['R'] for r in rewards]),
            'mean_R_over': average([r['R_over'] for r in rewards]),
            'mean_steps': average([r['R_step'] for r in rewards])},
        'R_over_without_quality_scoring': {'available': len(recovery_over), 'mean': average(recovery_over)},
        'quality_score_devices': dict(scored_devices), 'cpu_calibration': numeric,
        'cpu_to_gpu_selection': ({k: v for k, v in gpu_selection.items()
            if k not in {'selected', 'cpu_to_gpu_comparisons'}} if gpu_selection else {'status': 'pending'}),
        'selected_counts': {'global': len(selected), 'korean': 0},
        'selected': {'global': selected, 'korean': {}}, 'examples': chosen_examples,
        'accepted_insertion_examples_available': accepted_example_count,
        'insertion_example_shortfall': max(0, 5-accepted_example_count), 'slots': slots,
        'reward_scope': 'GLOBAL only; KOREAN and combined unmeasured',
        'scorer_selection_approval': approval,
        'score_source': 'gpu_reference', 'gpu_used': gpu_selection is not None,
        'gpu_calls_by_component': 0, 'training': False}
    write_json(root / 'metrics.json', metrics)
    if final:
        write_json(root / 'best_role_trajectories.json', metrics['selected'])
    lines = ['# G_DEL_LINK insertion data boost component', '',
        '**'+metrics['status']+'**. New separate $6 experiment; teacher/data preparation uses CPU/API only. '
        'Final quality rewards use the separately authorized deferred GPU reference; no training.', '',
        'The revised 30% gate is QC passes / judged. Archived retry artifacts remain unchanged. '
        'The prior 12 unjudged records were not re-sampled. Only the remaining 200 dependency-feasible train '
        'sources were added, preserving source/real/dev exclusions, exact Bareun dependency screen, Sol roles, '
        'and the three Sol QC criteria. The accepted original retry contributes 78 passing train records.', '',
        table(['cohort', 'planned sources', 'judged', 'QC passes', 'pass/judged'], [
            ['prior retry train+dev', 380, 270, 96, 96/270],
            ['new train', 200, qc['new_train'].get('judged', 0), qc['new_train'].get('passed', 0), qc['new_qc_pass_rate']],
            ['combined', 580, qc['combined']['judged'], qc['combined']['passed'], qc['combined']['qc_pass_rate']]]), '',
        f"Gate: **{qc['decision']}**. New role/surface exclusions: {qc['new_train']['role_or_surface_failure']}; "
        f"new unknown QC: {qc['new_train'].get('unknown_qc', 0)}. Passing train supply: {qc['passing_train_records']}.", '',
        'Pinned Luna low runs the unchanged v2 two-stage agents, no CHECK, 8,192 context/1,024 output, '
        'two independent unseeded attempts with ordering seeds 79/80. KOREAN still runs normally; only GLOBAL '
        'is scored and selected. Best per essay uses the original SFT GLOBAL R >= .80 rule, without adding '
        'the later RFT STOP/rejection gates. Raw failed attempts and reservations are preserved.', '',
        table(['planned slots', 'saved', 'completed', 'GLOBAL recovery judged', 'main recovery', 'R_over', 'selected GLOBAL'], [[
            len(slots), generation['saved'], generation['completed'], len(recoveries), average(recoveries),
            average([r['R_over'] for r in rewards]), len(selected)]]), '',
        f"GLOBAL full/partial/no recovery: {sum(r == 1 for r in recoveries)} / {sum(r == .5 for r in recoveries)} / "
        f"{sum(r == 0 for r in recoveries)}. Best-of-two full: {sum(r == 1 for r in best_recovery.values())} / "
        f"{len(best_recovery)}. Quality-scored GLOBAL attempts: {len(rewards)}. Not attempted: {metrics['not_attempted']}.", '',
        '## Actual inserted sentence and deleted original pairs', '']
    if accepted_example_count < 5:
        lines += [f'Only {accepted_example_count} distinct essays with an accepted insertion are available. '
                  'The remaining examples explicitly show no accepted insertion; these are failures, not fabricated successes.', '']
    for row in chosen_examples:
        lines += [f"- `{row['episode_id']}`, attempt {row['attempt']}, main recovery={row['GLOBAL_main_recovery']}, "
                  f"GLOBAL R={row['GLOBAL_R']}, R_over={row['GLOBAL_R_over']}",
                  '  - Deleted original: '+row['deleted_original'].replace('\n', ' '),
                  '  - Actual inserted: '+(' / '.join(row['inserted']).replace('\n', ' ') if row['inserted'] else '(none)'), '']
    lines += ['## Frozen scoring and cost', '',
        'CPU scoring uses an explicitly provisional FP32 approximation of the frozen NF4 model. '
        'Its observed drift motivated an engineering BF16-wrapper probe, preserved separately. The user then '
        'required unconditional GPU reference scoring, so the faster FP32 path is used only for provisional '
        'comparison and the frozen 200-source audit. Every final teacher quality reward and SFT selection is '
        'recomputed from the original GPU reference. No CPU values are substituted when GPU scoring fails. '
        'The prompt, 8-score first line, greedy generation, '
        'raw digit probabilities, genre dead zones, and reward weights are unchanged. The CPU cache has a separate '
        'device/dtype fingerprint. Exact saved GPU cache hits retain GPU provenance. '
        'Only eight required teacher-forced vocabulary positions are projected; a unit comparison matches full logits. '
        'The consistency audit contains 200 distinct source essays: genre quotas 58 explanation, 60 emotion, '
        '82 argument, including 62 near-.80 examples and 138 seeded samples. It is a threshold-stress sample, '
        'not a population accuracy estimate. Generated score-line digit agreement is reported separately '
        'from teacher-forced argmax digit agreement. Observed numerical errors are not universal bounds.', '',
        f'CPU/GPU consistency audit: `{numeric}`. Selection approval: `{approval}`. '
        f'CPU-to-GPU teacher eligibility changes: `{metrics["cpu_to_gpu_selection"]}`.', '',
        f"New confirmed cost **${budget['confirmed_usd']:.6f} / $6**; reserved ${budget['reserved_usd']:.6f}; "
        f"pending {budget['pending']}. The prior retry's $3.461224 is historical and is not charged to this cap. "
        'All new Sol/Luna generation and hidden recovery judgments share the same atomic ledger.', '',
        'The initial sandbox DNS failure produced zero API sends and zero charges. Exact errors and the '
        'empty provisional QC plan are retained under no_send_reconciliation; paid incomplete QC is never repeated. '
        'Bareun is cache-first, globally serialized at <=1 new request/sec, with priority pauses on busy/timeouts '
        'or new task-A Bareun errors. Only the root controller uses GPUs in the authorized gap or after RFT '
        'evaluation; this component never starts a GPU process. Active RFT runtime/corpora are untouched, '
        'and insertion selections remain separate for round 2.', '',
        'Artifacts: `'+str(root)+'`. KOREAN selections: 0; KOREAN and combined reward unmeasured.']
    path = root / 'component_report.md'
    path.write_text('\n'.join(lines)+'\n', encoding='utf-8')
    if final:
        write_json(root / 'component_metrics.json', metrics)
        write_json(root / 'complete.json', {
            'status': 'budget_stop' if metrics['not_attempted'] else 'complete',
            'metrics_path': str(root / 'component_metrics.json'),
            'metrics_sha256': file_sha(root / 'component_metrics.json'),
            'report_path': str(path), 'report_sha256': file_sha(path),
            'scorer_approval_sha256': file_sha(approval_path),
            'no_live_paid_calls': budget['pending'] == 0,
            'measurements_finished': True, 'stopped': True,
            'selected_counts': metrics['selected_counts'], 'score_source': 'gpu_reference',
            'gpu_used': metrics['gpu_used'], 'gpu_calls_by_component': 0, 'training': False})
    return {'report': str(path), **{k: v for k, v in metrics.items() if k not in ('selected', 'slots', 'examples')}}
