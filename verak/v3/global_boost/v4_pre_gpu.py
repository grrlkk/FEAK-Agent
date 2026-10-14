"""File-only collection report, published before waiting for scorer results."""
from collections import Counter

from ..common import file_sha, read_json, write_json
from .config import PHASE, OPERATORS
from .expansion import batch_configs, root_for
from .paid import BoostAPI
from .prepare import corpus, safe_id


def report(config):
    root = root_for(config)
    status = read_json(root / 'expansion_status.json')
    if status.get('stage') != 'generation_finished':
        raise RuntimeError('Finish all authorized teacher collection before the pre-GPU report')
    plan_path = root / 'v4/plan.json'
    plan = read_json(plan_path)
    values = {op: {'generated': 0, 'new_generated': 0, 'sources': set(), 'new_sources': set(),
        'source_practices': Counter(), 'source_saved_practices': Counter(),
        'qc_judged': 0, 'qc_passed': 0, 'qc_errors': 0, 'qc_unjudged': 0,
        'Luna_requested_attempts': 0, 'Luna_saved_attempts': 0, 'Luna_completed_attempts': 0,
        'Sol_rescue_saved_attempts': 0, 'Sol_rescue_completed_attempts': 0,
        'practices_with_both_Luna_attempts_saved': 0, 'teacher_errors': [],
        'new_Luna_unattempted_slots': 0, 'legacy_Luna_unattempted_slots': 0,
        'GLOBAL_STOP_and_rejection_gate_passes': 0,
        'termination': {'global': Counter(), 'korean': Counter()}}
        for op in OPERATORS}
    all_sources, all_new_sources = set(), set()
    for cfg in batch_configs(config):
        batch = cfg['paths'][PHASE + '_output']
        rows = corpus(cfg)
        saved_by_id = {}
        for path in sorted(batch.glob('attempt_*/episodes/*.json')):
            raw = read_json(path)
            saved_by_id.setdefault(raw['corpus_episode_id'], {})[int(path.parent.parent.name.split('_')[-1])] = raw
        for eid, candidate in rows.items():
            value = values[candidate['operator']]
            source = candidate['source_id']
            new = 'source_provenance' in candidate
            value['generated'] += 1
            value['new_generated'] += new
            value['sources'].add(source)
            value['source_practices'][source] += 1
            all_sources.add(source)
            if new:
                value['new_sources'].add(source)
                all_new_sources.add(source)
            verdict = batch / 'qc' / (safe_id(eid) + '.json')
            passed = False
            if verdict.exists():
                value['qc_judged'] += 1
                passed = bool(read_json(verdict)['passed'])
                value['qc_passed'] += passed
            elif (batch / 'qc_errors' / verdict.name).exists():
                value['qc_errors'] += 1
            else:
                value['qc_unjudged'] += 1
            attempts = saved_by_id.get(eid, {})
            luna_saved = sum(a in attempts for a in (1, 2))
            if passed:
                value['Luna_requested_attempts'] += 2
                value[('new' if new else 'legacy') + '_Luna_unattempted_slots'] += 2 - luna_saved
            value['practices_with_both_Luna_attempts_saved'] += luna_saved == 2
            if attempts:
                value['source_saved_practices'][source] += 1
            for attempt, raw in attempts.items():
                prefix = 'Sol_rescue' if attempt == 3 else 'Luna'
                value[prefix + '_saved_attempts'] += 1
                value[prefix + '_completed_attempts'] += bool(raw.get('generation_completed', raw.get('completed')))
                for role in ('global', 'korean'):
                    value['termination'][role][raw['termination'].get(role, 'not_completed')] += 1
                actions = raw.get('actions_by_role', {}).get('global', [])
                value['GLOBAL_STOP_and_rejection_gate_passes'] += bool(
                    raw['termination'].get('global') == 'STOP' and actions
                    and actions[-1].get('action') == 'STOP' and actions[-1].get('valid')
                    and sum(not a.get('valid', False) for a in actions) <= 1)
                if raw.get('runtime_error'):
                    value['teacher_errors'].append({'episode_id': eid, 'attempt': attempt, **raw['runtime_error']})
    for op, value in values.items():
        value['distinct_sources'] = len(value.pop('sources'))
        value['new_distinct_sources'] = len(value.pop('new_sources'))
        raw_counts, saved_counts = value.pop('source_practices'), value.pop('source_saved_practices')
        value['source_cap_inventory'] = {
            'raw_practices': sum(raw_counts.values()),
            'raw_source_groups_above_four': sum(n > 4 for n in raw_counts.values()),
            'raw_excess_above_four': sum(max(0, n - 4) for n in raw_counts.values()),
            'practices_with_saved_teacher_attempts': sum(saved_counts.values()),
            'saved_source_groups_above_four': sum(n > 4 for n in saved_counts.values()),
            'potential_saved_excess_before_GPU_eligibility': sum(max(0, n - 4) for n in saved_counts.values()),
            'actual_GPU_eligible_practices': None, 'actual_final_cap_exclusions': None}
        value['old_source_capped_pool'] = plan['old_source_capped_counts'][op]
        value['usable_diversified_candidate_pool'] = value['old_source_capped_pool'] + value['new_generated']
        value['qc_pass_rate_judged'] = value['qc_passed'] / value['qc_judged'] if value['qc_judged'] else None
        value['termination'] = {role: dict(counts) for role, counts in value['termination'].items()}
    api = BoostAPI(config, 0)
    account = api.accounting()
    api.close()
    if account['pending']:
        raise RuntimeError('Pre-GPU collection report must not close live paid calls')
    result = {'phase': PHASE, 'task_version': 'v4_prep', 'status': 'collection_finished_GPU_pending',
        'collection_stop_reason': status['stop_reason'], 'operators': values,
        'distinct_sources': len(all_sources), 'distinct_new_sources': len(all_new_sources),
        'api': account, 'budget_usd_cumulative': 40, 'historical_spending_included': True,
        'same92_comparison': read_json(root / 'v4/sol_luna_same92.json'),
        'rescue_status': read_json(root / 'v4/rescue_status.json') if (root / 'v4/rescue_status.json').exists() else None,
        'source_cap': 4, 'source_cap_ranking': 'GPU-reference role R only, still pending',
        'final_role_R': None, 'final_selected_GLOBAL': None, 'final_selected_KOREAN': 0,
        'quality_reward_status': 'Pending GPU reference; no CPU R reported or used for final selection',
        'diversity_plan_path': str(plan_path), 'diversity_plan_sha256': file_sha(plan_path),
        'no_live_paid_calls': True, 'teacher_collection_finished': True,
        'gpu_used_by_component': False, 'training': False}
    lines = ['# V4 prep A — collection complete, GPU results pending', '',
        'This report contains only collection and source-inventory facts. Final role rewards, eligible trajectories, '
        'and actual source-cap exclusions await GPU-reference scoring; no provisional CPU R is reported.', '',
        '| Operator | Practices / sources | New practices / sources | QC pass / judged | Luna completed / saved / requested | Sol completed / saved | Errors |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for op, v in values.items():
        lines.append(f"| {op} | {v['generated']} / {v['distinct_sources']} | {v['new_generated']} / {v['new_distinct_sources']} | "
            f"{v['qc_passed']} / {v['qc_judged']} | {v['Luna_completed_attempts']} / {v['Luna_saved_attempts']} / {v['Luna_requested_attempts']} | "
            f"{v['Sol_rescue_completed_attempts']} / {v['Sol_rescue_saved_attempts']} | {len(v['teacher_errors'])} |")
    lines += ['', 'Luna requested slots include explicitly archived, unfinished legacy slots. Their counts are separate from new unattempted slots in the JSON. '
        'QC failures, unknowns, runtime errors, and unfinished generations are not counted as completed teachers.', '',
        '| Operator | Teacher-bearing practices | Source groups with >4 | Potential excess before GPU eligibility |',
        '|---|---:|---:|---:|']
    for op, v in values.items():
        c = v['source_cap_inventory']
        lines.append(f"| {op} | {c['practices_with_saved_teacher_attempts']} | {c['saved_source_groups_above_four']} | {c['potential_saved_excess_before_GPU_eligibility']} |")
    lines += ['', 'The source-cap inventory describes raw available practices, not selected training data. '
        'After GPU scoring, select the best valid-STOP, rejected-actions≤1, GLOBAL R≥0.80 attempt per practice, then at most four practices per source/operator by GPU R. '
        'Original source IDs, active-source provenance, new-position proofs, and SFT source holdouts remain preserved.', '',
        'Saved same-92 comparison: G_PARA_SWAP Sol 10/21 versus Luna 10/21; G_SENT_MOVE Sol 9/13 versus Luna 6/13. '
        'The +23.08 percentage-point G_SENT_MOVE advantage enabled GLOBAL-only Sol rescue after both completed Luna attempts failed full recovery.', '',
        f"Cumulative cost including legacy work: ${account['confirmed_usd']:.6f} of $40; reserved ${account['reserved_usd']:.6f}; pending calls {account['pending']}.",
        f"Collection stop reason: {status['stop_reason']}. Final GPU component_metrics.json and component_report.md will follow. No training was started.", '']
    text = '\n'.join(lines)
    for directory in (root, root / 'v4'):
        write_json(directory / 'component_pre_gpu.json', result)
        (directory / 'component_pre_gpu.md').write_text(text, encoding='utf-8')
    return result
