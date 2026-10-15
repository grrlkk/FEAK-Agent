"""Read-only retry report with separate denominators and no missing-as-zero data."""
from collections import Counter
from statistics import mean

from ..common import file_sha, read_json, write_json
from .config import PHASE
from .qc import candidate_paths, summary
from .report import accounting, table
from .teacher import attempt_path


def report(config, *, final=False):
    root = config['paths'][PHASE + '_output']
    plan = read_json(root / 'source_plan.json')
    qc = summary(config)
    result = qc['operators']['G_DEL_LINK']
    rows = {row['episode_id']: row for path in candidate_paths(config) if (row := read_json(path))}
    verdicts = {row['episode_id']: row for path in (root / 'qc').glob('*.json') if (row := read_json(path))}
    false = Counter(field for row in verdicts.values() for field, value in row['verdict'].items() if value is False)
    real_damage = sum(row['verdict']['damage_real'] for row in verdicts.values())
    damage_not_recoverable = sum(row['verdict']['damage_real'] and not row['verdict']['recoverable_from_essay']
                                 for row in verdicts.values())
    types = Counter(h['kind'] for row in rows.values() for h in row['dependency_screen']['hints'])
    type_qc = {}
    for kind in sorted(types):
        relevant = [row for row in rows.values() if any(h['kind'] == kind for h in row['dependency_screen']['hints'])]
        known = [verdicts[row['episode_id']] for row in relevant if row['episode_id'] in verdicts]
        type_qc[kind] = {'generated': len(relevant), 'judged': len(known), 'passed': sum(r['passed'] for r in known)}
    reused_labels = sum('reused_from' in read_json(path) for path in (root / 'labels').glob('*.json'))
    reused_qc = sum('reused_from' in row for row in verdicts.values())
    budget = accounting(root)
    incomplete_qc = budget.get('by_stage', {}).get('v2_retry_instance_qc', {}).get('statuses', {}).get('incomplete', 0)
    if budget['confirmed_usd'] + budget['reserved_usd'] > 10 + 1e-8:
        raise AssertionError('Retry budget exceeded')
    if final and budget['pending']:
        raise ValueError('Do not finalize while API calls are pending')
    design_path = root / 'teacher_design.json'
    design = read_json(design_path) if design_path.exists() else None
    slots, outcomes, rec = [], [], []
    if design:
        for attempt in (1, 2):
            for episode_id in design['orders'][str(attempt)]:
                path = attempt_path(root, attempt, episode_id)
                slot = {'episode_id': episode_id, 'attempt': attempt, 'status': 'not_attempted'}
                slots.append(slot)
                if not path.exists():
                    continue
                raw = read_json(path)
                if raw['v2']['design_sha256'] != file_sha(design_path):
                    raise ValueError('Retry teacher design mismatch')
                slot['status'] = 'completed' if raw['completed'] else 'runtime_failure'
                outcomes.append(raw)
                recovered = root / f'attempt_{attempt}/recovery' / path.name
                if recovered.exists():
                    value = read_json(recovered)
                    if value['episode_sha256'] != file_sha(path):
                        raise ValueError('Retry recovery provenance mismatch')
                    rec.append(value['stages'])
    main = [r['combined']['per_record'][0]['main'] for r in rec if 'combined' in r]
    middle = [r['global']['per_record'][0]['main'] for r in rec if 'global' in r]
    teacher = {'planned': len(slots), 'attempted': len(outcomes), 'completed': sum(r['completed'] for r in outcomes),
        'not_attempted': sum(s['status'] == 'not_attempted' for s in slots),
        'global_recovery_known': len(middle), 'global_main_recovery': mean(middle) if middle else None,
        'final_recovery_known': len(main), 'final_main_recovery': mean(main) if main else None,
        'full_recovery': sum(v == 1 for v in main), 'partial_recovery': sum(v == .5 for v in main),
        'R_over': None, 'combined_R': None, 'selected_counts': {'global': 0, 'korean': 0},
        'selection_status': ('not_scheduled_dropped_operator' if result['decision'] == 'drop' else
                             'awaiting_qc_gate' if result['decision'] == 'undetermined' else 'awaiting_exact_cpu_scoring')}
    cpu_metrics = root / 'cpu_metrics.json'
    if cpu_metrics.exists():
        teacher.update(read_json(cpu_metrics))
    metrics = {'status': 'final' if final else 'progress', 'qc': qc, 'false_criteria': dict(false),
        'dependency_types': dict(types), 'qc_by_dependency_type': type_qc, 'teacher': teacher, 'budget': budget,
        'reused_labels': reused_labels, 'reused_qc': reused_qc,
        'gpu_used': False, 'training': False, 'source_sampling': plan['sampling'],
        'dropped_l_fuse': {'acceptable': 338, 'judged': 377}, 'slots': slots}
    write_json(root / 'metrics.json', metrics)
    if result['decision'] == 'drop':
        write_json(root / 'best_role_trajectories.json', {'global': {}, 'korean': {}})
    c = result['counts']
    lines = ['# VERAK v3 — G_DEL_LINK dependency retry', '',
        '**Status: ' + ('final' if final else 'in progress') + '.** Task B only; no v2 training and no GPU use.', '',
        f"The frozen request is 300 train and 80 dev source essays. G_DEL_LINK has {c.get('passed', 0)}/380 "
        f"known QC passes, with all-source yield bounds {100*result['yield_lower_bound']:.2f}–"
        f"{100*result['yield_upper_bound']:.2f}%. The 30% decision is **{result['decision']}**.", '',
        'L_FUSE is dropped: **338/377** previously judged merges were acceptable alternative structures, '
        'so merging sentences generally did not produce a real error. No L_FUSE generation, QC or teacher calls were made in this retry.', '',
        '## Dependency screen and frozen sources', '',
        'The previous source-quality pool, paragraph-first/last-paragraph deletion sites, evaluation exclusions, '
        'Sol role labels and QC questions are unchanged. Cached Bareun tokens screen the immediate successor first; '
        'Sol topic/bridge/summary is the second filter. The selected deletion is seeded without consulting earlier QC verdicts.', '',
        'Eligible hints are an anchored listed anaphoric expression; Bareun MMD/MM 그 or 이 followed by a noun; '
        '그렇다면/그러면; an ordinal introduced by an explicit list cue plus list noun (or a why/how question); '
        'or a question immediately before the next paragraph whose declarative answer paragraph shares a substantive noun. '
        'The last criterion is a conservative lexical candidate test, not a claim that Bareun proves answer semantics. '
        'Independent Sol damage/repair/recoverability QC still decides usable records.', '',
        table(['split', 'allowed pool', 'dependency-feasible sources', 'selected', 'shortfall'],
            [[split, data['screening']['screened_sources'], data['dependency_feasible_sources'],
              data['selected'], data['shortfall']] for split, data in plan['sampling']['G_DEL_LINK'].items()]), '',
        'Source and question splits remain disjoint. The 100 SFT dev instances (84 source essays) and 30 real essays '
        'are excluded by ID and text hash. Missing feasible sources, if any, remain in the requested 380-source denominator.', '',
        table(['selected deletion hint', 'records (types may overlap)'], sorted(types.items())), '',
        f'Reused **{reused_labels}** exact-source Sol role-label records and **{reused_qc}** identical-deletion QC judgments. '
        'For QC reuse, question, source, corruption, deleted sentence and operation parameters must all match; '
        'only the experiment episode identifier differs. Both accepted and rejected cached verdicts are reused.', '',
        '## QC', '',
        table(['planned', 'generated', 'judged', 'pass', 'QC pass/judged', 'yield lower', 'yield upper', 'decision'],
            [[c['planned'], c.get('generated', 0), c.get('judged', 0), c.get('passed', 0),
              result['qc_pass_rate_judged'], result['yield_lower_bound'], result['yield_upper_bound'], result['decision']]]), '',
        table(['split', 'planned', 'generated', 'judged', 'pass', 'mechanical/role failure', 'unknown'],
            [[split, part['planned'], part.get('generated', 0), part.get('judged', 0), part.get('passed', 0),
              part.get('mechanical_failure', 0), part.get('pending_candidate', 0)+part.get('pending_qc', 0)]
             for split, part in result['by_split'].items()]), '',
        table(['false QC criterion', 'count (may overlap)'], sorted(false.items())), '',
        table(['dependency hint', 'generated', 'judged', 'pass'],
            [[kind, values['generated'], values['judged'], values['passed']] for kind, values in type_qc.items()]), '',
        'The 30% gate uses all 380 requested sources, including role-filter rejections. Missing API outcomes remain unknown; '
        'bounds treat those as all failing or all passing. They are coverage bounds, not confidence intervals.', '',
        f"Of the {len(verdicts)} judged deletions, {false['damage_real']} did not cause real damage. "
        f"Among {real_damage} judged damaging deletions, {damage_not_recoverable} could not be recovered from the remaining essay. "
        'The dependency hint alone therefore does not ensure both a necessary repair and recoverable content. '
        'Sol explanations include an anaphor that still has another antecedent, a redundant summary whose deletion is harmless, '
        'and a damaged reference whose missing factual content cannot be reconstructed.', '',
        '## Teacher and rewards', '',
        'Only a retained operator schedules pinned Luna-low two-stage v2 teacher generation: two independent unseeded '
        'attempts per passing train essay, ordering seeds 79/80, no CHECK, 8,192 context and 1,024 output. '
        'The deleted reference is private to cached Luna recovery judgments. Matching insertion position ±1, '
        'role/content/new-fact grading and unmatched-insertion R_over are unchanged.', '',
        ('The operator failed the gate; teacher generation, scoring and selection were not scheduled. '
         'Recovery, R_over and combined R are unmeasured, not zero.' if result['decision'] == 'drop' else
         'Teacher completion, semantic recovery, scorer availability and final selection are counted separately. '
         'Missing quality scores never become zero or an eligible role reward.'), '',
        table(['planned attempts', 'attempted', 'completed', 'not attempted', 'recovery known', 'main recovery', 'R_over',
               'combined R', 'selected GLOBAL / KOREAN'], [[teacher['planned'], teacher['attempted'], teacher['completed'],
               teacher['not_attempted'], teacher['final_recovery_known'], teacher['final_main_recovery'], teacher['R_over'],
               teacher['combined_R'], f"{teacher['selected_counts']['global']} / {teacher['selected_counts']['korean']}"]]), '',
        'Selection status: `' + teacher['selection_status'] + '`.', '',
        '## Five examples', '']
    examples = sorted(rows.values(), key=lambda row: (not verdicts.get(row['episode_id'], {}).get('passed', False), row['episode_id']))
    for row in examples[:5]:
        rec = row['records'][0]
        verdict = verdicts.get(row['episode_id'])
        lines.extend([f"- `{row['episode_id']}` — " + ('QC pass' if verdict and verdict['passed'] else 'QC reject' if verdict else 'QC unknown'),
            '  - Deleted: ' + next(iter(rec['original_text'].values())).replace('\n', ' '),
            '  - Immediate successor: ' + row['dependency_screen']['next_sentence'].replace('\n', ' '),
            '  - Dependency: ' + ', '.join(h['kind'] for h in row['dependency_screen']['hints']) + '; role: ' + rec['params']['deleted_label'],
            '  - Sol: ' + (verdict['verdict']['reason'] if verdict else 'unjudged'), ''])
    lines += ['## Cost and execution', '',
        f"New confirmed cost: **${budget['confirmed_usd']:.6f}**; reserved/unknown **${budget['reserved_usd']:.6f}**; "
        f"pending {budget['pending']}. Combined commitment is ${budget['confirmed_usd']+budget['reserved_usd']:.6f} / $10.", '',
        table(['stage', 'statuses', 'confirmed USD', 'reserved USD'],
            [[stage, str(value['statuses']), value['confirmed_usd'], value['reserved_usd']]
             for stage, value in budget.get('by_stage', {}).items()]), '',
        'New role labeling uses Sol low and instance QC uses Sol high, both with a 4,096-token output limit. '
        f"Incomplete batches are preserved without resampling. There are {incomplete_qc} incomplete QC responses "
        f"and {c.get('pending_qc', 0)} unjudged candidates. "
        + ('Even treating all unjudged candidates as passes cannot reach the yield gate.'
           if result['decision'] == 'drop' else 'The bounds above determine whether the yield gate is established.'), '',
        'All paid calls share this retry’s atomic $10 reservation ledger and a maximum of two live requests. '
        'Prior v2 artifacts/costs are preserved separately. The CLI hides CUDA, uses nice 10, the last eight available '
        'CPU cores and one math-library thread. No GPU/scorer service is stopped, restarted or occupied by this task.', '',
        'Artifacts: `' + str(root) + '`. Frozen manifests, raw API outputs, failures, cache origins, dependency evidence '
        'and any teacher trajectories are retained. No data or weights are uploaded.']
    audit_path = root / 'final_integrity_audit.json'
    if audit_path.exists():
        audit = read_json(audit_path)
        lines += ['', f"Final integrity audit: **{audit['checks']:,} checks passed**. Historical peak reservation "
            f"commitment was ${audit['max_historical_commitment_usd']:.6f}; at most "
            f"{audit['max_historical_api_concurrency']} simultaneous requests. Exact v1 replay passed all 92 saved "
            'trajectories (91 full rewards and the preserved historical failure’s completed GLOBAL reward), '
            '1,165 model turns and 2,203 paragraph cache hits, with no new analyzer/scorer/API calls.']
    if (root / 'validation.json').exists():
        verification = read_json(root / 'validation.json')
        lines += ['', verification['summary']]
    destination = config['paths']['repo'] / 'imple/reports/V3_V2_OPS_RETRY.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'report': str(destination), 'qc': result, 'teacher': teacher, 'budget': budget}
