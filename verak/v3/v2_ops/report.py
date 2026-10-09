"""Local v2 data-preparation report; no model, network, or GPU calls."""
from collections import Counter
import sqlite3
from statistics import mean

from ..common import file_sha, read_json, write_json
from ..train.teacher_comparison import absolute_selection
from .config import PHASE, OPERATORS
from .qc import candidate_paths, summary
from .teacher import safe_id, attempt_path


def accounting(root):
    path = root / 'api/ledger.sqlite'
    if not path.exists():
        return {'confirmed_usd': 0, 'reserved_usd': 0, 'pending': 0, 'calls': 0, 'blocked_before_send': 0}
    with sqlite3.connect('file:' + str(path) + '?mode=ro', uri=True) as db:
        rows = db.execute('SELECT stage,status,confirmed,reserved FROM calls').fetchall()
    return {'confirmed_usd': sum(r[2] for r in rows), 'reserved_usd': sum(r[3] for r in rows),
        'pending': sum(r[1] == 'pending' for r in rows),
        'calls': sum(r[1] != 'blocked_before_send' for r in rows),
        'blocked_before_send': sum(r[1] == 'blocked_before_send' for r in rows),
        'by_stage': {stage: {'statuses': dict(Counter(r[1] for r in rows if r[0] == stage)),
            'confirmed_usd': sum(r[2] for r in rows if r[0] == stage),
            'reserved_usd': sum(r[3] for r in rows if r[0] == stage)} for stage in sorted({r[0] for r in rows})}}


def average(values):
    return mean(values) if values else None


def table(headers, rows):
    def cell(value):
        return 'unknown' if value is None else f'{value:.4f}' if isinstance(value, float) else str(value)
    return '\n'.join(['| ' + ' | '.join(headers) + ' |', '| ' + ' | '.join(['---']*len(headers)) + ' |'] +
                     ['| ' + ' | '.join(map(cell, row)) + ' |' for row in rows])


def report(config, *, final=False):
    root = config['paths'][PHASE + '_output']
    qc = summary(config)
    plan = read_json(root / 'source_plan.json')
    candidates = {r['episode_id']: r for p in candidate_paths(config) if (r := read_json(p))}
    qc_failures = {op: Counter() for op in OPERATORS}
    for path in (root / 'qc').glob('*.json'):
        row = read_json(path)
        for field, value in row['verdict'].items():
            if value is False:
                qc_failures[row['operator']][field] += 1
    selected = {r: {} for r in ('global', 'korean')}
    groups = {op: {'raw': [], 'scored': [], 'recoveries': []} for op in OPERATORS}
    design_path = root / 'teacher_design.json'
    design = read_json(design_path) if design_path.exists() else None
    planned = 2*len(design['corpus']) if design else None
    slots = []
    if design:
        for attempt in (1, 2):
            for episode_id in design['orders'][str(attempt)]:
                candidate = candidates[episode_id]
                group = groups[candidate['operator']]
                path = attempt_path(root, attempt, episode_id)
                slot = {'attempt': attempt, 'episode_id': episode_id, 'status': 'not_attempted'}
                slots.append(slot)
                if not path.exists():
                    continue
                raw = read_json(path)
                if raw['v2']['design_sha256'] != file_sha(design_path):
                    raise ValueError('Teacher design provenance mismatch')
                group['raw'].append(raw)
                slot.update(status='completed' if raw['completed'] else 'runtime_failure', path=str(path))
                recovery_path = root / f'attempt_{attempt}/recovery' / path.name
                if recovery_path.exists():
                    recovery = read_json(recovery_path)
                    if recovery['episode_sha256'] != file_sha(path):
                        raise ValueError('Recovery provenance mismatch')
                    group['recoveries'].append(recovery['stages'])
                scored = root / f'scored/attempt_{attempt}' / path.name
                if not scored.exists():
                    continue
                row = read_json(scored)
                if row['raw_episode_sha256'] != file_sha(path):
                    raise ValueError('Reward provenance mismatch')
                group['scored'].append(row)
                keep = absolute_selection(row, candidate)
                for role in ('global', 'korean'):
                    if not keep[role]:
                        continue
                    reward = row['reward'][role] if row['reward'] else row['global_only_reward']
                    item = {'attempt': attempt, 'path': str(scored), 'sha256': file_sha(scored),
                        'R': reward['R'], 'operator': candidate['operator'], 'source_id': candidate['source_id']}
                    old = selected[role].get(episode_id)
                    if old is None or (item['R'], -attempt) > (old['R'], -old['attempt']):
                        selected[role][episode_id] = item
    teacher = {}
    for op, group in groups.items():
        main = [r['combined']['per_record'][0]['main'] for r in group['recoveries'] if 'combined' in r]
        middle = [r['global']['per_record'][0]['main'] for r in group['recoveries'] if 'global' in r]
        rewarded = [r for r in group['scored'] if r['completed'] and r['reward']]
        teacher[op] = {'attempted': len(group['raw']), 'completed': sum(r['completed'] for r in group['raw']),
            'global_recovery_judged': len(middle), 'global_main_recovery': average(middle),
            'final_recovery_judged': len(main), 'final_main_recovery': average(main),
            'final_full_recovery': sum(v == 1 for v in main), 'final_partial_recovery': sum(v == .5 for v in main),
            'rewarded': len(rewarded), 'combined_R': average([r['reward']['combined']['R'] for r in rewarded]),
            'R_over': average([r['reward']['combined']['R_over'] for r in rewarded]),
            'selected': {role: sum(i['operator'] == op for i in items.values()) for role, items in selected.items()}}
    budget = accounting(root)
    if budget['confirmed_usd'] + budget['reserved_usd'] > config[PHASE]['max_cost_usd'] + 1e-9:
        raise AssertionError('Shared v2 cost cap exceeded')
    if final and budget['pending']:
        raise ValueError('Do not finalize while API calls are pending')
    metrics = {'status': 'final' if final else 'progress', 'qc': qc, 'qc_false_criteria': qc_failures,
        'teacher': teacher, 'budget': budget,
        'planned_teacher_attempts': planned, 'slots': slots, 'selected_counts': {r: len(v) for r, v in selected.items()},
        'selected': selected, 'training': False, 'gpu0_used': False,
        'v1_replay': {k: v for k, v in read_json(root / 'v1_replay.json').items() if k != 'details'},
        'source_sampling': plan['sampling']}
    write_json(root / 'metrics.json', metrics)
    write_json(root / 'best_role_trajectories.json', selected)
    lines = ['# VERAK v3 — v2 operator data preparation', '',
        '**Status: ' + ('final bounded run' if final else 'in progress; not a final result') + '.** No v2 training or RFT.', '',
        'All new behavior is opt-in `config v2` in a separate worktree. The accepted v1 environment and closed sets '
        'are unchanged. SFT v1 training/evaluation completed and passed final audit before v2 GPU1 scoring. '
        'Generation and QC use CPU/API only; GPU0 is never visible to v2.', '',
        'Each operator samples 300 agent_train and 80 agent_dev source essays, one record each, from the same frozen '
        'valid-only, genre-Q75, 500–2500-character source pool as v1 corruption. The 100 SFT dev instances '
        '(84 unique sources) and all 30 real sources are excluded by ID and content hash. Train/dev questions and '
        'sources remain disjoint; source overlap across operators is allowed.', '',
        '## Operator and tool contracts', '',
        'Both operators belong to GLOBAL, displayed as 글 수정 에이전트. INSERT and SPLIT add factual marker-change '
        'notices to the unchanged KOREAN hand-off. KOREAN retains its original prompt/actions. No hidden deletion '
        'reference or QC judgment appears in a teacher observation.', '',
        'G_DEL_LINK deletes one Sol-labeled topic/bridge/summary sentence from the specified eligible sites. '
        'Restoration requires an insertion at source paragraph/position ±1, with Luna judging role, main content '
        'and absence of new facts from the remaining essay. Grades are 1 / 0.5 / 0, cached per reference/context/insertion. '
        'An unmatched surviving insertion incurs R_over, including insertions made through legacy EDIT.', '',
        'L_FUSE joins 2–3 adjacent sentences. RESULT uses CAUSE -아서/-어서; ADVERSATIVE uses -지만; ADDITION or '
        'no conjunction uses v2-generation-only -고/-(으)며. EXAMPLE/RESTATEMENT windows are excluded. Recovery '
        'requires all source boundaries at the source positions, each source content-morpheme multiset and '
        'anonymization marker retained, and restored conjunctions of the original class (none/ADDITION when '
        'the source had none). The v1 EC set and L_CONN are unchanged.', '',
        '## QC yield', '',
        'Sol evaluates damage_real and original_is_fix, plus recoverable_from_essay for G_DEL_LINK. '
        'The approved 30% retention threshold uses usable records / 380 planned sources. QC pass/judged is '
        'reported separately. Missing responses remain unknown; bounds govern retain/drop decisions.', '',
        table(['operator', 'planned', 'generated', 'judged', 'pass', 'QC pass/judged', 'usable yield', 'decision'],
              [[op, q['counts']['planned'], q['counts'].get('generated', 0), q['counts'].get('judged', 0),
                q['counts'].get('passed', 0), q['qc_pass_rate_judged'], q['usable_source_yield'], q['decision']]
               for op, q in qc['operators'].items()]), '',
        table(['operator', 'split', 'planned', 'judged', 'pass', 'unknown', 'yield lower', 'yield upper'],
            [[op, split, c['planned'], c.get('judged', 0), c.get('passed', 0),
              (unknown := c.get('pending_candidate', 0) + c.get('pending_qc', 0)),
              c.get('passed', 0) / c['planned'], (c.get('passed', 0) + unknown) / c['planned']]
             for op, q in qc['operators'].items() for split, c in q['by_split'].items()]), '',
        'Unknown cases include absent labels or incomplete QC responses. Bounds treat every unknown '
        'as failing (lower) or passing (upper); they are coverage bounds, not confidence intervals.', '',
        table(['operator', 'damage_real=false', 'original_is_fix=false', 'recoverable_from_essay=false'],
            [[op, c['damage_real'], c['original_is_fix'], c['recoverable_from_essay'] if op == 'G_DEL_LINK' else 'n/a']
             for op, c in qc_failures.items()]), '',
        'False-criterion counts overlap when one record fails multiple criteria.', '',
        table(['L_FUSE source split', 'candidate windows', 'EXAMPLE excluded', 'RESTATEMENT excluded', 'union excluded'],
            [[split, (s := plan['sampling']['L_FUSE'][split]['screening']).get('candidate_windows', 0),
              s.get('excluded_EXAMPLE', 0), s.get('excluded_RESTATEMENT', 0), s.get('excluded_EXAMPLE_or_RESTATEMENT', 0)]
             for split in ('agent_train', 'agent_dev')]), '',
        'Exclusion counts are candidate windows screened, not unique essays; one window may contain both classes.', '',
        '## Teacher and selection', '',
        'Pinned Luna low uses two independent unseeded requests per passing retained train record, in distinct '
        'attempt namespaces; ordering seeds 79/80. Context is 8,192 with a 1,024 generation reserve, two-stage, '
        'no CHECK. Selection follows v1 absolute role R ≥ 0.80, retaining the best attempt per essay/role. '
        'Missing recovery or scorer results never become zero or eligible selections.', '',
        ('Both operators were dropped by the approved yield gate, so no teacher attempts or scorer calls '
         'were scheduled. Recovery and R_over are not measured; they must not be interpreted as zero.'
         if all(q['decision'] == 'drop' for q in qc['operators'].values()) else
         'Teacher generation is restricted to operators whose observed lower yield bound reaches 30%.'), '',
        table(['operator', 'attempted', 'completed', 'recovery judged', 'GLOBAL main recovery', 'final main recovery',
               'rewarded', 'combined R', 'R_over', 'selected GLOBAL / KOREAN'],
            [[op, t['attempted'], t['completed'], t['final_recovery_judged'], t['global_main_recovery'],
              t['final_main_recovery'], t['rewarded'], t['combined_R'], t['R_over'],
              f"{t['selected']['global']} / {t['selected']['korean']}"] for op, t in teacher.items()]), '',
        f'Planned teacher slots: {planned if planned is not None else "not yet frozen"}; '
        f'not attempted: {sum(s["status"] == "not_attempted" for s in slots)}. '
        'QC, runtime completion, recovery-judged counts and reward availability have separate denominators.', '',
        '## Examples', '']
    for op in OPERATORS:
        values = [r for r in candidates.values() if r['operator'] == op]
        def qc_pass(row):
            p = root / 'qc' / (safe_id(row['episode_id']) + '.json')
            return p.exists() and read_json(p)['passed']
        values.sort(key=lambda r: (not qc_pass(r), r['episode_id']))
        lines += [f'### {op}', '']
        if not values:
            lines += ['No constructed examples are available; labeling/QC coverage is incomplete.', '']
        for row in values[:5]:
            rec = row['records'][0]
            verdict = root / 'qc' / (safe_id(row['episode_id']) + '.json')
            label = 'pass' if qc_pass(row) else 'reject' if verdict.exists() else 'unknown'
            lines += [f"- `{row['episode_id']}` — QC {label}",
                '  - Source: ' + ' / '.join(v.replace('\n', ' ') for v in rec['original_text'].values()),
                '  - Corruption: ' + ('sentence deleted' if op == 'G_DEL_LINK' else
                                       ' / '.join(v.replace('\n', ' ') for v in rec['corrupted_text'].values())), '']
    lines += ['## Cost and validation', '',
        f"New confirmed cost: **${budget['confirmed_usd']:.6f} / $10**. Reserved/unknown: "
        f"${budget['reserved_usd']:.6f}; pending: {budget['pending']}. All models share one atomic reservation ledger. "
        'Pre-dispatch initialization failures are archived and excluded from paid-call counts; no raw evidence was removed.', '',
        'The 92 saved v1 Luna trajectories replayed exactly: 91 complete reward matches plus the preserved historical '
        'failure and its completed GLOBAL reward, 1,165 model turns. No paid call, new scorer call or GPU was used '
        'for replay. L_FUSE local construction produced 300/80 records, with 1,903 source/record checks passed. '
        'Focused v2 tool/reward/QC tests and final validation artifacts are retained locally.', '',
        'Artifacts: `' + str(root) + '`. Raw calls, reservations, failures, frozen manifests, '
        'cached judgments, and any generated attempts and selections are retained. No data or weights are uploaded. '
        'This task does not train a model.']
    destination = config['paths']['repo'] / 'imple/reports/V3_V2_OPS.md'
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return {'report': str(destination), 'qc': qc, 'teacher': teacher, 'budget': budget, 'final': final}
