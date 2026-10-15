"""A4: public rule candidates replace form tasks in the unchanged full loop."""
from collections import Counter
from copy import deepcopy
import time

from .common import (ROOT as PIPE_ROOT, WORKTREE, Backend, ReferenceScores, public_row,
    read_json, atomic_new, write_json, verified, ref, safe_id, tokenizer,
    load_config, local_analysis_environment)
from .check_judge import ROOT, REPORT, freeze
from .check_outputs import require_gate
from .check_rules import candidates, LocalSpacing, COMPATIBILITY
from verak.v4.environment_v46 import V46Environment
from verak.v4.scale2_projection import (save_document, restore_document, project_document,
    private_document, ProjectionUnknown)


class RuleEnvironment(V46Environment):
    def __init__(self, *args, spacing, audit_dir, **kwargs):
        super().__init__(*args, **kwargs)
        self.spacing = spacing
        self.audit_dir = audit_dir
        self.rule_round = 0

    def start_korean(self, items=()):
        self.rule_round += 1
        suggested, unavailable = candidates(self.document, self.shared_map, self.spacing)
        evidence = {'round': self.rule_round, 'snapshot': save_document(self.document),
            'lineage': deepcopy(self.lineage), 'map': deepcopy(self.shared_map),
            'discarded_orchestrator_items': deepcopy(items), 'candidates': suggested,
            'spacing_unavailable': unavailable, 'private_records_opened': False}
        # Persist before calling the worker. O tasks remain an audit field only.
        atomic_new(self.audit_dir/f'pass{self.rule_round:02d}.json', evidence)
        super().start_korean([c['item'] for c in suggested])


def prepare():
    require_gate()
    cases = read_json(PIPE_ROOT/'A/manifest.json')['cases']
    value = {'cases': [{'case_id': c['case_id'], 'row': c['row']} for c in cases],
        'private_manifest_for_posthoc_only': ref(PIPE_ROOT/'A/manifest.json'),
        'environment': 'v4.6', 'temperature': 0., 'top_p': 1., 'seed': 47,
        'frozen_models': {'orchestrator': 2, 'revision': 1, 'korean': 2},
        'rule_sources': {p: ref(WORKTREE/'verak/v5'/p) for p in ('check_a4.py', 'check_rules.py')},
        'conjunction_MAP_compatibility': [{'relation': k[0], 'direction': k[1],
            'allowed': sorted(v)} for k, v in COMPATIBILITY.items()],
        'handoff': 'At each original Korean stage, replace O items by all current rule candidates. '
            'One item per candidate; unchanged 14-step whole-essay Korean stage, unchanged stage scheduling and final pass. '
            'No extra stages or steps. Original structural O delegations remain unchanged.',
        'rules': ['off-majority known sentence style', 'initial conjunction class versus directional previous-sentence MAP relation',
            'whitespace-only local Bareun AnalyzeSyntax refinement', 'literal repeated initial explicit subject with previous sentence'],
        'rule_scope': 'flagged sentence plus its current predecessor; Korean retains the original whole-essay pass permission',
        'spacing': 'CorrectError is unimplemented on the local server. AnalyzeSyntax auto_spacing/auto_jointing used only for '
            'candidate generation; unchanged base profile auto_spacing/auto_jointing=False. Whitespace-only, verified offsets.',
        'precision_contract': 'candidate type and primary target (not incidental predecessor scope) must match a record; '
            'deduplicate by operator and ancestral original target per essay across rounds; unmatched candidates are not certified errors',
        'training': False, 'paid_calls': False, 'test_cohort': False}
    freeze(ROOT/'step3/design.json', value)
    return value


def infer():
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4 import fix1_infer as full
    require_gate()
    marker = ROOT/'step3/inference_complete.json'
    if marker.exists():
        return
    start = time.monotonic(); local_analysis_environment()
    design = read_json(ROOT/'step3/design.json')
    for source in design['rule_sources'].values():
        if ref(source['path']) != source:
            raise ValueError('Frozen rule implementation changed')
    config = load_config(); tok = tokenizer()
    analysis = BoundaryParagraphs(config, cache_dir=ROOT/'bareun')
    spacing = LocalSpacing(config)
    backend = Backend(); score = ReferenceScores(root=ROOT, fallback=PIPE_ROOT)
    original = full.V46Environment
    try:
        for index, case in enumerate(design['cases'], 1):
            row = public_row(verified(case['row']))  # No corruption records here.
            audit_dir = ROOT/'step3/candidates'/safe_id(case['case_id'])
            full.V46Environment = lambda r, a: RuleEnvironment(r, a, spacing=spacing, audit_dir=audit_dir)
            result = full.run_case(row, case_id=case['case_id'], output=ROOT/'step3',
                backend=backend, tok=tok, analysis=analysis, score=score)
            write_json(ROOT/'step3/progress.json', {'done': index, 'total': len(design['cases']),
                'case_id': case['case_id'], 'at': time.time(), 'protocol': result['completion']['no_protocol_error']})
            print('a4', index, '/100', case['case_id'], result['completion']['no_protocol_error'], flush=True)
        atomic_new(marker, {'wall_seconds': time.monotonic()-start, 'episodes': len(design['cases']),
            'private_records_opened_during_generation': False, 'new_API_calls': 0,
            'new_local_spacing_calls': spacing.calls})
    finally:
        full.V46Environment = original
        backend.close(); score.close()


def ancestral_sid(sid, lineage):
    seen = set()
    while sid in lineage and sid not in seen:
        seen.add(sid)
        parent = lineage[sid]['parent_sid']
        if parent == sid:
            break
        sid = parent
    return sid


def candidate_audit(prepared, passes):
    from .measurement import record_public_ids
    gold = {r['record_id']: (r['op'], record_public_ids(prepared, r)) for r in prepared['corpus']['records']}
    unique = {}; unavailable = []
    for stage in passes:
        unavailable.extend(stage['spacing_unavailable'])
        for c in stage['candidates']:
            sid = ancestral_sid(c['sid'], stage['lineage'])
            key = (c['operator'], sid)
            matched = sorted(rid for rid, (op, ids) in gold.items() if op == c['operator'] and sid in ids)
            if key not in unique:
                unique[key] = {'operator': c['operator'], 'ancestral_sid': sid, 'matched_records': matched, 'occurrences': 0}
            unique[key]['occurrences'] += 1
    records = [{'record_id': rid, 'operator': op, 'detected': any(rid in c['matched_records'] for c in unique.values())}
        for rid, (op, _) in gold.items()]
    return {'candidates': list(unique.values()), 'records': records, 'spacing_unavailable': unavailable}


def report():
    from .measurement import OPERATORS
    from verak.v3.reward.recovery import recover_records
    from verak.v3.reward.overedit import overedit
    if (ROOT/'step3/report_complete.json').exists():
        return
    start = time.monotonic(); rows = []
    for case in read_json(PIPE_ROOT/'A/manifest.json')['cases']:
        prepared = verified(case['prepared']); gold = prepared['corpus']
        path = ROOT/'step3/attempts'/(safe_id(case['case_id'])+'.json')
        result = read_json(path)
        calls = [c for c in result['calls'] if c['role'] in {'revision', 'korean'}]
        value = {'case_id': case['case_id'], 'artifact': ref(path), 'operators': sorted({r['op'] for r in gold['records']}),
            'protocol': result['completion']['no_protocol_error'], 'completion': result['completion'],
            'actions': dict(Counter(c.get('action', {}).get('action', 'NOT_EXECUTED') for c in calls)),
            'invalid_actions': sum(not c.get('action', {}).get('valid', False) for c in calls)}
        passes = [read_json(p) for p in sorted((ROOT/'step3/candidates'/safe_id(case['case_id'])).glob('pass*.json'))]
        value['candidate_audit'] = candidate_audit(prepared, passes)
        try:
            final = private_document(project_document(restore_document(result['final_token_document']),
                prepared['policy_row']['normalization'], result['final']['lineage']), prepared['stable_id_aliases'])
            source, initial = (restore_document(prepared[k]) for k in ('reward_source', 'reward_initial'))
            value.update(recovery=recover_records(source, final, gold['records'], corrupted=initial),
                R_over=overedit(source, initial, final, gold['records']), projection_status='exact')
        except (ProjectionUnknown, KeyError) as exc:
            value.update(recovery=None, R_over=None, projection_status='unknown', projection_reason=str(exc))
        rows.append(value)
    metrics = {}; candidate_stats = {}
    for op in OPERATORS:
        episodes = [r for r in rows if op in r['operators']]
        recs = [r for e in episodes for r in (e['recovery'] or []) if r['op'] == op]
        overs = [e['R_over']['value'] for e in episodes if e['R_over'] is not None]
        actions = sum((Counter(e['actions']) for e in episodes), Counter())
        metrics[op] = {'episodes': len(episodes), 'records': len(recs), 'full_restoration': sum(r['main'] == 1 for r in recs),
            'mean_main': sum(r['main'] for r in recs)/len(recs) if recs else None,
            'R_over': sum(overs)/len(overs) if overs else None, 'actions': dict(actions),
            'actions_per_episode': sum(actions.values())/len(episodes) if episodes else None,
            'protocol_complete': sum(e['protocol'] for e in episodes), 'unknown': sum(e['recovery'] is None for e in episodes)}
        candidates_ = [c for r in rows for c in r['candidate_audit']['candidates'] if c['operator'] == op]
        records = [c for r in rows for c in r['candidate_audit']['records'] if c['operator'] == op]
        candidate_stats[op] = {'records': len(records), 'detected': sum(r['detected'] for r in records),
            'recall': sum(r['detected'] for r in records)/len(records) if records else None,
            'candidates': len(candidates_), 'matched': sum(bool(c['matched_records']) for c in candidates_),
            'precision': sum(bool(c['matched_records']) for c in candidates_)/len(candidates_) if candidates_ else None,
            'occurrences': sum(c['occurrences'] for c in candidates_)}
    atomic_new(ROOT/'step3/record_audit.json', rows)
    summary = {'operators': metrics, 'candidates': candidate_stats, 'episodes': len(rows),
        'protocol_complete': sum(r['protocol'] for r in rows),
        'spacing_unavailable': sum(len(r['candidate_audit']['spacing_unavailable']) for r in rows)}
    atomic_new(ROOT/'step3/metrics.json', summary)
    complete = read_json(ROOT/'step3/inference_complete.json')
    f = lambda x: 'n/a' if x is None else f'{x:.4f}'
    lines = ['## 3. A4: rule-based Korean hand-off', '',
        f'Inference wall-clock {complete["wall_seconds"]:.1f} s; post-hoc measurement {time.monotonic()-start:.1f} s. '
        f'Protocol completion: {summary["protocol_complete"]}/{len(rows)}. No API calls or training.', '',
        read_json(ROOT/'step3/design.json')['handoff'], '',
        'Candidates use only current public text/profile/MAP. Each points to a primary sentence plus its current predecessor. '
        'They are suggestions, not error verdicts. The source implementations and compatibility table were frozen before inference. '
        'The local Bareun CorrectError endpoint was unavailable (UNIMPLEMENTED); its AnalyzeSyntax spacing refinement supplied '
        'whitespace-only suggestions with verified offsets. The original profile and environment were unchanged.', '',
        '| Operator | Full restoration / measurable | Mean main | R_over | Worker actions / episode | Protocol / essays | Unknown |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for op, m in metrics.items():
        lines.append(f'| {op} | {m["full_restoration"]}/{m["records"]} | {f(m["mean_main"])} | {f(m["R_over"])} | '
            f'{f(m["actions_per_episode"])} | {m["protocol_complete"]}/{m["episodes"]} | {m["unknown"]} |')
    lines += ['', 'The same frozen v1 restoration/projection/over-edit calculations and protocol definition as PIPE1 A are used. '
        'Action and R_over rows are episode means among essays containing each operator; multi-record essays contribute to several rows.', '',
        '| Operator | Candidate recall / records | Recall | Matched / distinct candidates | Record-relative precision | Candidate occurrences |',
        '|---|---:|---:|---:|---:|---:|']
    for op, m in candidate_stats.items():
        lines.append(f'| {op} | {m["detected"]}/{m["records"]} | {f(m["recall"])} | {m["matched"]}/{m["candidates"]} | '
            f'{f(m["precision"])} | {m["occurrences"]} |')
    lines += ['', 'A match requires the same operator and primary target, not merely the predecessor scope. '
        'Distinct candidates deduplicate operator + ancestral target per essay across hand-offs. '
        'Unmatched candidates may expose natural errors or changes made by Revision, so record-relative precision is not linguistic precision. '
        'L_CONN, L_POLARITY and structural operators have no candidate rule in this intervention; their candidate recall is zero. '
        f'Spacing-unavailable sentence occurrences: {summary["spacing_unavailable"]}.', '']
    with REPORT.open('a') as out:
        out.write('\n'+'\n'.join(lines))
    atomic_new(ROOT/'step3/report_complete.json', {'measurement_seconds': time.monotonic()-start})


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part', choices=['prepare', 'infer', 'gpu', 'report'])
    part = parser.parse_args().part
    if part == 'prepare':
        print('Frozen', len(prepare()['cases']), 'public dev cases')
    elif part == 'gpu':
        from .check_local import gpu
        gpu('step3')
    elif part == 'infer':
        infer()
    else:
        report()
