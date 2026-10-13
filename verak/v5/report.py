"""V5 diagnostic tables with explicit protocol and judgment denominators."""
from collections import Counter
import statistics
import time

from .common import ROOT, append_report, read_json, ref, verified, write_json
from .detector import changes, resolved
from .pilot import CONDITIONS, cases, directory, outputs
from verak.v4.environment_v43 import prepared_document
from verak.v4.scale2_projection import restore_document


def mean(values):return statistics.mean(values) if values else None


def resolution_delta(initial,final):
    before={v['notice_id']:v['resolved_by_rule'] for v in initial}
    return {'notices':len(final),'initially_rule_satisfied':sum(v is True for v in before.values()),
            'finally_rule_satisfied':sum(v['resolved_by_rule'] is True for v in final),
            'newly_rule_satisfied':sum(v['resolved_by_rule'] is True and before[v['notice_id']] is not True for v in final),
            'finally_false':sum(v['resolved_by_rule'] is False for v in final),
            'finally_unknown':sum(v['resolved_by_rule'] is None for v in final)}


def measure(case):
    row=verified(case['row']);out=directory(case);values=outputs(case)
    initial=read_json(out/'original/initial_map.json');baseline=read_json(out/'original_score.json')
    original=prepared_document(row);records=[]
    for name in CONDITIONS:
        family='onepass' if name.startswith('onepass') else 'luna'
        value=values[name];notice=read_json(out/(family+'_notices.json'))
        final=restore_document(value['final']);score=read_json(out/(name+'_score.json'))
        resolution=resolved(notice['notices'],final)
        protocol=values[family]['no_protocol_error'] and value['no_protocol_error']
        if family=='luna':protocol=protocol and initial['scores'] is not None
        if name.endswith('feak'):protocol=protocol and initial['no_protocol_error']
        calls=[c for s in value.get('stages',[]) for c in s['calls']]
        actions=[c['action'] for c in calls if 'action' in c] if 'stages' in value else [a['action'] for a in value.get('actions',[]) if isinstance(a,dict)]
        # One-pass has a count dictionary, not an editor action list.
        if name=='onepass':actions=[]
        judge=read_json(out/f'judge_{family}.json');verdict=None;notice_verdicts=[]
        if judge['valid']:
            vid=next(k for k,v in judge['variant_key'].items() if v==name)
            verdict=next(v for v in judge['verdict']['variants'] if v['id']==vid)
            broken={v['id']:v for v in judge['verdict']['notices']}
            notice_verdicts=[{**v,'broken_after_revision':broken[v['id']]['broken_after_revision']} for v in verdict['notices']]
        good_score='mean' in score and 'mean' in baseline
        records.append({'case_id':case['case_id'],'cohort':case['cohort'],'condition':name,'family':family,
            'pipeline_protocol_complete':bool(protocol),'stage_protocol_complete':value['no_protocol_error'],
            'notices':len(notice['notices']),'notices_by_type':notice['counts'],
            'remaining_changes_by_type':dict(Counter(n['type'] for n in changes(original,final,initial['map']))),
            'resolution':resolution_delta(notice['initial_rule_resolution'],resolution),
            'delta_Q':score['mean']-baseline['mean'] if good_score else None,
            'rubric_deltas':[b-a for a,b in zip(baseline['expected'],score['expected'])] if good_score else None,
            'repair_delta_Q':score['mean']-read_json(out/(family+'_score.json'))['mean']
                if 'mean' in score and 'mean' in read_json(out/(family+'_score.json')) else None,
            'judge_valid':judge['valid'],'judge_verdict':verdict,'notice_verdicts':notice_verdicts,
            'actions':dict(Counter(a.get('action') for a in actions)),
            'invalid_actions':sum(not a.get('valid',False) for a in actions),
            'stage_endings':dict(Counter(s['ending'] for s in value.get('stages',[]))),
            'alignment_failed':bool(values[family].get('alignment',{}).get('alignment_failed'))
                if values[family].get('alignment') is not None else family=='onepass'})
    return records


def aggregate(rows):
    results={}
    for cohort in ['all',*sorted({r['cohort'] for r in rows})]:
        results[cohort]={}
        for name in CONDITIONS:
            chosen=[r for r in rows if r['condition']==name and (cohort=='all' or r['cohort']==cohort)]
            judged=[r for r in chosen if r['judge_valid']];ns=[n for r in judged for n in r['notice_verdicts']]
            broken=[n for n in ns if n['broken_after_revision']=='yes']
            results[cohort][name]={'essays':len(chosen),'protocol_complete':sum(r['pipeline_protocol_complete'] for r in chosen),
                'stage_protocol_complete':sum(r['stage_protocol_complete'] for r in chosen),
                'notices':sum(r['notices'] for r in chosen),
                'notices_by_type':dict(sum((Counter(r['notices_by_type']) for r in chosen),Counter())),
                'resolution':dict(sum((Counter(r['resolution']) for r in chosen),Counter())),
                'Q_scored':sum(r['delta_Q'] is not None for r in chosen),
                'mean_delta_Q':mean([r['delta_Q'] for r in chosen if r['delta_Q'] is not None]),
                'mean_repair_delta_Q':mean([r['repair_delta_Q'] for r in chosen if r['repair_delta_Q'] is not None]),
                'rubric_deltas':[mean([r['rubric_deltas'][i] for r in chosen if r['rubric_deltas'] is not None]) for i in range(8)],
                'judge_essays':len(judged),
                'essay_verdicts':{k:dict(Counter(r['judge_verdict'][k] for r in judged))
                    for k in ('better_than_original','meaning_preserved','fabrication')},
                'broken_after_revision':dict(Counter(n['broken_after_revision'] for n in ns)),
                'fixed_among_broken':dict(Counter(n['fixed_after_repair'] for n in broken)),
                'undo_intended_change':dict(Counter(n['undo_intended_change'] for n in ns)),
                'actions':dict(sum((Counter(r['actions']) for r in chosen),Counter())),
                'invalid_actions':sum(r['invalid_actions'] for r in chosen),
                'stage_endings':dict(sum((Counter(r['stage_endings']) for r in chosen),Counter())),
                'alignment_failed':sum(r['alignment_failed'] for r in chosen)}
    return results


def run():
    if (ROOT/'C/report_complete.json').exists():return
    started=time.monotonic();rows=[r for case in cases() for r in measure(case)];stats=aggregate(rows)
    write_json(ROOT/'C/metrics_rows.json',rows);write_json(ROOT/'C/metrics.json',stats)
    contract=read_json(ROOT/'api_contract.json');money=read_json(ROOT/'api/accounting.json')
    fmt=lambda v:'n/a' if v is None else f'{v:.4f}'
    lines=['The same smoke20 + agent_dev10 inputs are paired across six conditions. This is a diagnostic pilot, not test-cohort evaluation. '
        'All local model generation uses temperature 0. API settings and pinned model IDs are recorded below. '
        'Luna variants reuse one initial revision; local variants reuse one one-pass revision. '
        'FEAK and Luna-with-notices receive the exact same frozen notice set for a Luna revision. Luna self-check receives neither notices nor the original graph.',
        '',f"Luna: `{contract['luna']['model']}`; Claude judge: `{contract['claude']['model']}`. Full sampling and billing contract: `verak/v5/outputs/pipe1/api_contract.json`.",
        '', 'Q is the mean of the eight 1–9 expected rubric digits from the frozen GPU reference scorer. No CPU scores are mixed in. '
        'Its output and the Claude judgments are model-based diagnostics, not human annotations.',
        '', '|Cohort|Condition|Protocol / essays|Notices per essay|Rule satisfied initially / finally|Newly satisfied|Rule unknown|ΔQ vs original|ΔQ vs initial revision|',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for cohort,table in stats.items():
        for name,s in table.items():
            r=s['resolution'];lines.append(f'|{cohort}|{name}|{s["protocol_complete"]}/{s["essays"]}|'
                f'{fmt(s["notices"]/s["essays"])}|{r.get("initially_rule_satisfied",0)} / {r.get("finally_rule_satisfied",0)}|'
                f'{r.get("newly_rule_satisfied",0)}|{r.get("finally_unknown",0)}|{fmt(s["mean_delta_Q"])}|{fmt(s["mean_repair_delta_Q"])}|')
    lines+=['','A change notice is not a diagnosed error. Returning to the original phi value is observable but does not prove semantic correctness. '
        'Initially satisfied style rules are not credited as repairs. Unknown rule outcomes stay separate. '
        'Pipeline completion includes preceding revision and required SCORE/MAP stages; raw one-pass completion is an untruncated text response, '
        'list completion means all proposed list actions were valid, and worker repair requires a valid STOP report. '
        'STOP(blocked) with a reason is a protocol report; controller-forced stops, context limits and step limits fail protocol.',
        '', '|Condition|Notice type|Notices / 30 essays|', '|---|---|---:|']
    for name,s in stats['all'].items():
        for kind,count in sorted(s['notices_by_type'].items()):lines.append(f'|{name}|{kind}|{count}|')
    lines+=['','Claude comparisons are anonymized and their order is shuffled with a fixed seed derived from each case ID. '
        'No model names, reference answers, scorer values, or machine resolution decisions enter judge input. '
        'One family-level judgment of whether the initial link was broken is shared across its variants. '
        'Uncertain and not-applicable judgments are retained in JSON, and invalid/incomplete calls are not silently replaced.',
        '', '|Condition|Judged essays|Better yes|Meaning yes|Fabrication yes|Broken yes / judged notices|Fixed yes / broken|Undo yes / judged notices|',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,s in stats['all'].items():
        v=s['essay_verdicts'];broken=s['broken_after_revision'];total=sum(broken.values());n=s['judge_essays']
        lines.append(f'|{name}|{n}|{v["better_than_original"].get("yes",0)}|{v["meaning_preserved"].get("yes",0)}|'
            f'{v["fabrication"].get("yes",0)}|{broken.get("yes",0)}/{total}|{s["fixed_among_broken"].get("yes",0)}/{broken.get("yes",0)}|'
            f'{s["undo_intended_change"].get("yes",0)}/{total}|')
    lines+=['',f'Alignment failures: {stats["all"]["onepass"]["alignment_failed"]}/30 one-pass outputs. '
        'They remain visible in protocol and quality tables; notice locations on those outputs are uncertain. '
        'The A restoration table separately excludes failed alignments from its measurable denominator.',
        '', f'Paid API cost: ${money["confirmed_usd"]:.6f} confirmed; ${money["reserved_usd"]:.6f} outstanding reservations. '
        f'Provider breakdown: `{money["by_provider"]}`. No new training or test-cohort inference.',
        '', 'Full per-essay verdicts, per-type counts, score denominators, rubric deltas, invalid actions and stage endings: '
        '`verak/v5/outputs/pipe1/C/metrics_rows.json` and `C/metrics.json`.']
    seconds=sum(read_json(ROOT/'C'/p)['wall_seconds'] for p in ('local_complete.json','api_complete.json','repair_complete.json','judge_complete.json'))
    append_report('C — paired revise-then-repair pilot','\n'.join(lines),seconds=seconds)
    write_json(ROOT/'C/report_complete.json',{'measurement_wall_seconds':time.monotonic()-started,'reported_seconds':seconds})


if __name__=='__main__':run()
