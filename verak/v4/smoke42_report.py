"""Frozen 20-source v4.1/v4.2 action comparison; no model or analyzer calls."""
from collections import Counter
import json

from .common import atomic_new,file_sha,read_json,safe_id
from . import policy_prompts_v42 as prompts
from .smoke42 import ROOT,PRIOR,accounting,action_metrics,freeze_contract
from .runtime_v42 import export


def rejection_category(call):
    a=call['action'];error=a.get('error','')
    if a['valid']: return 'valid'
    if a['action']=='INVALID' or any(x in error for x in ('JSON','Extra data','Expecting','Unterminated','Invalid control')):
        return 'malformed_JSON'
    if '종결부호' in error or '문장 경계' in error: return 'new_terminal_or_boundary'
    if '두 문장' in error: return 'Revision_split_over_two'
    if '문장 수' in error:
        from .smoke41_report import classify_edit
        return classify_edit(call)['category'] if a['action']=='EDIT' else 'analyzer_sentence_count'
    if '익명화' in error: return 'mask_change'
    if '부분 문자열' in error: return 'substring_not_unique_or_missing'
    if '위치 밖' in error or '목적지도' in error: return 'outside_assigned_scope'
    if '현재 문장 ID' in error: return 'missing_sentence_ID'
    if '2회 상한' in error: return 'INSERT_cap'
    if '바꾸지 않는' in error: return 'no_op'
    if '필드' in error or '형식' in error: return 'action_schema'
    return 'other'


def summarize(attempts):
    calls=[c for a in attempts for c in a.get('calls',[])]
    valid=sum(c['action']['valid'] for c in calls)
    return {'valid':valid,'returned':len(calls),'invalid':len(calls)-valid,
        'valid_rate':valid/len(calls) if calls else None,
        'invalid_categories':dict(Counter(rejection_category(c) for c in calls if not c['action']['valid'])),
        'by_role_invalid_categories':{r:dict(Counter(rejection_category(c) for c in calls
            if c['role']==r and not c['action']['valid'])) for r in ('revision','korean')},
        'protocol_complete':sum(bool(a.get('complete')) for a in attempts),
        'attempt_status':dict(Counter(a['status'] for a in attempts))}


def gate_fields(metrics,all_started):
    passed=metrics['primary_gate_pass'] and all_started
    return {'minimum_valid_action_rate':.9,'primary_gate_pass':metrics['primary_gate_pass'],
        'all_20_teacher_attempts_started':all_started,'passed':passed,'stop_all_B':not passed,
        'stop_all_B_compatibility_scope':'B2/B3 only; B4 is independent under the latest user instruction',
        'stop_B2_B3':not passed,'B4_independent':True,
        'valid_actions':metrics['valid_actions'],'returned_actions':metrics['returned_action_calls']}


def report(tokenizer):
    if (ROOT/'complete.json').exists(): return read_json(ROOT/'complete.json')
    freeze_contract();sources=read_json(ROOT/'sample.json')['source_ids']
    attempts=[];old=[];cases=[];paths={}
    for source in sources:
        path=ROOT/'content/attempts'/(safe_id(source)+'_a1.json')
        attempt=read_json(path);attempts.append(attempt);paths[str(path)]=file_sha(path)
        old.append(read_json(PRIOR/'content/attempts'/(safe_id(source)+'_a1.json')))
        plan=read_json(ROOT/'content/items'/(safe_id(source)+'.json'))
        cases.append({'source_id':source,'items':plan['items'],'attempt':attempt})
    if len(attempts)!=20 or [a['source_id'] for a in attempts]!=sources: raise ValueError('All20 fixed outcomes are required')
    account=accounting()
    if account['pending'] or account['confirmed_usd']+account['reserved_usd']>2.+1e-9:
        raise ValueError('Unsettled API request or cap violation')
    metrics=action_metrics(attempts,account);all_started=all(a.get('calls') for a in attempts)
    gate=gate_fields(metrics,all_started);passed=gate['passed']
    entries=[d for a in attempts for d in a.get('delegations',[])]
    invalid=[{'source_id':a['source_id'],'role':c['role'],'delegation':c['delegation'],'turn':c['turn'],
        'category':rejection_category(c),'raw':c['raw'],'action':c['action'],'phase_call':c['phase_call']}
        for a in attempts for c in a.get('calls',[]) if not c['action']['valid']]
    paired=[{'source_id':new['source_id'],'v41':summarize([before]),'v42':summarize([new])}
        for before,new in zip(old,attempts)]
    splits=[{'source_id':a['source_id'],'role':c['role'],'delegation':c['delegation'],'turn':c['turn'],
        'notice':c['action']['split_notice'],'created_sids':c['action']['created_sids'],
        'retained_ids':[sid for sid in c['action']['created_sids']
            if any(s['id']==sid for p in a['final_paragraphs'] for s in p['sentences'])]}
        for a in attempts for c in a.get('calls',[]) if c['action'].get('split_notice')]
    prepared=read_json(ROOT/'prepared.json');prior_metrics=read_json(PRIOR/'metrics.json')
    total=prior_metrics['api']['confirmed_usd']+account['confirmed_usd']
    metrics.update(prompt_version=prompts.VERSION,prompt_manifest_sha256=file_sha(prompts.FROZEN),
        environment_sha256=prompts.environment_hashes(),planned_sources=20,planned_attempts=20,
        saved_outcomes=20,actual_teacher_attempts=sum(bool(a.get('calls')) for a in attempts),
        protocol_complete=sum(bool(a.get('complete')) for a in attempts),
        attempt_status=dict(Counter(a['status'] for a in attempts)),
        revision_delegation_endings=dict(Counter(d['ending'] for d in entries)),
        revision_STOP_status=dict(Counter(d['stop_status'] for d in entries if d['ending']=='STOP')),
        korean_endings=dict(Counter(a.get('korean_stage',{}).get('ending','not_started') for a in attempts)),
        korean_STOP_status=dict(Counter(a.get('korean_stage',{}).get('stop_status') for a in attempts if a.get('korean_stage',{}).get('ending')=='STOP')),
        invalid_records=invalid,errors=[{'source_id':a['source_id'],'status':a['status'],'error':a.get('error')}
            for a in attempts if a['status']!='completed'],genres=dict(Counter(a['genre'] for a in attempts)),
        normalization=prepared,planner_calls=0,plans_reused=20,revision_splits=splits,
        paired_sources=paired,paired_summary={'v41':summarize(old),'v42':summarize(attempts)},
        attempt_files_sha256=paths,api=account,smoke_cap_usd=2.,content_cumulative_cap_usd=50.,
        prior_v41_confirmed_usd=prior_metrics['api']['confirmed_usd'],content_cumulative_confirmed_usd=total,
        B2_remaining_cap_usd=50-total-account['reserved_usd'],gate=gate,
        training=False,gpu_used=False,scorer_calls=0,quality_selection_performed=False)
    metrics['contract_export']=export(cases,tokenizer,output_root=ROOT/'export_probe')
    metrics['export_is_validation_only_not_training_selection']=True
    # Recheck the old code, prompts, method and every old B1 artifact after export.
    freeze_contract()
    atomic_new(ROOT/'metrics.json',metrics);atomic_new(ROOT/'gate.json',gate)
    atomic_new(ROOT/'invalid_actions.json',invalid);atomic_new(ROOT/'paired.json',paired)
    p=lambda v:'NA' if v is None else f'{v:.2%}'
    lines=['# B1 — v4.2 paired 20-essay smoke','',
        f"**{'PASS' if passed else 'FAIL — stop B2/B3; B4 independent'}**: {metrics['valid_actions']}/{metrics['returned_action_calls']} returned actions valid "
        f"({p(metrics['primary_valid_action_rate'])}); required ≥90%. All20 fixed outcomes are saved; {metrics['actual_teacher_attempts']}/20 have returned editor actions.",'',
        'The same20 source essays, original order and saved Sol Dv3 assignments are reused. There are no new planner calls. '
        'The environment normalizes sentence boundaries before editing; source content is unchanged apart from whitespace. '
        'This is an action-validity check, not an essay-quality assessment or training selection. '
        'Under the latest user instruction, B4 is independent of this gate; the retained stop_all_B compatibility key refers only to B2/B3.','',
        '|Version|Valid / returned actions|Rate|Invalid|Protocol-complete essays|Attempt statuses|',
        '|---|---:|---:|---:|---:|---|']
    for name,value in metrics['paired_summary'].items():
        lines.append(f'|{name}|{value["valid"]}/{value["returned"]}|{p(value["valid_rate"])}|{value["invalid"]}|{value["protocol_complete"]}/20|{json.dumps(value["attempt_status"],ensure_ascii=False)}|')
    lines += ['',f"All scheduled logical editor requests: {metrics['all_scheduled_logical_editor_calls']}; without returned action: {metrics['scheduled_without_returned_action']}. "
        f"Valid/all scheduled: {p(metrics['valid_per_all_scheduled_editor_calls'])}. Malformed JSON and environment rejections stay in the primary denominator; API failures are separately included in scheduled coverage.",'',
        '|Role|Returned|Valid|Invalid|All action counts|','|---|---:|---:|---:|---|']
    for role,value in metrics['by_role'].items():
        lines.append(f'|{role}|{value["returned_actions"]}|{value["valid_actions"]}|{value["invalid_actions"]}|{json.dumps(value["all_action_names"],ensure_ascii=False)}|')
    lines += ['', '|Rejected action category|v4.1 Revision|v4.1 Korean|v4.2 Revision|v4.2 Korean|', '|---|---:|---:|---:|---:|']
    categories=sorted(set(metrics['paired_summary']['v41']['invalid_categories'])|set(metrics['paired_summary']['v42']['invalid_categories']))
    for category in categories:
        values=[metrics['paired_summary'][v]['by_role_invalid_categories'][r].get(category,0)
            for v in ('v41','v42') for r in ('revision','korean')]
        lines.append('|'+category+'|'+'|'.join(map(str,values))+'|')
    lines += ['', 'These are action-weighted counts; repeated rejected proposals are counted each time. '
        'Different trajectories can have different action counts. The v4.1 gate and artifacts were not adjusted.','',
        f"Revision delegation endings: {metrics['revision_delegation_endings']}; STOP statuses: {metrics['revision_STOP_status']}. "
        f"Korean endings: {metrics['korean_endings']}; STOP statuses: {metrics['korean_STOP_status']}.",'',
        '|Role|Body / system-message tokens|Prompt SHA-256|','|---|---:|---|']
    for role,value in prompts.verify_frozen()['roles'].items():
        lines.append(f'|{role}|{value["policy_tokens"]}/{value["system_message_tokens"]}|`{value["sha256"]}`|')
    lines += ['',f'Prompt version `{prompts.VERSION}`; environment `{prompts.verify_frozen()["environment_version"]}`. '
        'Korean is byte-identical to v4.1; Revision adds only “긴 문장은 EDIT로 두 문장까지 나눌 수 있다.” '
        'Both retain the mask-preservation and one-JSON-object instructions. Frozen environment hashes:', '']
    lines += [f'- `{name}`: `{digest}`' for name,digest in metrics['environment_sha256'].items()]
    lines += ['', 'Terminal `.?!` boundaries no longer depend on following spaces or Bareun sentence counts. '
        'Mask contents, decimal dots and URL/domain-internal punctuation are protected; consecutive punctuation is one boundary. '
        'URL protection is conservative until whitespace/quotes/brackets (trailing punctuation is terminal); ambiguous adjacent Korean text inside an unspaced URL is not reinterpreted. '
        'Rendered sentence units are separated by one space. Korean cannot add terminal punctuation/boundaries; Revision EDIT can split at most two sentences. '
        'Derived IDs carry parent/range provenance, inherit scope, generate notices and support MOVE/UNDO/Korean handoff.','',
        f"Initial normalization changed rendered text in {prepared['rendered_text_changed']}/20 essays; {prepared['original_units_split']} original units split into "
        f"{prepared['derived_initial_units']} additional units. All non-whitespace characters were preserved. "
        f"All13 archived whitespace-boundary rejections are fixed in the initial environment, not credited as editor actions. "
        f"All{prepared['tasks']} planner tasks preserve their exact evidence and original instruction; {prepared['mapped_tasks']} have an appended ID correspondence. "
        f"Executed Revision EDIT splits: {len(splits)}; new split IDs still present in final text: {sum(len(s['retained_ids']) for s in splits)}.",'',
        '|Source|v4.1 valid/returned|v4.1 invalid|v4.2 valid/returned|v4.2 invalid|', '|---|---:|---:|---:|---:|']
    for case in paired:
        a,b=case['v41'],case['v42'];lines.append(f'|{case["source_id"]}|{a["valid"]}/{a["returned"]}|{a["invalid"]}|{b["valid"]}/{b["returned"]}|{b["invalid"]}|')
    lines += ['', 'API outcomes and all episode errors:', '', '```json',json.dumps({'stages':account['by_stage'],'errors':metrics['errors']},ensure_ascii=False,indent=2),'```','',
        f"v4.2 smoke cost ${account['confirmed_usd']:.9f}, reserved ${account['reserved_usd']:.9f}, cap $2. "
        f"Prior v4.1 smoke ${metrics['prior_v41_confirmed_usd']:.9f}; cumulative content spend ${total:.9f}/$50. "
        f"Remaining content cap after reservations ${metrics['B2_remaining_cap_usd']:.9f}.",'',
        'No new quality judge, scorer, GPU, training, scale teacher or Wikipedia workload was started by this smoke. '
        'All invalid actions are saved in `invalid_actions.json`; per-source comparison is in `paired.json`. '
        'Valid-action export is only a masking/prefix contract check. Raw feedback/scores remain outside editor observations.','']
    report_path=ROOT/'component_report.md';report_path.write_text('\n'.join(lines),encoding='utf-8')
    marker={'status':'passed' if passed else 'failed','component':'B1_v42','metrics_path':str(ROOT/'metrics.json'),
        'metrics_sha256':file_sha(ROOT/'metrics.json'),'report_path':str(report_path),'report_sha256':file_sha(report_path),
        'gate_path':str(ROOT/'gate.json'),'gate_sha256':file_sha(ROOT/'gate.json'),'gate':gate,
        'api':account,'B2_remaining_cap_usd':metrics['B2_remaining_cap_usd'],'no_live_paid_calls':True,
        'stopped':True,'gpu_used':False,'training':False}
    atomic_new(ROOT/'complete.json',marker);return marker
