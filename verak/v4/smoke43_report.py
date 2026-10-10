"""Immutable same-20 gate: executed-action validity and explicit trimming counts."""
from collections import Counter
import json

from .common import read_json,write_json,file_sha,safe_id
from .smoke41 import action_metrics
from .smoke42_report import rejection_category
from .teacher_actions_v43 import first_action


def trimming_metrics(attempts):
    result={'returned':0,'trimmed_total':0,'trimmed_accepted':0,'trimmed_invalid_first':0,
            'untrimmed_accepted':0,'untrimmed_invalid':0,'kinds':Counter(),'by_role':{}}
    for role in ('revision','korean'):
        result['by_role'][role]={'returned':0,'trimmed_total':0,'trimmed_accepted':0,
                               'trimmed_invalid_first':0}
    for attempt in attempts:
        for call in attempt['calls']:
            canonical,diagnostic=first_action(call['raw'])
            if canonical!=call['canonical_action'] or any(call['teacher_trim'].get(k)!=v for k,v in diagnostic.items()):
                raise ValueError('Saved trimming diagnostic does not replay')
            result['returned']+=1;result['by_role'][call['role']]['returned']+=1
            valid=call['action']['valid']
            if diagnostic['trimmed']:
                result['trimmed_total']+=1;result['kinds'][diagnostic['kind']]+=1
                key='trimmed_accepted' if valid else 'trimmed_invalid_first'
                result[key]+=1
                result['by_role'][call['role']]['trimmed_total']+=1
                result['by_role'][call['role']][key]+=1
            else: result['untrimmed_accepted' if valid else 'untrimmed_invalid']+=1
    result['kinds']=dict(result['kinds'])
    return result


def summarize(attempts):
    return {role:dict(Counter(rejection_category(call) for a in attempts for call in a['calls']
                            if call['role']==role and not call['action']['valid']))
            for role in ('revision','korean')}


def gate_fields(metrics,attempts):
    started=len(attempts)==20 and all(a.get('calls') for a in attempts)
    passed=bool(started and metrics['primary_gate_pass'])
    return {'minimum_valid_action_rate':.90,'passed':passed,'primary_gate_pass':metrics['primary_gate_pass'],
        'all20_saved':len(attempts)==20,'all20_started':started,
        'valid_actions':metrics['valid_actions'],'returned_actions':metrics['returned_action_calls'],
        'trimmed_outputs_counted_by_first_action_environment_validity':True,
        'stop_B2_B3':not passed,'stop_all_B':not passed,
        'downstream_launch_owner':'root; this component never launches scale collection'}


def report(tokenizer):
    from .smoke43 import ROOT,PRIOR41,PRIOR42,accounting,export,freeze_contract,immutable
    if (ROOT/'complete.json').exists(): return read_json(ROOT/'complete.json')
    freeze_contract();sources=read_json(ROOT/'sample.json')['source_ids']
    versions={};paths={'v41':PRIOR41,'v42':PRIOR42,'v43':ROOT}
    for name,root in paths.items():
        versions[name]=[read_json(root/'content/attempts'/f'{safe_id(source)}_a1.json') for source in sources]
    attempts=versions['v43'];api=accounting()
    if api['pending'] or api['reserved_usd']: raise ValueError('Settle all calls before final gate/report')
    if len(attempts)!=20: raise ValueError('Exactly the same20 saved outcomes are required')
    metrics=action_metrics(attempts,api)
    trim=trimming_metrics(attempts)
    metrics.update(planned_sources=20,planned_attempts=20,saved_outcomes=len(attempts),
        protocol_complete=sum(a['complete'] for a in attempts),
        attempt_status=dict(Counter(a['status'] for a in attempts)),
        teacher_trimming=trim,invalid_categories={v:summarize(a) for v,a in versions.items()},
        revision_delegation_endings=dict(Counter(d['ending'] for a in attempts for d in a['delegations'])),
        revision_STOP_status=dict(Counter(d['stop_status'] for a in attempts for d in a['delegations'] if d['ending']=='STOP')),
        korean_endings=dict(Counter(a.get('korean_stage',{}).get('ending','not_started') for a in attempts)),
        korean_STOP_status=dict(Counter(a['korean_stage']['stop_status'] for a in attempts if a.get('korean_stage',{}).get('ending')=='STOP')),
        errors=[{'source_id':a['source_id'],'status':a['status'],'error':a.get('error')} for a in attempts if a['status']!='completed'],
        api=api,search_calls=sum(c['action']['action']=='SEARCH' for a in attempts for c in a['calls']),
        sourced_inserts=sum(a.get('successful_sourced_inserts',0) for a in attempts),
        frozen_contract_sha256=file_sha(ROOT/'contract.json'),prepared=read_json(ROOT/'prepared.json'),
        context_audit_sha256=file_sha(ROOT/'context_audit.json'),gpu_used=False,training=False,
        scorer_calls=0,quality_selection_performed=False)
    paired=[]
    for source in sources:
        entry={'source_id':source}
        for version,rows in versions.items():
            attempt=next(a for a in rows if a['source_id']==source)
            entry[version]={'valid':sum(c['action']['valid'] for c in attempt['calls']),
                'returned':len(attempt['calls']),'protocol_complete':attempt['complete'],
                'status':attempt['status'],'invalid_categories':summarize([attempt])}
            if version=='v43': entry[version]['trimming']=trimming_metrics([attempt])
        paired.append(entry)
    metrics['paired']=paired
    metrics['attempt_files_sha256']={str(ROOT/'content/attempts'/f'{safe_id(s)}_a1.json'):
        file_sha(ROOT/'content/attempts'/f'{safe_id(s)}_a1.json') for s in sources}
    prior=read_json(ROOT/'budget_contract.json')['prior_v41_and_v42_confirmed_usd']
    metrics.update(content_cumulative_confirmed_usd=prior+api['confirmed_usd'],
        content_remaining_cap_usd=50-prior-api['confirmed_usd'],smoke_cap_usd=2.)
    cases=[{'source_id':a['source_id'],'items':read_json(ROOT/'content/items'/(safe_id(a['source_id'])+'.json'))['items'],
            'attempt':a} for a in attempts]
    metrics['contract_export']=export(cases,tokenizer,output_root=ROOT/'export_probe')
    metrics['export_is_validation_only_not_training_selection']=True
    gate=gate_fields(metrics,attempts)
    immutable(ROOT/'gate.json',gate);immutable(ROOT/'paired.json',paired)
    immutable(ROOT/'metrics.json',metrics)
    invalid=[{'source_id':a['source_id'],**c} for a in attempts for c in a['calls'] if not c['action']['valid']]
    immutable(ROOT/'invalid_actions.json',invalid)
    immutable(ROOT/'trimmed_actions.json',[{'source_id':a['source_id'],**c} for a in attempts for c in a['calls'] if c['teacher_trim']['trimmed']])
    lines=['# B1 — v4.3 same-20 smoke','',
        f'**{"PASS" if gate["passed"] else "FAIL"}**: {metrics["valid_actions"]}/{metrics["returned_action_calls"]} '
        f'environment-valid actions ({metrics["primary_valid_action_rate"]:.2%}); required ≥90%. '
        'All20 fixed outcomes are saved. Scale collection is released only by the root controller.','',
        'The same source order, exact v4.2 normalized text/profile/IDs and20 saved Sol plans are reused. '
        'No new planner calls or revised assignments. Teacher sampling is independent and unseeded. '
        'This checks execution validity; no essay-quality judge or training selection is performed.','',
        '|Version|Valid/returned|Rate|Invalid|Protocol complete|Attempt statuses|',
        '|---|---:|---:|---:|---:|---|']
    for version,rows in versions.items():
        n=sum(c['action']['valid'] for a in rows for c in a['calls']);den=sum(len(a['calls']) for a in rows)
        lines.append(f'|{version}|{n}/{den}|{n/den:.2%}|{den-n}|{sum(a["complete"] for a in rows)}/20|{dict(Counter(a["status"] for a in rows))}|')
    lines+=['',f'All scheduled editor requests: {metrics["all_scheduled_logical_editor_calls"]}; '
        f'without returned action: {metrics["scheduled_without_returned_action"]}; '
        f'valid/all scheduled: {metrics["valid_per_all_scheduled_editor_calls"]:.2%}. '
        'All invalid first actions remain in the denominator.','',
        '|Teacher output handling|Count|','|---|---:|',
        f'|Trimmed raw outputs|{trim["trimmed_total"]}|',
        f'|Trimmed, first action accepted by environment|{trim["trimmed_accepted"]}|',
        f'|Trimmed, first action invalid|{trim["trimmed_invalid_first"]}|',
        f'|Untrimmed, accepted|{trim["untrimmed_accepted"]}|',
        f'|Untrimmed, invalid|{trim["untrimmed_invalid"]}|','',
        'Only the first action is executed. The full raw response and discarded-action diagnostic remain saved; '
        'valid export targets use the canonical single action. Later raw actions are never executed or trained. '
        'Inference remains strict about one JSON action. Trimming does not bypass scope, role, mask, source or step guards.','',
        '|Role|Returned|Valid|Invalid|Action counts including rejections|','|---|---:|---:|---:|---|']
    for role,values in metrics['by_role'].items():
        lines.append(f'|{role}|{values["returned_actions"]}|{values["valid_actions"]}|{values["invalid_actions"]}|{values["all_action_names"]}|')
    categories=sorted({k for v in metrics['invalid_categories'].values() for role in v.values() for k in role})
    lines+=['','|Rejected category|v41 R|v41 K|v42 R|v42 K|v43 R|v43 K|','|---|---:|---:|---:|---:|---:|---:|']
    for category in categories:
        values=[metrics['invalid_categories'][v][r].get(category,0) for v in ('v41','v42','v43') for r in ('revision','korean')]
        lines.append('|'+category+'|'+'|'.join(map(str,values))+'|')
    lines+=['','Six archived v4.2 mask-preserving EDIT proposals pass the new token-byte/count guard; '
        'partial token changes and actual marker deletion remain invalid. Diagnostic replays are not new teacher actions. '
        'Previous gates/rewards/results are not recomputed. Repeated proposals count each time.','',
        f'Revision delegation endings: {metrics["revision_delegation_endings"]}; STOP status: {metrics["revision_STOP_status"]}. '
        f'Korean endings: {metrics["korean_endings"]}; STOP status: {metrics["korean_STOP_status"]}.','',
        f'SEARCH calls: {metrics["search_calls"]}; sourced INSERTs: {metrics["sourced_inserts"]}. '
        'SEARCH requires an active needs_search=yes public task alias. Korean prompt/rules remain unchanged.','',
        '|Source|v41 valid/returned|v42 valid/returned|v43 valid/returned|v43 trimmed accepted/total|',
        '|---|---:|---:|---:|---:|']
    for row in paired:
        values=[f'{row[v]["valid"]}/{row[v]["returned"]}' for v in ('v41','v42','v43')]
        t=row['v43']['trimming'];lines.append('|'+row['source_id']+'|'+'|'.join(values)+f'|{t["trimmed_accepted"]}/{t["trimmed_total"]}|')
    manifest=read_json(ROOT/'contract.json')['prompts']
    lines+=['','|Role|Body/system tokens|Prompt SHA256|','|---|---:|---|']
    for role,p in manifest['roles'].items(): lines.append(f'|{role}|{p["policy_tokens"]}/{p["system_message_tokens"]}|`{p["sha256"]}`|')
    lines+=['',f'Prompt version `{manifest["version"]}`; environment `{manifest["environment_version"]}`. '
        'The renderer serializes current full text/profile once, with complete journal, handoff and notice references; '
        'decoding is exactly equivalent and no essay/profile is truncated. Frozen implementation hashes:','']
    lines += [f'- `{name}`: `{sha}`' for name,sha in manifest['environment_sha256'].items()]
    context=read_json(ROOT/'context_audit.json')
    lines+=['','|Archived essay|Condition|Role|Old mandatory + update|v43 current prefix|+1024 output reserve|',
        '|---|---|---|---:|---:|---:|']
    lines += [f'|{c["source_id"]}|{c["condition"]}|{c["role"]}|{c["old_mandatory_plus_last_tokens"]}|{c["new_prefix_tokens"]}|{c["with_1024_output_reserve"]}|' for c in context['cases']]
    lines+=['','These are file/tokenizer-only current-state migration probes of the archived histories; '
        'they do not rerun or change v1 evaluation. Full IDs/text/order, public profile facts, journal and handoff are verified.','',
        f'Cost: new v4.3 ${api["confirmed_usd"]:.9f}, reserved ${api["reserved_usd"]:.9f}, '
        f'cap $2; prior two smokes ${prior:.9f}; cumulative content ${metrics["content_cumulative_confirmed_usd"]:.9f}/$50. '
        f'API calls {api["calls"]}; pending {api["pending"]}.','',
        f'All episode errors: {json.dumps(metrics["errors"],ensure_ascii=False)}.','',
        'No GPU, scorer, training or scale workload is launched by this component. '
        'Export is an action-only masking/prefix probe; raw teacher outputs stay in attempt and API artifacts.','']
    path=ROOT/'component_report.md';path.write_text('\n'.join(lines))
    complete={'status':'passed' if gate['passed'] else 'failed','component':'B1_v43',
        'metrics_path':str(ROOT/'metrics.json'),'metrics_sha256':file_sha(ROOT/'metrics.json'),
        'report_path':str(path),'report_sha256':file_sha(path),'gate_path':str(ROOT/'gate.json'),
        'gate_sha256':file_sha(ROOT/'gate.json'),'gate':gate,'api':api,
        'content_cumulative_confirmed_usd':metrics['content_cumulative_confirmed_usd'],
        'B2_remaining_cap_usd':metrics['content_remaining_cap_usd'],
        'no_live_paid_calls':True,'stopped':True,'gpu_used':False,'training':False}
    immutable(ROOT/'complete.json',complete)
    return complete
