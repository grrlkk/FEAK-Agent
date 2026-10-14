"""D3 selections, action-only exports and retained-INSERT manual examples."""
from collections import Counter
from copy import deepcopy
import json
import random

from verak.v3.train.formatting import encode_labels
from .content_env import dumps
from .policy_prompts import assert_messages, verify_frozen
from .prep3_content import selection
from .prep3_common import (ROOT, REPO, GENRES, read_json, write_json, file_sha,
    safe_id, accounting, PROVENANCE)


def export(cases,tokenizer):
    output=ROOT/'content/export'; output.mkdir(parents=True,exist_ok=True)
    counts={}
    for role in ('revision','korean'):
        count=maximum=0; sources=set(); episodes=set()
        path=output/(role+'.jsonl')
        with path.open('w') as stream:
            for case in cases:
                for call in case['attempt']['calls']:
                    if call['role']!=role or not call['action']['valid']: continue
                    assert_messages(call['public_messages'],role)
                    messages=deepcopy(call['public_messages'])+[{'role':'assistant','content':call['raw']}]
                    encoded=encode_labels(tokenizer,messages,last_assistant_only=True)
                    prefix=tokenizer.apply_chat_template(call['public_messages'],tokenize=True,add_generation_prompt=True)
                    if (len(encoded['input_ids'])>8192 or encoded['input_ids'][:len(prefix)]!=prefix
                            or any(x!=-100 for x in encoded['labels'][:len(prefix)])
                            or encoded['labels'][len(prefix):]!=encoded['input_ids'][len(prefix):]):
                        raise ValueError('Frozen public action-only export contract failed')
                    public=dumps(call['public_messages'])
                    if any(i['item_id'] in public for i in case['items']) or any(k in public for k in ('LLM_rubric_feedback','LLM_feedback_reference','grader_1_scores','grader_2_scores','feedback_index')):
                        raise ValueError('Privileged feedback leaked into public observation')
                    stream.write(dumps({'source_id':case['source_id'],'attempt':case['attempt']['attempt'],
                        'role':role,'delegation':call['delegation'],'turn':call['turn'],
                        'prompt_version':verify_frozen()['version'],'messages':messages,**encoded})+'\n')
                    count+=1; maximum=max(maximum,len(encoded['input_ids']))
                    sources.add(case['source_id']); episodes.add((case['source_id'],case['attempt']['attempt']))
        counts[role]={'action_targets':count,'source_essays':len(sources),'attempts':len(episodes),
            'max_tokens':maximum,'path':str(path),'sha256':file_sha(path)}
    write_json(output/'contract.json',{'roles':counts,'trained':False,'observations_masked':True,
        'only_current_action_and_end_turn_have_loss':True,'raw_feedback_and_scores_removed':True,
        'public_tasks_retained':True,'frozen_prompts':verify_frozen(),'feedback_provenance':PROVENANCE})
    return counts


def review(kept):
    eligible=[c for c in kept if c['attempt']['surviving_insertions']]
    random.Random(263).shuffle(eligible)
    chosen=eligible[:15]
    public=[{k:v for k,v in c.items() if k!='attempt'}|{
        'attempt':c['attempt']['attempt'],'revised':c['attempt']['final_text'],
        'revision_stage':c['attempt']['phase_text'].get('revision'),
        'inserted_sentences':c['attempt']['surviving_insertions'],
        'delegations':c['attempt']['delegations'],'execution_status':c['attempt']['status'],
        'termination':c['attempt']['termination'],'korean_stage':c['attempt'].get('korean_stage')} for c in chosen]
    write_json(ROOT/'content/manual_insert_review_15.json',public)
    lines=['# V4 D v3: kept INSERT cases','',
        f'Random sample of kept attempts with at least one surviving INSERT, seed 263: {len(public)}/15 cases. '
        'Items and verdicts are LLM-written supervision, not a human standard.','']
    for n,c in enumerate(public,1):
        lines += [f'## {n}. {c["source_id"]}, attempt {c["attempt"]} ({c["genre"]})','',
            f'Question: {c["question"]}','','### Original','',c['original'],'','### Assigned items','']
        lines += [f'- {i["owner"]} {i["location"]} {i["action"]}: {i["instruction"]} (LLM-flagged: {i["problem"]})' for i in c['items']]
        lines += ['','### Revised','',c['revised'],'','### Retained inserted sentences','']
        lines += [f'- {s["id"]}: {s["text"]}' for s in c['inserted_sentences']]
        lines += ['','### Verdicts','','```json',json.dumps({k:c[k] for k in
            ('verdict','selection','preferred','execution_status','delegations','korean_stage')},ensure_ascii=False,indent=2),'```','']
    text='\n'.join(lines)+'\n'
    (ROOT/'content/manual_insert_review_15.md').write_text(text)
    (REPO/'imple/reports/V4_CONTENT_V3_INSERT_REVIEW_15.md').write_text(text)
    return {'eligible_kept_INSERT_attempts':len(eligible),'cases':len(public),
        'distinct_sources':len({c['source_id'] for c in public}),'shortfall':max(0,15-len(public))}


def report(tokenizer=None):
    design=read_json(ROOT/'content/design.json')
    cases=[]; attempts=[]; plans=[]; errors=[]; judgments=[]
    for source in design['source_ids']:
        row=read_json(ROOT/'essays'/(safe_id(source)+'.json'))
        pp=ROOT/'content/items'/(safe_id(source)+'.json')
        if not pp.exists(): continue
        plan=read_json(pp); plans.append(plan)
        if plan['status']!='completed': errors.append(plan); continue
        jp=ROOT/'content/judgments'/(safe_id(source)+'.json')
        judgment=read_json(jp) if jp.exists() else None
        if judgment and judgment['status']=='completed': judgments.append(judgment)
        elif judgment: errors.append(judgment)
        for a in (1,2):
            ap=ROOT/'content/attempts'/f'{safe_id(source)}_a{a}.json'
            if not ap.exists(): continue
            attempt=read_json(ap); attempts.append(attempt)
            if attempt['status']!='completed':
                errors.append({'source_id':source,'attempt':a,'status':attempt['status'],'error':attempt.get('error')})
            if judgment and judgment['status']=='completed':
                verdict=next(v for v in judgment['attempts'] if v['attempt']==a)
                cases.append({'source_id':source,'genre':row['genre'],'question':row['question'],'original':row['text'],
                    'items':plan['items'],'writer_notes':plan['writer_notes'],'attempt':attempt,'verdict':verdict,
                    'selection':selection(plan['items'],attempt,verdict),'preferred':judgment['preferred']})
    goodplans=[p for p in plans if p['status']=='completed']
    items=[i for p in goodplans for i in p['items']]
    revision_items=[i for i in items if i['owner']=='revision']
    kept=[c for c in cases if c['selection']['quality_keep']]
    delegations=[d for a in attempts for d in a['delegations']]
    insert_tasks=sum(i['action']=='INSERT' for i in items)
    calls=[call for a in attempts for call in a['calls']]
    def frac(n,d): return n/d if d else None
    metrics={'planned_sources':100,'planned_attempts':200,'plans_completed':len(goodplans),
        'attempts':len(attempts),'execution_status':dict(Counter(a['status'] for a in attempts)),
        'judged_pairs':len(judgments),'judged_attempts':len(cases),'kept_attempts':len(kept),
        'kept_sources':len({c['source_id'] for c in kept}),'kept_rate_all_planned':len(kept)/200,
        'kept_rate_judged':frac(len(kept),len(cases)),
        'kept_execution_status':dict(Counter(c['attempt']['status'] for c in kept)),
        'assigned_items':len(items),'revision_items':len(revision_items),'INSERT_tasks':insert_tasks,
        'essays_with_INSERT_task':sum(any(i['action']=='INSERT' for i in p['items']) for p in goodplans),
        'INSERT_task_share_all_items':frac(insert_tasks,len(items)),
        'INSERT_task_share_revision_items':frac(insert_tasks,len(revision_items)),
        'essays_with_INSERT_task_share':frac(insert_tasks,len(goodplans)),
        'attempts_with_successful_INSERT':sum(a['insert_count']>0 for a in attempts),
        'attempts_with_surviving_INSERT':sum(bool(a['surviving_insertions']) for a in attempts),
        'kept_with_surviving_INSERT':sum(bool(c['attempt']['surviving_insertions']) for c in kept),
        'max_successful_INSERTs':max((a['insert_count'] for a in attempts),default=0),
        'revision_delegations':len(delegations),'delegation_endings':dict(Counter(d['ending'] for d in delegations)),
        'delegation_STOP_rate':frac(sum(d['ending']=='STOP' for d in delegations),len(delegations)),
        'delegation_STOP_status':dict(Counter(d['stop_status'] for d in delegations if d['ending']=='STOP')),
        'korean_endings':dict(Counter(a.get('korean_stage',{}).get('ending','not_started') for a in attempts)),
        'notices_with_two_steps':sum(json.loads(c['public_messages'][1]['content'])['steps_left']==2 for c in calls),
        'valid_action_rate':frac(sum(c['action']['valid'] for c in calls),len(calls)),
        'actions_by_role':{r:dict(Counter(c['action']['action'] for c in calls if c['role']==r and c['action']['valid'])) for r in ('revision','korean')},
        'item_actions':dict(Counter(i['action'] for i in items)),'item_owners':dict(Counter(i['owner'] for i in items)),
        'planner_dropped':sum(len(p.get('dropped',[])) for p in plans),
        'mechanical_drops':dict(Counter(d['reason'] for p in plans for d in p.get('mechanically_dropped',[]))),
        'writer_notes':sum(len(p.get('writer_notes',[])) for p in plans),
        'gate_failures':dict(Counter(k for c in cases for k,v in c['selection']['criteria'].items() if not v)),
        'by_genre':{g:{'planned_attempts':68 if g=='argumentative' else 66,
            'attempts':sum(a['genre']==g for a in attempts),'judged':sum(c['genre']==g for c in cases),
            'kept':sum(c['genre']==g for c in kept),'INSERT_tasks':sum(any(i['action']=='INSERT' for i in p['items'])
                for p in goodplans if read_json(ROOT/'essays'/(safe_id(p['source_id'])+'.json'))['genre']==g)} for g in GENRES},
        'errors':errors,'api':accounting('content'),'feedback_provenance':PROVENANCE,'scorer_calls':0,'gpu_used':False,'training':False}
    write_json(ROOT/'content/selections.json',[{'source_id':c['source_id'],'attempt':c['attempt']['attempt'],**c['selection']} for c in cases])
    metrics['manual_review']=review(kept)
    if tokenizer is not None: metrics['export']=export(kept,tokenizer)
    write_json(ROOT/'content/metrics.json',metrics)
    return metrics
