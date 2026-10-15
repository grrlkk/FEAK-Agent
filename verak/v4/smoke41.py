"""Authorized B1: 20 new-source, one-attempt v4.1 smoke; never starts scale or training."""
from collections import Counter
import json
from pathlib import Path
import random
import sqlite3
import time

from .common import (REPO,GENRES,atomic_new,collection_lock,constrain_cpu,file_sha,load_config,
                     read_json,safe_id,sha_text,write_json)
from . import policy_prompts_v41 as prompts
from .runtime_v41 import VersionedAPI,make_plan,teach,export

SCALE=REPO/'verak/v4/outputs/scale'
ROOT=SCALE/'B1'
BASE_COMMIT='a1b62695d6bffd533d8ea233df00b33a5422b4aa'
PROTECTED=('policy_prompts.py','prompts/v4_editors.json','policy_env.py','content_env.py',
           'prep2_content.py','prep3_content.py','prep3_content_report.py','prep3_common.py')


def protected_hashes():
    base=Path(__file__).parent
    return {str(base/p):file_sha(base/p) for p in PROTECTED}|{
        str(REPO/'imple/FEAK_AGENT_METHOD.md'):file_sha(REPO/'imple/FEAK_AGENT_METHOD.md')}


def freeze_contract():
    authorization=read_json(SCALE/'authorization.json')
    approved=authorization['contract']['B']
    if (approved['smoke_sources'],approved['smoke_attempts'],approved['min_valid_action_rate'],
        approved['content_cap_usd_including_smoke'])!=(20,1,.9,50):
        raise ValueError('B1 authorization contract mismatch')
    if file_sha(REPO/'imple/FEAK_AGENT_METHOD.md')!=authorization['contract']['method_sha256']:
        raise ValueError('Protected method file changed')
    manifest=prompts.verify_frozen()
    budget={'schema_version':1,'content_cumulative_cap_usd':50.,
        'smoke':{'component':'B1','cap_usd':2.,'ledger':str(ROOT/'api/ledger.sqlite'),
                 'confirmed_and_reserved_count_toward_content_cap':True},
        'B2_remaining_cap_rule':'50 minus B1 confirmed_usd minus B1 reserved_usd',
        'B2_requires_B1_gate_pass':True,'B3_B4_require_B1_gate_pass':True,
        'authorization_sha256':file_sha(SCALE/'authorization.json')}
    target=SCALE/'budget_contract.json'
    if target.exists() and read_json(target)!=budget:
        raise ValueError('Scale budget contract changed')
    if not target.exists(): atomic_new(target,budget)
    value={'version':'v4.1_B1_smoke','base_commit':BASE_COMMIT,'prompt_version':prompts.VERSION,
        'prompt_manifest_sha256':file_sha(prompts.FROZEN),'prompts':manifest,
        'protected_files':protected_hashes(),'authorization_sha256':file_sha(SCALE/'authorization.json'),
        'budget_contract_sha256':file_sha(target),'sources':20,'attempts_per_source':1,
        'primary_gate':'valid returned action records / all returned action records >=0.90',
        'malformed_JSON_and_environment_invalid_in_denominator':True,
        'API_incomplete_errors_separate_and_all_scheduled_denominator_reported':True,
        'planner':'unchanged Dv3 Sol low','teacher':'pinned Luna low, independent unseeded requests',
        'revision_steps_per_delegation':6,'korean_whole_essay_steps':14,'workers':1,
        'feedback_or_scores_in_observation':False,'quality_judging':False,'training':False,
        'GPU_used':False,'new_scale_or_Wiki_calls':False}
    path=ROOT/'contract.json'
    if path.exists() and read_json(path)!=value:
        raise ValueError('B1 frozen execution contract changed')
    if not path.exists(): atomic_new(path,value)
    return value


def api_for(kind):
    from verak.v3.v2_ops.local import load_environment
    if kind not in {'sol','luna'}: raise ValueError('B1 has no other paid model')
    freeze_contract();config=load_config();load_environment(config)
    luna=read_json(config['paths']['phase4_output']/'models.json')['luna_model']
    phase='v41_B1_smoke'
    config[phase]={'model':'gpt-6.1-sol' if kind=='sol' else luna,'max_cost_usd':2.,
                   'max_concurrent_requests':2,'phase_api_ceiling':2000}
    config['paths'][phase+'_output']=ROOT
    api=VersionedAPI(config,2000,phase=phase);api.allowed_models={luna,'gpt-6.1-sol'}
    return api


def prior_cohorts():
    base=REPO/'verak/v4/outputs'
    paths=[base/'prep/design.json',base/'prep2/sample.json',base/'prep3/content/sample.json']
    metadata={};questions=set()
    for path in paths:
        value=read_json(path);metadata.update(value['source_metadata'])
        for key in ('selected_test_questions','test_question_hashes','eligible_test_questions'):
            questions.update(value.get(key,[]))
    return metadata,questions,{str(p):file_sha(p) for p in paths}


def freeze_sample():
    from verak.v3.common import extract_question_essay
    from .data import read_rows,genre_of,normalized_source
    freeze_contract();path=ROOT/'sample.json'
    if path.exists():
        saved=read_json(path)
        if file_sha(saved['input_path'])!=saved['input_sha256']:
            raise ValueError('B1 raw input changed')
        for p,digest in saved['prior_cohort_sha256'].items():
            if file_sha(p)!=digest: raise ValueError('Prior cohort identity changed')
        return saved
    prior,questions,hashes=prior_cohorts()
    forbidden_norm={r['normalized_source_hash'] for r in prior.values()}
    pool={g:[] for g in GENRES};seen=set();excluded=Counter();rawpath=REPO/'data/data_jsonl/train.jsonl'
    for number,raw in read_rows(rawpath):
        source=f'train:{number}';question,text=extract_question_essay(raw);norm=normalized_source(text)
        if source in prior or norm in forbidden_norm:
            excluded['prior_source_or_normalized_text']+=1;continue
        if sha_text(question) in questions:
            excluded['test_question']+=1;continue
        if norm in seen:
            excluded['duplicate_source']+=1;continue
        genre=genre_of(raw)
        if genre not in GENRES:
            excluded['unknown_genre']+=1;continue
        scores=raw.get('grader_1_scores',[])+raw.get('grader_2_scores',[])
        if len(scores)!=16 or any(type(x) not in (int,float) or not 1<=x<=5 for x in scores):
            excluded['invalid_human_scores']+=1;continue
        seen.add(norm);pool[genre].append({'source_id':source,'genre':genre,'question_hash':sha_text(question),
            'essay_hash':sha_text(text),'normalized_source_hash':norm,'human_mean':sum(scores)/16,'characters':len(text)})
    chosen=[];eligibility={}
    for offset,genre in enumerate(GENRES):
        values=sorted(pool[genre],key=lambda r:(r['human_mean'],r['source_id']))
        cutoff=values[(2*len(values)-1)//3]['human_mean']
        eligible=[r for r in values if r['human_mean']<=cutoff]
        random.Random(401+offset).shuffle(eligible);n=(7,7,6)[offset]
        if len(eligible)<n: raise ValueError('Insufficient new low/middle sources')
        chosen+=eligible[:n]
        eligibility[genre]={'eligible':len(values),'low_middle_cutoff':cutoff,'low_middle_count':len(eligible),'chosen':n}
    random.Random(409).shuffle(chosen)
    value={'version':prompts.VERSION,'input_path':str(rawpath),'input_sha256':file_sha(rawpath),
        'source_ids':[r['source_id'] for r in chosen],'source_metadata':{r['source_id']:r for r in chosen},
        'prior_cohort_sha256':hashes,'prior_distinct_sources':len(prior),'test_question_hashes':sorted(questions),
        'exclusions':dict(excluded),'eligibility':eligibility,'sampling_seeds':[401,402,403,409],
        'provider_sampling_seed':None,'human_sampling':'mean of 16 stored grader values on 1–5; lower two-thirds cutoff per genre; 7/7/6 genres',
        'no_quality_or_length_cherry_picking':True}
    atomic_new(path,value);return value


def materialize():
    from verak.v3.common import extract_question_essay
    from verak.v3.corrupt.document import Document
    from verak.v3.insertion_boost.resources import BoostParagraphs
    from verak.v3.v2_ops.local import load_environment
    from .data import read_rows,feedback_parts
    sample=freeze_sample();sources=set(sample['source_ids']);config=load_config();load_environment(config)
    analyzer=BoostParagraphs(config,cache_dir=ROOT/'bareun_sources')
    for number,raw in read_rows(sample['input_path']):
        source=f'train:{number}'
        if source not in sources: continue
        path=ROOT/'essays'/(safe_id(source)+'.json')
        if path.exists(): continue
        question,text=extract_question_essay(raw);profile=analyzer.profile(text);doc=Document.from_profile(text,profile)
        row={**sample['source_metadata'][source],'split':'train','question':question,'text':text,
            'profile':profile.to_dict(),'layout':doc.snapshot(),
            'paragraphs':[{'id':p.pid,'sentences':[{'id':u.sid,'text':u.text} for u in p.units]} for p in doc.paragraphs],
            'human_scores':[raw['grader_1_scores'],raw['grader_2_scores']],
            'feedback':feedback_parts(raw['assistant']),'feedback_origin':'strong_LLM','scorer_seen':True}
        if sha_text(text)!=row['essay_hash'] or len(row['feedback'])!=8:
            raise ValueError('B1 source materialization mismatch')
        atomic_new(path,row)
        write_json(ROOT/'materialization.json',{'saved':len(list((ROOT/'essays').glob('*.json'))),'planned':20,'at':time.time()})
    files={s:{'path':str(ROOT/'essays'/(safe_id(s)+'.json')),
              'sha256':file_sha(ROOT/'essays'/(safe_id(s)+'.json'))} for s in sample['source_ids']}
    path=ROOT/'source_files.json'
    if path.exists() and read_json(path)!=files: raise ValueError('B1 source artifacts changed')
    if not path.exists(): atomic_new(path,files)
    return [read_json(files[s]['path']) for s in sample['source_ids']]


def accounting():
    path=ROOT/'api/ledger.sqlite'
    if not path.exists(): return {'calls':0,'confirmed_usd':0.,'reserved_usd':0.,'pending':0,'by_stage':{}}
    with sqlite3.connect(f'file:{path}?mode=ro',uri=True) as db:
        rows=db.execute('SELECT stage,item_id,status,reserved,confirmed FROM calls').fetchall()
    return {'calls':sum(r[2]!='blocked_before_send' for r in rows),'confirmed_usd':sum(r[4] for r in rows),
        'reserved_usd':sum(r[3] for r in rows),'pending':sum(r[2]=='pending' for r in rows),
        'by_stage':{s:{'requests':sum(r[0]==s and r[2]!='blocked_before_send' for r in rows),
            'logical_requests':len({r[1] for r in rows if r[0]==s and r[2]!='blocked_before_send'}),
            'statuses':dict(Counter(r[2] for r in rows if r[0]==s)),
            'confirmed_usd':sum(r[4] for r in rows if r[0]==s)} for s in sorted({r[0] for r in rows})}}


def action_metrics(attempts,api_accounting):
    calls=[c for a in attempts for c in a.get('calls',[])]
    valid=sum(bool(c['action']['valid']) for c in calls)
    scheduled=sum(v['logical_requests'] for k,v in api_accounting['by_stage'].items() if 'teacher' in k)
    primary=valid/len(calls) if calls else None
    per_role={}
    for role in ('revision','korean'):
        own=[c for c in calls if c['role']==role]
        per_role[role]={'returned_actions':len(own),'valid_actions':sum(bool(c['action']['valid']) for c in own),
            'invalid_actions':sum(not c['action']['valid'] for c in own),
            'all_action_names':dict(Counter(c['action']['action'] for c in own)),
            'valid_action_names':dict(Counter(c['action']['action'] for c in own if c['action']['valid'])),
            'invalid_reasons':dict(Counter(c['action'].get('error',c['action'].get('reason','unspecified')) for c in own if not c['action']['valid']))}
    return {'returned_action_calls':len(calls),'valid_actions':valid,'invalid_actions':len(calls)-valid,
        'primary_valid_action_rate':primary,'all_scheduled_logical_editor_calls':scheduled,
        'valid_per_all_scheduled_editor_calls':valid/scheduled if scheduled else None,
        'scheduled_without_returned_action':scheduled-len(calls),'by_role':per_role,
        'primary_gate_pass':primary is not None and primary>=.9}


def report(tokenizer):
    sample=read_json(ROOT/'sample.json');plans=[];attempts=[];cases=[]
    for source in sample['source_ids']:
        plan=read_json(ROOT/'content/items'/(safe_id(source)+'.json'));plans.append(plan)
        attempt=read_json(ROOT/'content/attempts'/(safe_id(source)+'_a1.json'));attempts.append(attempt)
        cases.append({'source_id':source,'items':plan.get('items',[]),'attempt':attempt})
    account=accounting();metrics=action_metrics(attempts,account)
    entries=[d for a in attempts for d in a.get('delegations',[])]
    invalid=[{'source_id':a['source_id'],'role':c['role'],'delegation':c['delegation'],'turn':c['turn'],
        'raw':c['raw'],'action':c['action'],'phase_call':c['phase_call']} for a in attempts for c in a.get('calls',[]) if not c['action']['valid']]
    all_started=all(a.get('calls') for a in attempts)
    gate_pass=metrics['primary_gate_pass'] and len(attempts)==20 and all_started
    metrics.update(planned_sources=20,planned_attempts=20,saved_outcomes=len(attempts),
        actual_teacher_attempts=sum(bool(a.get('calls')) for a in attempts))
    metrics.update(prompt_version=prompts.VERSION,prompt_manifest_sha256=file_sha(prompts.FROZEN),
        plans_by_status=dict(Counter(p['status'] for p in plans)),attempt_status=dict(Counter(a['status'] for a in attempts)),
        protocol_complete=sum(bool(a.get('complete')) for a in attempts),
        revision_delegation_endings=dict(Counter(d['ending'] for d in entries)),
        revision_STOP_status=dict(Counter(d['stop_status'] for d in entries if d['ending']=='STOP')),
        korean_endings=dict(Counter(a.get('korean_stage',{}).get('ending','not_started') for a in attempts)),
        korean_STOP_status=dict(Counter(a.get('korean_stage',{}).get('stop_status') for a in attempts if a.get('korean_stage',{}).get('ending')=='STOP')),
        invalid_records=invalid,errors=[{'source_id':a['source_id'],'status':a['status'],'error':a.get('error')} for a in attempts if a['status']!='completed'],
        plans_errors=[p for p in plans if p['status']!='completed'],genres=dict(Counter(a['genre'] for a in attempts)),
        api=account,smoke_cap_usd=2.,content_cumulative_cap_usd=50.,
        B2_remaining_cap_usd=50-account['confirmed_usd']-account['reserved_usd'],
        gate={'minimum_valid_action_rate':.9,'primary_gate_pass':metrics['primary_gate_pass'],
              'all_20_teacher_attempts_started':all_started,'passed':gate_pass,'stop_all_B':not gate_pass},
        training=False,gpu_used=False,scorer_calls=0,quality_selection_performed=False)
    # Export is only a real-prefix/masking contract probe, never a quality selection or training launch.
    metrics['contract_export']=export(cases,tokenizer,output_root=ROOT/'export_probe')
    metrics['export_is_validation_only_not_training_selection']=True
    if protected_hashes()!=read_json(ROOT/'contract.json')['protected_files']:
        raise ValueError('Frozen v4/method file changed during B1')
    if account['pending'] or account['confirmed_usd']+account['reserved_usd']>2.+1e-9:
        raise ValueError('B1 unsettled call or cap violation')
    write_json(ROOT/'metrics.json',metrics);write_json(ROOT/'gate.json',metrics['gate'])
    write_json(ROOT/'invalid_actions.json',invalid)
    primary_display='NA (no returned action)' if metrics['primary_valid_action_rate'] is None else f"{metrics['primary_valid_action_rate']:.2%}"
    lines=['# B1 — v4.1 20-essay smoke','',
        f"Gate: {'PASS' if gate_pass else 'FAIL / STOP ALL B'}. Valid returned actions {metrics['valid_actions']}/{metrics['returned_action_calls']} "
        f"({primary_display}). Required >=90%. All 20 scheduled outcomes are retained; actual teacher starts {metrics['actual_teacher_attempts']}/20.",
        'A scale PASS also requires all 20 fixed teacher attempts to have at least one returned action; a planner/API failure cannot replace a smoke essay or disappear from coverage.','',
        'The two system prompts are versioned separately from frozen v4. Dv3 planner, delegation, environment validation and mandatory whole-essay Korean pass are unchanged. '
        'Scores are used only for source sampling. LLM feedback goes only to the unchanged planner, not editor observations. No quality selection, training, scorer or GPU call was made.','',
        '|Role|Body tokens|System-message tokens|SHA-256|','|---|---:|---:|---|']
    for role,value in prompts.verify_frozen()['roles'].items():
        lines.append(f'|{role}|{value["policy_tokens"]}|{value["system_message_tokens"]}|`{value["sha256"]}`|')
    lines += ['',f'Prompt version: `{prompts.VERSION}`. The old v4 prompt artifacts and `FEAK_AGENT_METHOD.md` hashes stayed identical.','',
        '|Role|Returned actions|Valid|Invalid|All action counts|Invalid reasons|','|---|---:|---:|---:|---|---|']
    for role,value in metrics['by_role'].items():
        lines.append(f'|{role}|{value["returned_actions"]}|{value["valid_actions"]}|{value["invalid_actions"]}|{json.dumps(value["all_action_names"],ensure_ascii=False)}|{json.dumps(value["invalid_reasons"],ensure_ascii=False)}|')
    lines += ['',f"All scheduled logical editor requests: {metrics['all_scheduled_logical_editor_calls']}; no returned action: {metrics['scheduled_without_returned_action']}. "
        f"Valid/all scheduled: {metrics['valid_per_all_scheduled_editor_calls']}. Incomplete/error HTTP outcomes remain in the per-stage ledger table below; they are not silently treated as valid actions.",'',
        '```json',json.dumps({'plans':metrics['plans_by_status'],'attempts':metrics['attempt_status'],
            'protocol_complete':metrics['protocol_complete'],'revision_endings':metrics['revision_delegation_endings'],
            'revision_STOP_status':metrics['revision_STOP_status'],'korean_endings':metrics['korean_endings'],
            'korean_STOP_status':metrics['korean_STOP_status'],'API_stages':account['by_stage'],'errors':metrics['errors']},ensure_ascii=False,indent=2),'```','',
        f"Source genres: {metrics['genres']}. Sampling seeds401/402/403, order409; new sources and normalized essays exclude PREP1/2/3, and all frozen test question hashes are excluded. "
        'Sampling uses the lower two-thirds human-score cutoff in each genre; no post-output source replacement.','',
        f"B1 cost: confirmed ${account['confirmed_usd']:.8f}, reserved ${account['reserved_usd']:.8f} / $2. "
        f"This is part of the $50 content cap; B2 remaining cap is ${metrics['B2_remaining_cap_usd']:.8f} including held reservations.",'',
        'Every rejected/malformed returned action is in `invalid_actions.json`; all raw API records are preserved. '
        'Action-only export files are contract probes, not accepted training selections. No 1,000/1,430-source or Wikipedia teacher workload was started.','']
    path=ROOT/'component_report.md';path.write_text('\n'.join(lines),encoding='utf-8')
    marker={'status':'passed' if gate_pass else 'failed','component':'B1','metrics_path':str(ROOT/'metrics.json'),
        'metrics_sha256':file_sha(ROOT/'metrics.json'),'report_path':str(path),'report_sha256':file_sha(path),
        'gate':metrics['gate'],'api':account,'B2_remaining_cap_usd':metrics['B2_remaining_cap_usd'],
        'no_live_paid_calls':True,'stopped':True,'gpu_used':False,'training':False}
    atomic_new(ROOT/'complete.json',marker);return marker


def run():
    constrain_cpu()
    with collection_lock(ROOT):
        if (ROOT/'complete.json').exists(): return read_json(ROOT/'complete.json')
        freeze_contract();rows=materialize()
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)
        manifest=prompts.freeze(tokenizer,load_config()['paths']['policy_base'])
        luna,sol=api_for('luna'),api_for('sol')
        try:
            luna.settle_interrupted()
            for n,row in enumerate(rows,1):
                plan=make_plan(row,sol,output_root=ROOT)
                if plan['status']=='completed':
                    attempt=teach(row,plan,1,luna,tokenizer,output_root=ROOT)
                else:
                    path=ROOT/'content/attempts'/(safe_id(row['source_id'])+'_a1.json')
                    attempt={'source_id':row['source_id'],'genre':row['genre'],'attempt':1,'status':'planner_'+plan['status'],
                        'error':plan.get('error'),'calls':[],'delegations':[],'complete':False,'prompt_version':manifest['version']}
                    if not path.exists(): atomic_new(path,attempt)
                write_json(ROOT/'status.json',{'stage':'smoke','processed':n,'planned':20,
                    'last_source':row['source_id'],'last_status':attempt['status'],'api':accounting(),'at':time.time()})
            result=report(tokenizer);write_json(ROOT/'status.json',result);return result
        finally:
            sol.close();luna.close()


def main():
    import argparse,os,subprocess,sys
    constrain_cpu();parser=argparse.ArgumentParser();parser.add_argument('command',choices=('freeze','run','launch'))
    args=parser.parse_args()
    if args.command=='freeze':
        result=freeze_sample()
        result={'sources':len(result['source_ids']),'eligibility':result['eligibility'],'sample_sha256':file_sha(ROOT/'sample.json')}
    elif args.command=='run': result=run()
    else:
        ROOT.mkdir(parents=True,exist_ok=True)
        if (ROOT/'complete.json').exists(): result=read_json(ROOT/'complete.json')
        else:
            marker=ROOT/'launch.json'
            if marker.exists():
                old=read_json(marker);p=Path('/proc')/str(old['pid'])/'cmdline'
                if p.exists() and b'verak.v4.smoke41' in p.read_bytes().split(b'\0'):
                    print(json.dumps(old));return
            with (ROOT/'worker.log').open('a') as log:
                process=subprocess.Popen([sys.executable,'-m','verak.v4.smoke41','run'],cwd=Path(__file__).resolve().parents[2],
                    env=dict(os.environ),stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
            result={'pid':process.pid,'worktree':str(Path(__file__).resolve().parents[2]),'command':'python -m verak.v4.smoke41 run','gpu_used':False}
            write_json(marker,result)
    print(json.dumps(result,ensure_ascii=False,indent=2))


if __name__=='__main__': main()
