"""D3 grounded INSERT planning and bounded teachers with frozen v4 editors."""
from concurrent.futures import ThreadPoolExecutor, as_completed
from copy import deepcopy
import json
import socket
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.insertion_boost.resources import BoostParagraphs
from .content import JUDGE as BASE_JUDGE, selection as base_selection
from .content_env import dumps
from .policy_env import V4Environment
from .policy_prompts import assert_messages, verify_frozen
from .prep2_content import (PLAN as PREVIOUS_PLAN, plan_schema,
    validate_plan as previous_validate_plan, judge_schema as previous_judge_schema,
    validate_judge as previous_validate_judge)
from .prep3_common import (ROOT, REPO, read_json, write_json, atomic_new, file_sha, sha_text,
    safe_id, api_for, accounting, load_config, collection_lock, freeze_contract, PROVENANCE)

PLAN=PREVIOUS_PLAN.replace(
    '다음 루브릭 피드백과 1–9 점수는 강한 LLM이 작성한 것으로 사람 교사의 정답이 아니다.',
    '다음 루브릭 피드백 문장은 강한 LLM이 작성한 것으로 사람 교사의 정답이 아니다. 점수는 작업 근거로 제공하지 않는다.'
).replace(
    'INSERT는 이 글 전체에서 최대 2문장이다. 새 사실이 필요한 작업은 배정하지 않는다.',
    '글의 기존 내용 사이에 빠진 설명이나 연결을 한 문장으로 보완하는 작업은 INSERT로 반드시 보존한다. '
    '이는 기존 문장을 재진술하는 것이 아니라 그 내용 사이의 빠진 관계를 명시하는 작업이어야 한다. '
    '그런 항목이 있으면 최대 4개 작업 안에서 하나를 포함한다. INSERT 작업은 글당 최대 하나다. '
    'instruction에 삽입 위치와 연결할 기존 내용을 명시하되 완성된 문장을 주지 않는다. '
    '새 사실·새 사례·글쓴이 경험을 요청하는 항목만 writer_notes로 보낸다. '
    '글 안의 내용으로 가능한 설명/연결 작업을 writer로 보내지 않는다. '
    '원문에 그런 문제가 없으면 INSERT 작업을 억지로 만들지 않는다.'
)
JUDGE=BASE_JUDGE+'''
피드백과 항목은 LLM이 작성한 제안이며 사람의 정답 기준이 아니다. 실제 원문/수정문을 근거로 독립 판단한다.
repetition_introduced: 원문에 이미 있는 내용의 재진술이나 불필요한 중복을 이번 수정이 새로 만들었는가(yes/no/unknown).
edits_introduced_awkwardness: 이번 수정 때문에 새로 생기거나 심해진 문맥/한국어 표현의 부자연스러움이 있는가(yes/no/unknown).
원문부터 있던 오류나 손대지 않은 학생 표현은 edits_introduced_awkwardness의 실패 사유가 아니다. 바뀐 표현과 인접 문맥을 원문과 대조하라.
개선된 부분이 있어도 새 중복/새 부자연스러움을 별도로 판정한다. STOP 보고 자체를 해결로 인정하지 않는다.'''


def validate_plan(value,row):
    value=previous_validate_plan(value,row)
    kept=[]; seen_insert=False
    for item in value['items']:
        if item['action']=='INSERT':
            if seen_insert:
                value['mechanically_dropped'].append({'problem':item['problem'],
                    'reason':'one_INSERT_task_per_essay','raw_item':item})
                continue
            seen_insert=True
        item['item_id']=item['item_id'].replace(':D2I',':D3I')
        kept.append(item)
    value['items']=kept
    return value


def make_plan(row,api):
    path=ROOT/'content/items'/(safe_id(row['source_id'])+'.json')
    if path.exists(): return read_json(path)
    payload={'question':row['question'],'genre':row['genre'],'paragraphs':row['paragraphs'],
        'LLM_rubric_feedback':row['feedback']}
    try:
        response=api.request([{'role':'system','content':PLAN},{'role':'user','content':dumps(payload)}],
            stage='prep3_content_plan',item_id=row['source_id'],max_output=4096,schema=plan_schema(row))
        result={'source_id':row['source_id'],'status':'completed','phase_call':response['phase_call'],
            **validate_plan(json.loads(response['raw']),row)}
    except CallBudgetExceeded:
        result={'source_id':row['source_id'],'status':'budget_stop','items':[]}
    except Exception as exc:
        result={'source_id':row['source_id'],'status':'error','items':[],
            'error':type(exc).__name__+': '+str(exc)}
    atomic_new(path,result)
    return result


def teach(row,plan,attempt,api,tokenizer):
    path=ROOT/'content/attempts'/f'{safe_id(row["source_id"])}_a{attempt}.json'
    if path.exists(): return read_json(path)
    env=V4Environment(row,BoostParagraphs(load_config(),cache_dir=ROOT/'bareun_content'))
    revision=[i for i in plan['items'] if i['owner']=='revision']
    korean=[i for i in plan['items'] if i['owner']=='korean']
    result={'source_id':row['source_id'],'attempt':attempt,'genre':row['genre'],'calls':[],
        'delegations':[],'termination':{},'status':'running','scorer_calls':0,'gpu_used':False,'phase_text':{},
        'prompt_version':verify_frozen()['version']}
    def stage(role,number,items):
        entry={'role':role,'delegation':number,'item_ids':[x['item_id'] for x in items],
            'steps':0,'ending':None,'stop_status':None}
        if role=='revision': result['delegations'].append(entry)
        else: result['korean_stage']=entry
        while env.remaining and not env.closed:
            turn=env.steps_used+1
            public=env.public_messages(role)
            assert_messages(public,role)
            prefix=len(tokenizer.apply_chat_template(public,tokenize=True,add_generation_prompt=True))
            if prefix>7168:
                entry['ending']='context_limit'; break
            response=api.request(public,stage='prep3_content_teacher',
                item_id=f'{row["source_id"]}:a{attempt}:{role}:d{number}:t{turn}',max_output=1024)
            raw=response['raw']
            total=len(tokenizer.apply_chat_template(public+[{'role':'assistant','content':raw}],tokenize=True))
            record=env.step(raw,role)
            result['calls'].append({'role':role,'delegation':number,'turn':turn,'public_messages':public,
                'raw':raw,'phase_call':response['phase_call'],'action':record,'prompt_tokens':prefix,'full_tokens':total})
            entry['steps']=turn
            write_json(ROOT/'content/partial'/f'{safe_id(row["source_id"])}_a{attempt}.json',
                {**result,'current_stage':entry,'current_text':env.document.text,'at':time.time()})
            if total>8192:
                entry['ending']='target_context_limit'; break
            if record['valid'] and record['action']=='STOP':
                entry.update(ending='STOP',stop_status=record['value']['status'],issues=record['value']['issues'])
                break
        if entry['ending'] is None: entry['ending']='step_limit'
        return entry
    try:
        for number,start in enumerate(range(0,len(revision),2),1):
            batch=revision[start:start+2]
            env.delegate(batch,number); stage('revision',number,batch)
        result['phase_text']['revision']=env.document.text
        result['termination']['revision']='STOP' if revision and all(d['ending']=='STOP' for d in result['delegations']) else 'no_items' if not revision else 'not_all_STOP'
        env.start_korean(korean); stage('korean',1,korean)
        result['termination']['korean']=result['korean_stage']['ending']
        result['phase_text']['korean']=env.document.text
        result['status']='completed'
    except CallBudgetExceeded:
        result.update(status='budget_stop',error='PREP3 content hard cap; request not dispatched')
    except Exception as exc:
        result.update(status='error',error=type(exc).__name__+': '+str(exc))
    for entry in result['delegations']+([result['korean_stage']] if 'korean_stage' in result else []):
        if entry['ending'] is None: entry['ending']=result['status']
    inserted={s for a in env.actions['revision'] if a['valid'] and a['action']=='INSERT' for s in a.get('created_sids',[])}
    result.update(final_text=env.document.text,actions=env.actions,insert_count=env.insert_count,
        declared_relations=env.relations,
        surviving_insertions=[{'id':u.sid,'text':u.text} for u in env.document.units if u.sid in inserted],
        final_paragraphs=[{'id':p.pid,'sentences':[{'id':u.sid,'text':u.text} for u in p.units]} for p in env.document.paragraphs],
        complete=result['status']=='completed' and result['termination'].get('revision') in {'STOP','no_items'} and result['termination'].get('korean')=='STOP')
    atomic_new(path,result)
    return result


def judge_schema(items):
    schema=previous_judge_schema(items)
    out=schema['properties']['attempts']['items']
    out['properties']['edits_introduced_awkwardness']=out['properties'].pop('reads_naturally')
    out['required']=['edits_introduced_awkwardness' if k=='reads_naturally' else k for k in out['required']]
    return schema


def validate_judge(value,items):
    translated=deepcopy(value)
    for a in translated['attempts']:
        if 'reads_naturally' in a:
            raise ValueError('Whole-essay naturalness is not a D3 judgment field')
        a['reads_naturally']=a.pop('edits_introduced_awkwardness')
    previous_validate_judge(translated,items)
    return value


def selection(items,attempt,verdict):
    result=base_selection(items,attempt,verdict)
    result['criteria'].update(repetition_no=verdict['repetition_introduced']=='no',
        introduced_awkwardness_no=verdict['edits_introduced_awkwardness']=='no')
    result['quality_keep']=result['export_keep']=all(result['criteria'].values())
    return result


def judge(row,plan,attempts,api):
    path=ROOT/'content/judgments'/(safe_id(row['source_id'])+'.json')
    if path.exists(): return read_json(path)
    payload={'question':row['question'],'original':row['paragraphs'],
        'LLM_feedback_reference':row['feedback'],'items':plan['items'],'writer_notes':plan['writer_notes'],
        'attempts':[{'attempt':a['attempt'],'revision_stage':a['phase_text'].get('revision'),
            'final_text':a['final_text'],'execution_status':a['status'],'delegations':a['delegations'],
            'korean_end':a.get('korean_stage')} for a in attempts]}
    try:
        response=api.request([{'role':'system','content':JUDGE},{'role':'user','content':dumps(payload)}],
            stage='prep3_content_judge',item_id=row['source_id'],max_output=4096,schema=judge_schema(plan['items']))
        result={'source_id':row['source_id'],'status':'completed','phase_call':response['phase_call'],
            **validate_judge(json.loads(response['raw']),plan['items'])}
    except CallBudgetExceeded:
        result={'source_id':row['source_id'],'status':'budget_stop'}
    except Exception as exc:
        result={'source_id':row['source_id'],'status':'error','error':type(exc).__name__+': '+str(exc)}
    atomic_new(path,result)
    return result


def run():
    from .prep3_data import materialize
    with collection_lock(ROOT/'content'):
        socket.getaddrinfo('api.openai.com',443); freeze_contract()
        rows=materialize('content')
        frozen={'source_ids':[r['source_id'] for r in rows],'attempts':2,'revision_steps':6,'korean_steps':14,
            'planner_max_items':4,'planner_max_INSERT_tasks':1,'environment_successful_INSERT_cap':2,
            'prompts_sha256':{'plan':sha_text(PLAN),'judge':sha_text(JUDGE),
                **{k:v['sha256'] for k,v in verify_frozen()['roles'].items()}},
            'input_manifest_sha256':file_sha(ROOT/'content/source_files.json'),
            'feedback_provenance':PROVENANCE,'public_tasks_retained':True,'raw_feedback_and_scores_in_observations':False,
            'manual_review':'15 random kept attempts with at least one surviving INSERT sentence; seed 263; no replacement if fewer'}
        path=ROOT/'content/design.json'
        if path.exists() and read_json(path)!=frozen: raise ValueError('D3 design changed')
        if not path.exists(): atomic_new(path,frozen)
        from transformers import AutoTokenizer
        tokenizer=AutoTokenizer.from_pretrained(str(load_config()['paths']['policy_base']),local_files_only=True)
        luna,sol=api_for('content'),api_for('content','sol')
        try:
            luna.settle_interrupted()
            def one(row):
                plan=make_plan(row,sol)
                if plan['status']!='completed': return
                attempts=[teach(row,plan,a,luna,tokenizer) for a in (1,2)]
                return judge(row,plan,attempts,sol)
            with ThreadPoolExecutor(max_workers=3) as pool:
                futures=[pool.submit(one,row) for row in rows]
                for n,future in enumerate(as_completed(futures),1):
                    future.result()
                    write_json(ROOT/'content/status.json',{'processed':n,'planned':100,'api':luna.accounting(),'at':time.time()})
            from .prep3_content_report import report
            report(tokenizer)
            write_json(ROOT/'content/complete.json',{'at':time.time(),'api':accounting('content')})
        finally:
            luna.close(); sol.close()
