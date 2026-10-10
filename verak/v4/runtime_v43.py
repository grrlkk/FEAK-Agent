"""v4.3 Dv3 teacher: identical state/step procedure, teacher-only first action."""
from copy import deepcopy
import json
from pathlib import Path
import time

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.v3.eval.api import Phase6API
from .common import load_config, safe_id, atomic_new, write_json, read_json
from .content_env import parse_json
from .paid import PrepAPI
from .runtime_v41 import bind
from .environment_v43 import V43Environment as V4Environment, BoundaryParagraphs as BoostParagraphs, ENVIRONMENT_VERSION
from . import policy_prompts_v43 as prompts
from .policy_prompts_v43 import assert_messages,verify_frozen
from .teacher_actions_v43 import first_action, VERSION as TRIMMING_VERSION

ROOT=Path('.')  # Every caller must bind its immutable component root.


def require_prepared(row):
    if row.get('environment_version')!=ENVIRONMENT_VERSION:
        raise ValueError('Call normalize_row and freeze paragraphs before planner or teacher')


def make_plan(row,api,*,output_root):
    from .prep3_content import make_plan as original
    require_prepared(row);prompts.verify_frozen()
    return bind(original,ROOT=output_root)(row,api)


def teach(row,plan,attempt,api,tokenizer,*,output_root,prompt_version=prompts.VERSION):
    require_prepared(row);prompts.verify_frozen(version=prompt_version)
    result=bind(teach_impl,ROOT=output_root)(row,plan,attempt,api,tokenizer)
    if result.get('prompt_version')!=prompt_version or result.get('teacher_trimming_version')!=TRIMMING_VERSION:
        raise ValueError('Cached teacher attempt has wrong v4.3 identity')
    for call in result.get('calls',[]): prompts.assert_messages(call['public_messages'],call['role'])
    return result


def export(cases,tokenizer,*,output_root,prompt_version=prompts.VERSION):
    from .prep3_content_report import export as original
    prompts.verify_frozen(version=prompt_version)
    canonical_cases=deepcopy(cases)
    for case in canonical_cases:
        for call in case['attempt']['calls']:
            if not call['action']['valid']: continue
            if first_action(call['raw'])[0]!=call['canonical_action']:
                raise ValueError('Canonical target is not the first saved raw teacher action')
            if parse_json(call['canonical_action'])!=call['action']['value']:
                raise ValueError('Canonical target differs from the one executed action')
            call['raw']=call['canonical_action']
    return bind(original,ROOT=output_root,assert_messages=prompts.assert_messages,
        verify_frozen=prompts.verify_frozen)(canonical_cases,tokenizer)


def teach_impl(row,plan,attempt,api,tokenizer):
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
            canonical,trim=first_action(raw)
            raw_total=len(tokenizer.apply_chat_template(public+[{'role':'assistant','content':raw}],tokenize=True))
            total=len(tokenizer.apply_chat_template(public+[{'role':'assistant','content':canonical}],tokenize=True))
            record=env.step(canonical,role)
            result['calls'].append({'role':role,'delegation':number,'turn':turn,'public_messages':public,
                'raw':raw,'canonical_action':canonical,'teacher_trim':{**trim,'first_action_valid':record['valid']},'raw_full_tokens':raw_total,'phase_call':response['phase_call'],'action':record,'prompt_tokens':prefix,'full_tokens':total})
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
    result.update(env.search.export(env.document.units))
    result['teacher_trimming_version']=TRIMMING_VERSION
    atomic_new(path,result)
    return result



class VersionedAPI(PrepAPI):
    def request(self,messages,*,stage,item_id,effort='low',max_output=2048,schema=None):
        from .common import sha_text
        prompts.verify_frozen()
        if 'teacher' in stage: prompts.assert_editor_request(messages)
        stage='v43_'+stage
        contract={'stage':stage,'item_id':item_id,'model':self.model,'reasoning_effort':effort,
            'max_output_tokens':max_output,'messages':messages,'schema':schema}
        fingerprint=sha_text(json.dumps(contract,ensure_ascii=False,sort_keys=True))
        with self.db() as db:
            prior=db.execute("SELECT id,status FROM calls WHERE fingerprint=? AND status!='blocked_before_send' ORDER BY id",(fingerprint,)).fetchall()
        if prior and not any(status=='completed' for _,status in prior):
            raise RuntimeError(f'Preserved v4.3 API outcome {prior}; no silent regeneration')
        return Phase6API.request(self,messages,stage=stage,item_id=item_id,effort=effort,max_output=max_output,schema=schema)
