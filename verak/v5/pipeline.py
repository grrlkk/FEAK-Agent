"""SCORE/MAP -> revision -> rule notices -> scoped Korean repair, no learning."""
from copy import deepcopy
import json
import time

from .common import *
from . import detector
from verak.v4 import maps_v45, orchestrator_v44 as orch
from verak.v4.content_env import dumps, parse_json
from verak.v4.environment_v46 import V46Environment, CONTEXT, OUTPUT, REJECT_LIMIT
from verak.v4.environment_v43 import prepared_document
from verak.v4.render_v43 import assert_rendered, render
from verak.v4.scale2_projection import save_document, restore_document
from verak.v4.dryrun_student import document_state, prompt_token_audit
from verak.v4.fix1_infer import blocked_reason, paragraphs


class ScopedEnvironment(V46Environment):
    """V4.6 editor, with user-requested narrow Korean scope and oracle MOVE anchors.

    This wrapper is never used by A.a1. Frozen v4 behavior/files remain unchanged.
    """
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.oracle_move_targets=None
        self.narrow_korean=False

    def guard(self,value):
        if isinstance(value,dict) and value.get('action')=='MOVE' and self.oracle_move_targets is not None:
            if value.get('sentence') not in self.oracle_move_targets:
                raise ValueError('MOVE 대상은 해당 record의 문장/문단만 허용합니다.')
            dest=value.get('to')
            ids={u.sid for u in self.document.units}|{p.pid for p in self.document.paragraphs}
            if not isinstance(dest,dict) or any(x not in ids for x in dest.values()):
                raise ValueError('MOVE 목적지는 현재 글에 있는 앵커여야 합니다.')
            return
        return super().guard(value)

    def public_messages(self,role,steps_left=None):
        messages=super().public_messages(role,steps_left)
        payload=assert_rendered(messages)
        if role=='korean' and self.narrow_korean:
            payload['whole_essay_typo_spacing_pass']=False
            payload['task']='배정된 알림/항목과 허용 범위만 검토하고 필요한 형식 수정 후 STOP한다.'
            payload['scope']='assigned_locations_only'
        if role=='revision' and self.oracle_move_targets is not None:
            payload['MOVE_rule']={'editable_targets':sorted(self.oracle_move_targets),
                                 'destination_anchors':'all current sentence/paragraph IDs; no original destination given'}
        return render(payload,messages[0]['content'])

    def step(self,raw,role):
        if role=='korean':
            try: value=parse_json(raw)
            except (ValueError,TypeError): value=None
            if isinstance(value,dict) and value.get('action')=='EDIT' and value.get('sentence') not in self.scope:
                if self.closed or not self.remaining or self.active_role!=role:
                    raise RuntimeError('No live scoped delegation')
                self.steps_used+=1;self.rejected_in_a_row+=1
                record={'action':'EDIT','value':value,'valid':False,'scope_rejection':True,
                    'error':'배정된 위치 밖은 수정할 수 없습니다. 필요한 일은 STOP blocked로 보고하세요.',
                    'before_hash':sha_text(self.document.text),'after_hash':sha_text(self.document.text),
                    'text_changed':False,'changed_sids':[],'marker_change_notices':[]}
                if self.rejected_in_a_row>=REJECT_LIMIT:
                    self.closed=True
                    self.controller_stop={'action':'STOP','status':'blocked','summary':'연속 거절 3회',
                        'issues':[record['error']],'controller_generated':True,'reason':'three_consecutive_rejections','failure':True}
                    record['controller_stop']=deepcopy(self.controller_stop)
                self.actions[role].append(record);self.last_result=deepcopy(record)
                return record
        return super().step(raw,role)


def request(backend,tok,messages,role,path,*,limit=None,reserve=None):
    """Save exact input before dispatch; local backends audit chat-template counts."""
    limit=limit or CONTEXT[role];reserve=reserve or OUTPUT[role]
    prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
    prefix_path=path.with_suffix('.prefix.json');response_path=path.with_suffix('.response.json')
    if response_path.exists():
        saved=read_json(response_path)
        if verified(saved['prefix'])['messages']!=messages:
            raise ValueError('Cannot replay a response against different observations')
        return saved
    if prefix_path.exists():
        raise RuntimeError('Unfinished inference request preserved; do not silently regenerate')
    atomic_new(prefix_path,{'messages':messages,'prefix_input_ids':prefix,'prefix_tokens':len(prefix),
                            'role':role,'context_limit':limit,'output_limit':reserve})
    if len(prefix)+reserve>limit:
        value={'status':'context_limit','dispatched':False,'prefix':ref(prefix_path),'prefix_tokens':len(prefix)}
    else:
        try:
            response=backend.generate(messages,role,max_tokens=reserve)
            audit=prompt_token_audit(len(prefix),response.get('usage')) if response.get('generation',{}).get('model','').startswith('v5-') else None
            value={**response,'status':'completed','prefix':ref(prefix_path),'template_audit':audit}
            if audit and audit['status']!='match':value['status']='template_mismatch'
            total=len(tok.apply_chat_template(messages+[{'role':'assistant','content':response['raw']}],tokenize=True))
            value['full_tokens']=total
            if total>limit:value['status']='target_context_limit'
        except Exception as exc:
            from .api import BudgetStop
            if isinstance(exc,BudgetStop):raise
            value={'status':'request_error','error':type(exc).__name__+': '+str(exc),'prefix':ref(prefix_path)}
    atomic_new(response_path,value)
    return value


def stage(env,role,items,backend,tok,directory,*,number,scope=None,move_targets=None):
    if role=='revision':env.delegate(items,number)
    else:env.start_korean(items)
    env.narrow_korean=role=='korean' and scope is not None
    if scope is not None:env.scope=set(scope)
    env.oracle_move_targets=move_targets
    state={'role':role,'delegation':number,'items':deepcopy(items),'scope':sorted(env.scope),
           'initial':save_document(env.document),'calls':[],'ending':None,'stop_status':None}
    started=time.monotonic()
    while env.remaining and not env.closed:
        response=request(backend,tok,env.public_messages(role),role,directory/f't{env.steps_used+1:02d}')
        if response['status']!='completed':
            state['ending']=response['status'];state['calls'].append(response);break
        before=save_document(env.document)
        action=deepcopy(env.step(response['raw'],role))
        state['calls'].append({**response,'action':action,'before':before,'after':save_document(env.document)})
        if env.controller_stop:
            state.update(ending='controller_STOP_blocked',stop_status='blocked',report=deepcopy(env.controller_stop));break
        if action['valid'] and action['action']=='STOP':
            state.update(ending='STOP',stop_status=action['value']['status'],report=deepcopy(action['value']));break
    if state['ending'] is None:state['ending']='step_limit'
    state['no_protocol_error']=state['ending']=='STOP' and (state['stop_status']=='done' or
        state['stop_status']=='blocked' and blocked_reason(state.get('report',{})))
    state['all_done']=state['no_protocol_error'] and state['stop_status']=='done'
    state['final']=save_document(env.document);state['wall_seconds']=time.monotonic()-started
    saved_path=directory/'stage.json'
    if saved_path.exists():
        saved=read_json(saved_path)
        if saved['final']!=state['final'] or saved['ending']!=state['ending'] or saved['scope']!=state['scope']:
            raise ValueError('Saved worker stage replay differs')
        return saved
    atomic_new(saved_path,state)
    return state


def initial_map(row,backend,tok,analysis,score,directory):
    path=directory/'initial_map.json'
    if path.exists():return read_json(path)
    env=V46Environment(row,analysis);history=[];last=None;scores=None;phase='score';calls=[]
    failures=0;graph=None;started=time.monotonic()
    for n in range(8):
        messages=orch.messages(row['question'],paragraphs(env),phase=phase,scores=scores,
                               history=history,last_result=last)
        payload=json.loads(messages[1]['content'])
        payload['tools']={k:v for k,v in payload['tools'].items() if k in {'SCORE','MAP'}}
        payload['pipeline_contract']='이번 단계는 원문 SCORE와 MAP만 실행한다. DELEGATE/FINISH는 실행하지 않는다.'
        messages[1]['content']=dumps(payload)
        response=request(backend,tok,messages,'orchestrator',directory/f'o{n+1:02d}')
        if response['status']!='completed':calls.append(response);break
        record={'valid':False}
        try:
            value,diagnostics=maps_v45.validate(response['raw'],paragraphs(env))
            if value['action']!=('SCORE' if phase=='score' else 'MAP'):
                raise ValueError('Only original SCORE then MAP allowed')
            record.update(action=value['action'],value=value,**diagnostics)
            if phase=='score':
                result=score(row['question'],env.document.text)
                scores={'rubrics':['task','clarity','specificity','appropriateness','connection','unity','vocabulary','grammar'],
                        'expected':result['expected'],'integers':result['integers']}
                phase='map'
            else:
                graph,_=maps_v45.protected_map(value['map'],paragraphs(env))
            record['valid']=True;history.append(value);failures=0
        except (ValueError,TypeError,KeyError) as exc:
            record['error']=str(exc);failures+=1
        last=record;calls.append({**response,'action':record})
        if graph is not None or failures>=3:break
    value={'scores':scores,'map':graph,'calls':calls,'no_protocol_error':graph is not None,
           'original_document':save_document(env.document),'wall_seconds':time.monotonic()-started}
    atomic_new(path,value);return value


def onepass(row,backend,tok,analysis,directory):
    from verak.v3.rft1.oneshot import messages_for
    from .alignment import align
    path=directory/'onepass.json'
    if path.exists():return read_json(path)
    started=time.monotonic();original=prepared_document(row)
    messages=messages_for(row['question'],row['text'])
    count=len(tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True))
    reserve=min(4096,8192-count)
    if reserve<=0:
        response={'status':'context_limit','dispatched':False};final=original.clone();alignment=None
    else:
        response=request(backend,tok,messages,'oneshot',directory/'onepass_call',limit=8192,reserve=reserve)
        alignment=None;final=original.clone()
        if response['status']=='completed':
            try:final,alignment=align(original,response['raw'],analysis)
            except Exception as exc:response.update(status='alignment_exception',error=str(exc))
    complete=response['status']=='completed' and response.get('finish_reason')=='stop'
    result={'response':response,'initial':save_document(original),'final':save_document(final),
        'alignment':alignment,'no_protocol_error':complete,'generation_complete':complete,
        'wall_seconds':time.monotonic()-started,'actions':{'ONE_PASS':1 if response.get('raw') is not None else 0}}
    atomic_new(path,result);return result


def repair(row,revised,notices,graph,backend,tok,analysis,directory,*,lineage=None,declared_relations=None):
    path=directory/'repair.json'
    if path.exists():return read_json(path)
    started=time.monotonic();env=ScopedEnvironment(row,analysis)
    env.document=restore_document(revised);env.shared_map=deepcopy(graph)
    env.relations=deepcopy(declared_relations or {})
    env.executor.sentences={u.sid:u.sid for u in env.document.units}
    env.executor.paragraphs={p.pid:p.pid for p in env.document.paragraphs}
    if lineage:env.lineage=deepcopy(lineage)
    env.notices=deepcopy(notices);stages=[]
    for number,n in enumerate(notices,1):
        live={u.sid for u in env.document.units};scope=[s for s in n['scope'] if s in live]
        if n['sid'] not in live:
            stages.append({'notice_id':n['id'],'ending':'missing_flagged_sentence','no_protocol_error':False,'all_done':False,'calls':[]});continue
        item={'location':scope,'action':'EDIT','needs_search':'no',
              'instruction':'이 관찰은 오류 판정이 아니다. 바뀐 연결이 현재 문맥에서 잘못된 경우만 고친다. '
                '의도한 내용 수정을 되돌리지 않는다. '+n['message']+' 원래 그래프 관계: '+
                dumps(n['original_graph_relations'])+'; 수정 단계가 선언한 관계: '+dumps(env.relations.get(n['sid']))+
                '; 문제가 없으면 STOP done으로 보고한다.'}
        s=stage(env,'korean',[item],backend,tok,directory/f'n{number:03d}',number=number,scope=scope)
        stages.append({'notice_id':n['id'],**s})
    result={'stages':stages,'final':save_document(env.document),'lineage':deepcopy(env.lineage),
        'no_protocol_error':all(s['no_protocol_error'] for s in stages),
        'all_done':all(s['all_done'] for s in stages),'resolution':detector.resolved(notices,env.document),
        'wall_seconds':time.monotonic()-started}
    atomic_new(path,result);return result
