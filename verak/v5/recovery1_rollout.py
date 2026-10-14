"""Bounded multi-step training rollouts when teacher-state errors are sparse."""
from collections import Counter, defaultdict
from copy import deepcopy
import argparse
import json
import time

from .recovery1 import (ROOT, prepare, load, verified_path, restore_seed, tokenizer,
    local_analysis_environment, load_config, Backend, read_json, atomic_new,
    write_json, ref, choose_diverse, key, encode, save_rows, REPLAY)
from .common import sha_text
from .common import REPO, public_row, verified, ReferenceScores, safe_id
from verak.v4.content_env import dumps
from verak.v4.render_v43 import assert_rendered


def state_key(sample, messages):
    p=assert_rendered(messages)
    tasks=[{k:v for k,v in t.items() if k!='item_id'} for t in p.get('assigned_tasks',[])]
    return sha_text(dumps([sample['source_id'],p['question'],p['paragraphs'],tasks]))


def teacher_index(rows):
    result=defaultdict(dict)
    for row in rows:
        target=json.loads(row['messages'][-1]['content'])
        if target['action']=='UNDO': continue  # Requires identical prior undo stack, not just text.
        if target['action']=='STOP' and target.get('status')=='done' and not target['issues']:
            target={**target,'summary':'현재 글의 한국어 형식 검토를 마쳤습니다.'}
        result[state_key(row,row['messages'][:-1])][dumps(target)]=target
    return {k:next(iter(v.values())) for k,v in result.items() if len(v)==1}


def prepare_rollouts():
    path=ROOT/'rollout_design.json'
    if path.exists(): return read_json(path)
    design=prepare(); rows=load(verified_path(design['training_source']))
    pool=[]
    for i,s in enumerate(rows):
        p=assert_rendered(s['messages'][:-1]); a=json.loads(s['messages'][-1]['content'])
        if (s['turn']==1 and p['steps_left']==14 and s['prefix_tokens']<=5300
                and a['action']=='EDIT'):
            pool.append({**s,'recovery_source_index':i})
    selected=choose_diverse(pool,48)
    result={'version':'recovery1_multistep_extension','created_at':time.time(),
        'source':design['training_source'],'selected_indices':[s['recovery_source_index'] for s in selected],
        'seeds':len(selected),'step_limit':14,'stop_at_first_rejection':True,
        'reason':'Single-action probes on familiar teacher states yielded few actual failures.',
        'labels':'Exact matching cleaned teacher state, or separately saved Codex review; '
            'no inferred label after an unmatched accepted change.',
        'new_API_calls':0,'test_cohort_untouched':True,'training_mix':'384 correction slots plus 384 replay; '
            'choose replay STOP count to retain about 30% STOP overall where available.'}
    atomic_new(path,result); return result


def collect():
    if (ROOT/'rollout_complete.json').exists(): return
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4.dryrun_student import prompt_token_audit
    design=prepare_rollouts(); rows=load(verified_path(design['source']))
    teachers=teacher_index(rows); local_analysis_environment(); tok=tokenizer()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'rollout_bareun')
    backend=Backend(); started=time.monotonic()
    try:
        for number,index in enumerate(design['selected_indices'],1):
            path=ROOT/'rollouts'/f'{index:05d}.json'
            if path.exists(): continue
            sample=rows[index]; env=restore_seed(sample,analysis)
            value={'source_index':index,'source_id':sample['source_id'],'calls':[],'candidates':[]}
            while env.remaining and not env.closed:
                messages=env.public_messages('korean')
                prefix=tok.apply_chat_template(messages,tokenize=True,add_generation_prompt=True)
                if len(prefix)+1024>8192:
                    value['ending']='training_context_limit'; break
                response=backend.generate(messages,'korean')
                if prompt_token_audit(len(prefix),response.get('usage'))['status']!='match':
                    raise RuntimeError('Serving template differs')
                # At the first action there is no replayed undo history; after a new edit there is.
                try: proposed=json.loads(response['raw'])
                except (ValueError,TypeError): proposed=None
                if isinstance(proposed,dict) and proposed.get('action')=='UNDO' and not value['calls']:
                    value.update(ending='unrestorable_undo_history',unapplied_response=response); break
                action=env.step(response['raw'],'korean')
                value['calls'].append({'messages':messages,'response':response,'action':deepcopy(action)})
                if not action['valid']:
                    candidates=[('before_rejection',messages)]
                    if not env.closed and env.remaining:
                        candidates.append(('after_rejection',env.public_messages('korean')))
                    for kind,observation in candidates:
                        target=teachers.get(state_key(sample,observation))
                        value['candidates'].append({'kind':kind,'messages':observation,
                            'teacher_match':target,'id':sha_text(dumps(observation))})
                    value['ending']='first_student_rejection'; break
                if action['action']=='STOP':
                    value['ending']='student_STOP'
                    # Matching a complete teacher snapshot supplies an independent STOP target.
                    target=teachers.get(state_key(sample,messages))
                    if target and target['action']=='STOP' and target.get('status')=='done':
                        value['candidates'].append({'kind':'student_visited_done','messages':messages,
                            'teacher_match':target,'id':sha_text(dumps(messages))})
            value.setdefault('ending','step_limit')
            atomic_new(path,value)
            print(f'multistep {number}/{len(design["selected_indices"])} {value["ending"]} '
                  f'calls={len(value["calls"])} candidates={len(value["candidates"])}',flush=True)
        atomic_new(ROOT/'rollout_complete.json',{'seconds':time.monotonic()-started,'seeds':len(design['selected_indices'])})
    finally: backend.close()


def pending():
    values=[]
    for path in sorted((ROOT/'rollouts').glob('*.json')):
        item=read_json(path)
        for c in item['candidates']:
            if c['teacher_match'] is None:
                values.append({'id':c['id'],'source_index':item['source_index'],
                    'source_id':item['source_id'],'kind':c['kind'],'messages':c['messages'],
                    'rollout':ref(path)})
    write_json(ROOT/'pending_teacher_review.json',values)
    print('unmatched states',len(values),flush=True)


def full_design():
    path=ROOT/'full_rollout_design.json'
    if path.exists(): return read_json(path)
    rows=load(verified_path(prepare()['training_source']))
    # Mine difficult TRAINING sources; diagnostic/test/validation exclusions are inherited.
    counts=Counter(s['source_id'] for s in rows if s['kind']=='B2'
                   and json.loads(s['messages'][-1]['content'])['action']=='EDIT')
    selected=sorted(counts,key=lambda sid:(-counts[sid],sha_text(sid)))[:16]
    cases=[]
    for sid in selected:
        source=REPO/'verak/v4/outputs/scale3/B2/essays'/(safe_id(sid)+'.json')
        cases.append({'case_id':sid,'row':ref(source),'teacher_edit_targets':counts[sid],
            'source_index':next(i for i,s in enumerate(rows) if s['source_id']==sid and s['kind']=='B2')})
    result={'cases':cases,'at':time.time(),'reason':'Include the actual student Revision-to-Korean handoff, '
        'which cannot be reproduced by starting from teacher handoffs.',
        'selection':'16 cleaned SFT training sources with most accepted Korean EDIT targets; hash tie break.',
        'test_cohort_untouched':True,'new_paid_calls':0,'roles':'All three original frozen agents, cooperate1 runtime'}
    atomic_new(path,result); return result


def full_collect():
    if (ROOT/'full_rollout_complete.json').exists(): return
    from verak.v4.environment_v43 import BoundaryParagraphs
    from verak.v4.fix1_infer import run_case
    local_analysis_environment(); tok=tokenizer(); backend=Backend()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'full_rollout_bareun')
    score=ReferenceScores(root=ROOT/'full_rollout_scores',fallback=REPO/'verak/v5/outputs/pipe1')
    started=time.monotonic()
    try:
        for number,case in enumerate(full_design()['cases'],1):
            result=run_case(public_row(verified(case['row'])),case_id=case['case_id'],output=ROOT/'full_rollouts',
                backend=backend,tok=tok,analysis=analysis,score=score,cooperation=True,capture_states=True)
            print(f'full training rollout {number}/16 {case["case_id"]} protocol={result["complete"]}',flush=True)
        atomic_new(ROOT/'full_rollout_complete.json',{'cases':16,'seconds':time.monotonic()-started})
    finally: backend.close(); score.close()


def full_candidates():
    """At most the first rejection and its real next observation per Korean stage."""
    output=[]
    for case in full_design()['cases']:
        path=ROOT/'full_rollouts/attempts'/(safe_id(case['case_id'])+'.json')
        if not path.exists(): continue
        value=read_json(path); seen_stages=set()
        for i,call in enumerate(value['calls']):
            if call['role']!='korean' or call['action']['valid'] or call['stage'] in seen_stages: continue
            seen_stages.add(call['stage'])
            choices=[('before_rejection',call)]
            if i+1<len(value['calls']):
                following=value['calls'][i+1]
                if following['role']=='korean' and following['stage']==call['stage']:
                    choices.append(('after_rejection',following))
            for kind,item in choices:
                messages=verified(item['prefix'])['messages']
                output.append({'id':sha_text(dumps(messages)),'source_id':case['case_id'],
                    'source_index':case['source_index'],'kind':'full_'+kind,'messages':messages,
                    'rollout':ref(path),'rejected_action':call['action']})
    return output


def validate_reviews():
    """Execute reviewed actions in the actual replayed stage, including real UNDO history."""
    from verak.v4.environment_v43 import BoundaryParagraphs
    review_reference=ref(ROOT/'teacher_review.json')
    reviewed=verified(review_reference)
    rows=load(verified_path(prepare()['training_source']))
    local_analysis_environment()
    analysis=BoundaryParagraphs(load_config(),cache_dir=ROOT/'teacher_validation_bareun')
    results={}
    def check_stage(index,calls,initial_messages):
        sample={**rows[index],'messages':initial_messages+[rows[index]['messages'][-1]]}
        env=restore_seed(sample,analysis)
        for i in range(len(calls)+1):
            if env.closed or not env.remaining: break
            messages=env.public_messages('korean'); identity=sha_text(dumps(messages))
            if i<len(calls):
                recorded=calls[i].get('messages') or verified(calls[i]['prefix'])['messages']
                if (messages[0]!=recorded[0] or assert_rendered(messages)!=assert_rendered(recorded)):
                    # Rendering key order can differ after restoring a dict. State must not.
                    write_json(ROOT/'replay_mismatch.json',{'index':index,'step':i,
                        'actual':messages,'expected':recorded})
                    raise ValueError('Stage replay observation differs at '+str(index)+':'+str(i))
                identity=sha_text(dumps(recorded))  # Training retains the original exact prefix bytes.
            annotation=reviewed.get(identity)
            if annotation and annotation.get('status')=='keep':
                fork=restore_seed(sample,analysis)
                for previous in calls[:i]:
                    action=fork.step(previous.get('response',previous)['raw'],'korean')
                    assert action['valid']==previous['action']['valid']
                before=fork.document.text
                result=fork.step(dumps(annotation['target']),'korean')
                results[identity]={'valid':result['valid'],'action':deepcopy(result),
                    'before_text':before,'after_text':fork.document.text,'replayed_actions':i}
            if i<len(calls):
                call=calls[i]; action=env.step(call.get('response',call)['raw'],'korean')
                assert action['valid']==call['action']['valid']
    for path in sorted((ROOT/'rollouts').glob('*.json')):
        value=read_json(path)
        if not any(c['id'] in reviewed for c in value['candidates']): continue
        if value['calls']: check_stage(value['source_index'],value['calls'],value['calls'][0]['messages'])
    wanted={c['source_id'] for c in full_candidates() if c['id'] in reviewed}
    for case in full_design()['cases']:
        if case['case_id'] not in wanted: continue
        value=read_json(ROOT/'full_rollouts/attempts'/(safe_id(case['case_id'])+'.json'))
        stages=defaultdict(list)
        for call in value['calls']:
            if call['role']=='korean': stages[call['stage']].append(call)
        for calls in stages.values():
            check_stage(case['source_index'],calls,verified(calls[0]['prefix'])['messages'])
    expected={k for k,v in reviewed.items() if v.get('status')=='keep'}
    if expected-set(results): raise ValueError('Unvalidated review targets: '+str(sorted(expected-set(results))))
    if any(not r['valid'] for r in results.values()): raise ValueError('A reviewed target is invalid')
    if ref(ROOT/'teacher_review.json')!=review_reference:
        raise ValueError('Review labels changed during execution validation')
    value={'review':review_reference,'results':results,'at':time.time()}
    write_json(ROOT/'teacher_validation.json',value)
    print('validated reviewed targets',len(results),flush=True)


def build_data():
    from .recovery1 import verify_data
    if (ROOT/'data/manifest.json').exists(): return verify_data()
    if not all((ROOT/p).exists() for p in ('collection_complete.json','rollout_complete.json','full_rollout_complete.json')):
        raise RuntimeError('All collection phases must finish')
    design=prepare(); extension=prepare_rollouts(); raw=load(verified_path(design['training_source']))
    tok=tokenizer(); corrections=[]; seen=set(); audit=Counter()
    reviewed=read_json(ROOT/'teacher_review.json') if (ROOT/'teacher_review.json').exists() else {}
    if any(x.get('status')=='keep' for x in reviewed.values()):
        validation=read_json(ROOT/'teacher_validation.json')
        if validation['review']!=ref(ROOT/'teacher_review.json'):
            raise ValueError('Reviewed labels changed after execution validation')
        required={k for k,v in reviewed.items() if v.get('status')=='keep'}
        if not required<=validation['results'].keys() or any(not validation['results'][k]['valid'] for k in required):
            raise ValueError('A kept review lacks successful execution validation')
    def add(index,messages,target,provenance,kind):
        identity=dumps([messages,target])
        if identity in seen: return
        try:
            row=encode(raw[index],messages,target,tok,{'kind':kind,'source_index':index,
                       'target_evidence':provenance})
        except OverflowError:
            audit['context_excluded']+=1; return
        row['recovery1']=row.pop('fix1'); corrections.append(row); seen.add(identity); audit[kind]+=1
    for index in design['selected_indices']:
        p=ROOT/'collection'/f'{index:05d}.json'; value=read_json(p)
        for call in value['calls']:
            if call['action']['valid']: continue
            audit['single_step_actual_rejections']+=1
            for name in ('messages','recovery_messages'):
                if name in call:
                    add(index,call[name],value['target'],{'collection':ref(p),'unchanged_teacher_document':True},
                        'single_step_'+name)
    for index in extension['selected_indices']:
        path=ROOT/'rollouts'/f'{index:05d}.json'; value=read_json(path)
        audit['rollout_'+value['ending']]+=1
        for c in value['candidates']:
            target=c['teacher_match']; evidence={'rollout':ref(path),'candidate_id':c['id']}
            if target is None:
                annotation=reviewed.get(c['id'])
                if not annotation or annotation.get('status')!='keep':
                    audit['unmatched_unlabeled']+=1; continue
                target=annotation['target']; evidence['review']=ref(ROOT/'teacher_review.json')
            else: evidence['exact_teacher_state_match']=True
            add(index,c['messages'],target,evidence,'multistep_'+c['kind'])
    for c in full_candidates():
        annotation=reviewed.get(c['id'])
        if not annotation or annotation.get('status')!='keep':
            audit['full_unmatched_unlabeled']+=1; continue
        add(c['source_index'],c['messages'],annotation['target'],
            {'rollout':c['rollout'],'candidate_id':c['id'],'review':ref(ROOT/'teacher_review.json')},c['kind'])
    failures=[s for s in corrections if s['recovery1']['kind']!='multistep_student_visited_done']
    done=[s for s in corrections if s['recovery1']['kind']=='multistep_student_visited_done']
    if len(corrections)<16 or len(failures)<8:
        raise RuntimeError('Insufficient correction coverage: '+str((len(corrections),len(failures))))
    # Prioritize all observed post-rejection states and retain the full unique pool up to the fixed cap.
    corrections.sort(key=lambda s:(not s['recovery1']['kind'].startswith('full_'),
                                  not s['recovery1']['kind'].startswith('multistep_'),key(s)))
    failures.sort(key=lambda s:(not s['recovery1']['kind'].startswith('full_'),key(s)))
    done.sort(key=key)
    done_slots=REPLAY//4 if done else 0
    corrective=[deepcopy(failures[i%len(failures)]) for i in range(REPLAY-done_slots)]
    corrective += [deepcopy(done[i%len(done)]) for i in range(done_slots)]
    stop_count=sum(json.loads(s['messages'][-1]['content'])['action']=='STOP' for s in corrective)
    replay_stop=max(96,min(288,round(.30*REPLAY*2)-stop_count))
    pools={a:[s for s in raw if json.loads(s['messages'][-1]['content'])['action']==a] for a in ('EDIT','STOP')}
    replay=choose_diverse(pools['STOP'],replay_stop)+choose_diverse(pools['EDIT'],REPLAY-replay_stop)
    training=corrective+deepcopy(replay)
    __import__('random').Random(71).shuffle(training)
    validation=load(verified_path(design['validation_source']))
    assert not ({s['source_group'] for s in training}&{s['source_group'] for s in validation})
    for part,rows in [('train',training),('validation',validation)]: save_rows(ROOT/f'data/korean.{part}.jsonl.gz',rows)
    manifest={'roles':{'korean':{part:{**ref(ROOT/f'data/korean.{part}.jsonl.gz'),'turns':len(rows)}
        for part,rows in [('train',training),('validation',validation)]}},
        'source_manifest':design['source_manifest'],'clean_training_source':design['training_source'],
        'clean_validation_source':design['validation_source'],'design':ref(ROOT/'design.json'),
        'rollout_design':ref(ROOT/'rollout_design.json'),'unique_corrective_states':len(corrections),
        'full_rollout_design':ref(ROOT/'full_rollout_design.json'),
        'teacher_validation':ref(ROOT/'teacher_validation.json') if reviewed else None,
        'teacher_review':ref(ROOT/'teacher_review.json') if reviewed else None,
        'unique_failure_states':len(failures),'unique_student_done_states':len(done),
        'failure_slots':REPLAY-done_slots,'student_done_slots':done_slots,
        'sampled_corrective_rows':len(corrective),'sampled_replay_rows':len(replay),'audit':dict(audit),
        'actions':dict(Counter(json.loads(s['messages'][-1]['content'])['action'] for s in training)),
        'train_sources':len({s['source_group'] for s in training}),'new_paid_calls':0,
        'adaptive_collection_note':'Teacher-state errors were sparse; scheduler paused before training for 48 actual multi-step rollouts.'}
    atomic_new(ROOT/'data/manifest.json',manifest)
    print(dumps({'unique_corrective_states':len(corrections),'audit':dict(audit),'actions':manifest['actions']}),flush=True)
    return manifest


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('part',choices=['prepare_rollouts','collect','pending','build_data','full_collect','validate_reviews'])
    globals()[parser.parse_args().part]()
