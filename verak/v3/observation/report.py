"""Observation comparison report, with explicit missingness and immutable reuse."""
from collections import Counter, defaultdict
from statistics import mean
from pathlib import Path
import json

from ..common import read_json, write_json, file_sha
from ..phase2 import read_jsonl
from ..eval.api import Phase6API
from ..eval.summary import paired
from ..train.formatting import lengths
from ..train.pilot import safe_id
from ..train.pilot_report import action_type
from .experiment import PHASE, SETTINGS, prepare, graph_path, quality_graph
from .markers import episode_files, extract_cases
from .graph import render


def avg(values):
    values=list(values)
    return mean(values) if values else None


def fraction(n,d):
    return n/d if d else None


def stats(rows,tokenizer):
    result={'attempted':len(rows),'completed':sum(r['completed'] for r in rows),
        'errors':dict(Counter(r['runtime_error']['message'] for r in rows if r.get('runtime_error'))),
        'cost_usd':sum(r.get('confirmed_episode_cost',r['cost_usd']) for r in rows),
        'roles':{},'recovery':{}}
    for role in ('global','korean','combined'):
        values=[r['reward'][role] for r in rows if r.get('completed') and r.get('reward')]
        data={'reward_n':len(values),**{k:avg(v[k] for v in values) for k in ('R','R_rec','R_over','R_q','R_step')},
              **{k:avg(v['overedit'][k] for v in values) for k in ('morpheme','order')}}
        if role!='combined':
            acts=[a for r in rows for a in r['actions_by_role'].get(role,[])]
            calls=[c for r in rows for c in r['calls'] if c['role']==role]
            inputs=[len(tokenizer.apply_chat_template(c['messages'],tokenize=True,add_generation_prompt=True)) for c in calls]
            totals=[len(tokenizer.apply_chat_template(c['messages']+[{'role':'assistant','content':c['raw']}],tokenize=True)) for c in calls]
            data.update(steps=avg(r.get('steps',{}).get(role,0) for r in rows),
                invalid=sum(not a['valid'] for a in acts),actions=len(acts),
                invalid_rate=fraction(sum(not a['valid'] for a in acts),len(acts)),
                action_types=dict(Counter(action_type(a) for a in acts if a['valid'])),
                input_tokens=lengths(inputs),total_tokens=lengths(totals),
                truncations=sum(c.get('history_compacted',False) for c in calls),
                termination=dict(Counter(r['termination'].get(role,'not_completed') for r in rows)))
        result['roles'][role]=data
    by_op=defaultdict(list)
    for r in rows:
        if r.get('completed') and r.get('reward'):
            for rec in r['reward']['combined']['per_record']:
                by_op[rec['op']].append(rec)
    result['recovery']={op:{'n':len(v),'main':avg(x['main'] for x in v),'recovery':avg(x['recovery'] for x in v)} for op,v in by_op.items()}
    return result


def edge_set(saved,key='sentence_edges'):
    reverse={v:k for k,v in saved['sentence_ids' if key=='sentence_edges' else 'paragraph_ids'].items()}
    return {(reverse[e['source']],reverse[e['target']],e['label']) for e in saved['discourse'][key]}


def graph_quality(config,design,dev):
    def load(i,run=1):
        return quality_graph(config,i,run)
    checks=[]
    for i in design['quality_corrupted_ids']:
        row=dev[i];corrupt=load(i);source=load(row['source_id'])
        for rec in row['records']:
            op=rec['op']
            if op not in {'G_SENT_MOVE','G_PARA_SWAP','G_OFFTOPIC'}:continue
            r={'id':i,'record_id':rec['record_id'],'op':op,'pass':None}
            if corrupt['status']!='completed' or (op!='G_OFFTOPIC' and source['status']!='completed'):
                r['missing_reason']='graph_invalid_or_missing';checks.append(r);continue
            ce=edge_set(corrupt)
            if op=='G_OFFTOPIC':
                targets=set(rec['sids'])
                r['pass']=not any(a in targets or b in targets for a,b,_ in ce)
            elif op=='G_PARA_SWAP':
                r['pass']=ce==edge_set(source)
            else:
                sids=set(rec['sids']);se=edge_set(source)
                ct={b for a,b,_ in ce if a in sids};st={b for a,b,_ in se if a in sids}
                r.update(source_targets=sorted(st),corrupted_targets=sorted(ct),
                         both_empty=not ct and not st,pass_including_both_empty=ct==st)
                # Empty-to-empty is agreement but cannot show retention of a target.
                r['pass']=ct==st and bool(st)
            checks.append(r)
    summary={}
    for op in ('G_OFFTOPIC','G_SENT_MOVE','G_PARA_SWAP'):
        values=[r for r in checks if r['op']==op];observed=[r for r in values if r['pass'] is not None]
        summary[op]={'cases':len(values),'evaluable':len(observed),'passes':sum(r['pass'] for r in observed),
                     'rate':fraction(sum(r['pass'] for r in observed),len(observed)),
                     'missing':len(values)-len(observed)}
        if op=='G_SENT_MOVE':
            target_present=[r for r in observed if r['source_targets']]
            summary[op].update(source_has_target=len(target_present),
                retention_given_source_target=fraction(sum(r['pass'] for r in target_present),len(target_present)),
                both_empty=sum(r['both_empty'] for r in observed),
                exact_including_empty=fraction(sum(r['pass_including_both_empty'] for r in observed),len(observed)))
    stability=[]
    for i in design['stability_ids']:
        a,b=load(i,1),load(i,2)
        if a['status']!='completed' or b['status']!='completed':
            stability.append({'id':i,'status':'unavailable'});continue
        item={'id':i,'status':'completed'}
        for key in ('sentence_edges','paragraph_edges'):
            x,y=edge_set(a,key),edge_set(b,key)
            item[key]={'a':len(x),'b':len(y),'intersection':len(x&y),'union':len(x|y),
                'f1':2*len(x&y)/(len(x)+len(y)) if x or y else 1.,
                'jaccard':len(x&y)/len(x|y) if x or y else 1.,'exact':x==y}
        ra={r['id']:r['role'] for r in a['discourse']['roles']};rb={r['id']:r['role'] for r in b['discourse']['roles']}
        item['roles']={'n':len(ra),'agree':sum(v==rb.get(k) for k,v in ra.items())}
        stability.append(item)
    complete=[r for r in stability if r['status']=='completed']
    stab={'requested':100,'evaluable':len(complete),
          'policy_valid_pairs':sum(load(i,1).get('policy_graph_valid',False) and load(i,2).get('policy_graph_valid',False) for i in design['stability_ids'])}
    for key in ('sentence_edges','paragraph_edges'):
        intersection=sum(r[key]['intersection'] for r in complete)
        stab[key]={'micro_f1':fraction(2*intersection,sum(r[key]['a']+r[key]['b'] for r in complete)),
            'micro_jaccard':fraction(intersection,sum(r[key]['union'] for r in complete)),
            'macro_f1':avg(r[key]['f1'] for r in complete),'exact_set_rate':avg(r[key]['exact'] for r in complete)}
    stab['node_role_agreement']=fraction(sum(r['roles']['agree'] for r in complete),sum(r['roles']['n'] for r in complete))
    stab['role_nodes']=sum(r['roles']['n'] for r in complete)
    spot_rows=[];spot_status=Counter()
    for i in design['spot_ids']:
        p=config['paths'][PHASE+'_output']/'spot'/(safe_id(i)+'.json')
        if not p.exists():spot_status['not_run']+=1;continue
        saved=read_json(p);spot_status[saved['status']]+=1
        if saved['status']!='completed':continue
        edges={e['id']:e for e in saved['edges']}
        for v in saved['judgments']:
            spot_rows.append({'essay':i,**edges[v['id']],**v})
    def count(values):
        n=len(values);counts=Counter(v['verdict'] for v in values)
        return {'n':n,**{k:counts[k] for k in ('correct','incorrect','unclear')},
                'correct_share':fraction(counts['correct'],n)}
    spot={'requested':50,'essay_status':dict(spot_status),'overall':count(spot_rows),
        'by_type':{k:count([r for r in spot_rows if r['kind']+':'+r['label']==k])
                   for k in sorted({r['kind']+':'+r['label'] for r in spot_rows})}}
    files=list((config['paths'][PHASE+'_output']/'graphs').glob('*.json'))
    validation=Counter(read_json(p)['status'] for p in files)
    return {'raw_extraction_status':dict(validation),'corruption_checks':summary,'corruption_details':checks,'stability':stab,
            'stability_details':stability,'spot':spot,'spot_edges':spot_rows}


def marker_summary(config):
    actions,cases=extract_cases(config)
    root=config['paths'][PHASE+'_output'];judgments={}
    for p in (root/'marker_judgments').glob('*.json'):
        row=read_json(p)
        if row['status']=='completed':judgments[row['case_id']]=row['judgment']
    result={};details=[]
    for a in actions:
        evaluated=[cid for cid in a['case_ids'] if cid in judgments]
        bad=[cid for cid in evaluated if any(not v['still_fits'] for v in judgments[cid].values())]
        missing=not a['completed'] or len(evaluated)<len(a['case_ids'])
        details.append({**a,'bad_cases':bad,'unknown':missing,'bad_fields':sorted({f for cid in bad for f,v in judgments[cid].items() if not v['still_fits']})})
    for cohort in ('corrupted','real'):
        result[cohort]={}
        for setting in SETTINGS:
            values=[a for a in details if a['cohort']==cohort and a['setting']==setting]
            n=len(values);bad=sum(bool(a['bad_cases']) for a in values)
            unknown=sum(a['unknown'] and not a['bad_cases'] for a in values)
            observed=[a for a in values if not a['unknown']]
            result[cohort][setting]={'structural_actions':n,'flagged_actions':bad,'unknown_actions':unknown,
                'fully_evaluated_actions':len(observed),
                'share':bad/n if n and not unknown else None,
                'lower_bound':fraction(bad,n),'upper_bound':fraction(bad+unknown,n),
                'complete_case_share':fraction(sum(bool(a['bad_cases']) for a in observed),len(observed)),
                'cases':sum(len(a['case_ids']) for a in values),
                'deleted_targets':sum(len(a['deleted_targets']) for a in values),
                'by_field':{f:sum(f in a['bad_fields'] for a in values) for f in sorted(set(F for a in values for F in a['bad_fields']))}}
    return {'rates':result,'actions':details,'judged_cases':len(judgments),'requested_cases':sum(c['completed'] for c in cases)}


def pct(x):return '—' if x is None else f'{100*x:.1f}%'
def num(x):return '—' if x is None else f'{x:.4f}'


def examples_section(config,design):
    root=config['paths'][PHASE+'_output']
    lines=['## 5. 그래프 예제 5편', '',
        'dev 2편과 dev source 1편은 그래프 품질 점검 대상이며 교사 실행 대상이 아니다. 따라서 해당 세 편의 실제 (c) 행동은 없다. 추가 API 호출 금지에 따라 행동을 만들어 넣지 않았다. 이 세 편의 view는 동일 직렬화 함수의 출력이며 실제 에이전트 실행 기록이 아니다.', '']
    _,_,dev,_=prepare(config)
    # Display selection only: require an available graph, then prefer short inputs.
    # It never changes the evaluation cohort or its failure denominators.
    def available(i):
        path=graph_path(config,i)
        return path.exists() and read_json(path)['status']=='completed'
    viable=[dev[i] for i in design['quality_corrupted_ids'] if available(i) and available(dev[i]['source_id'])]
    chosen=[]
    for allowed in ({'G_SENT_MOVE'},{'G_PARA_SWAP','G_OFFTOPIC'}):
        values=[r for r in viable if any(x['op'] in allowed for x in r['records']) and r['episode_id'] not in chosen]
        if values:chosen.append(min(values,key=lambda r:len(r['corrupted_text']))['episode_id'])
    corrupted_sources={dev[i]['source_id'] for i in chosen}
    sources=[i for i in design['stability_ids'] if available(i) and i not in corrupted_sources]
    length=lambda i:sum(len(u['text']) for p in read_json(graph_path(config,i))['input_layout']['paragraphs'] for u in p['units'])
    def has_action(i):
        path=root/'graph/episodes'/(safe_id(i)+'.json')
        return path.exists() and any(a['valid'] and action_type(a) in {'MOVE','sentence_insert','sentence_delete'}
            for a in read_json(path)['actions_by_role'].get('global',[]))
    real=sorted([i for i in design['real_ids'] if available(i)],key=lambda i:(not has_action(i),length(i)))[:2]
    items=[('corrupted dev',i) for i in chosen]+([('source dev',min(sources,key=length))] if sources else [])+[('real',i) for i in real]
    write_json(root/'example_selection.json',{'items':items,'rule':'available full graphs, shortest; real examples prefer an actual structural action',
        'selection_after_extraction':True,'evaluation_cohorts_unchanged':True})
    lines.extend(['실제 예시는 추출에 성공한 글 중 짧은 글을 골랐고, 실제 글은 구조 행동 기록이 있는 글을 우선했다. 이 예시 선택은 위의 전체 집계 표본을 바꾸지 않는다.',''])
    def text_block(saved,label):
        lines.extend([label,'','```text'])
        for p in saved['input_layout']['paragraphs']:
            lines.append('['+saved['paragraph_ids'][p['pid']]+']')
            for u in p['units']:lines.append(saved['sentence_ids'][u['sid']]+' | '+u['text'])
        lines.extend(['```',''])
    for kind,i in items:
        lines.extend([f'### {kind}: {i}',''])
        saved=read_json(graph_path(config,i))
        if kind=='corrupted dev':
            source=read_json(graph_path(config,dev[i]['source_id']))
            text_block(source,'원문 (표시 ID는 각 입력 순서 기준; 비교는 저장된 stable ID 매핑으로 수행):')
        text_block(saved,'분석 입력:')
        if saved['status']!='completed':
            lines.extend(['그래프 추출 검증 실패: '+json.dumps(saved.get('error'),ensure_ascii=False),''])
            continue
        runtime=root/'graph/graph_history'/(safe_id(i)+'.json')
        hist=read_json(runtime) if runtime.exists() else None
        ep_path=root/'graph/episodes'/(safe_id(i)+'.json')
        ep=read_json(ep_path) if ep_path.exists() else None
        g=hist['views'][0]['graph'] if hist and hist['views'] else saved['graph']
        lines.extend(['담화 edges:','','```text'])
        for key in ('sentence_edges','paragraph_edges'):
            lines.append(key+':')
            lines.extend(f"  {e['source']} {e['label']} {e['target']}" for e in g[key])
            if not g[key]:lines.append('  없음')
        lines.extend(['```','','전체 marker graph (절 구간은 해당 문장의 문자 offset):','','```text'])
        for c in g['clauses']:
            lines.append(f"{c['id']} [{c['start']}:{c['end']}] {c['text']} / boundary={c['boundary']}")
        for e in g['clause_edges']:
            lines.append(f"{e['source']} — {e['form']}:{e['class'] or 'none'} → {e['target']}")
        for m in g['sentences']:
            c=m['conjunction'];cs=f"{c['form']}({c['coarse_class'] or c['visible_relation']}) → {m['predecessor']}" if c else 'none'
            ds=m['predecessor'] if m['subject_omitted'] else 'none'
            lines.append(f"{m['id']}@{m['paragraph']}: CONJ {cs}; omitted→{ds}; register={m['register']}; polarity={m['polarity']}; modality={m['modality'] or 'none'}")
        lines.extend(['```',''])
        for role in ('global','korean'):
            v=next((v for v in hist['views'] if v['role']==role),None) if hist else None
            received=ep and any(c['role']==role for c in ep['calls'])
            label=('실제 수신 observation' if received else '생성된 observation (해당 역할의 완료된 응답 없음)') if v else '동일 serializer의 graph 부분'
            lines.extend([label+' — '+role.upper()+':','','```text',v['view'] if v else render(g,role),'```',''])
        if kind=='corrupted dev':
            lines.append('손상과의 대응 (표시 ID / 원래 stable ID):')
            se=edge_set(source) if source['status']=='completed' else None
            for r in dev[i]['records']:
                if not r['op'].startswith('G_'):continue
                lines.append(f"- {r['op']}, stable IDs {', '.join(r['sids'])}")
                if r['op']=='G_SENT_MOVE':
                    for sid in r['sids']:
                        pub=saved['sentence_ids'].get(sid)
                        outgoing=[e for e in g['sentence_edges'] if e['source']==pub]
                        original_targets={b for a,b,_ in se if a==sid} if se is not None else set()
                        current_targets={b for a,b,_ in edge_set(saved) if a==sid}
                        members={s['id']:s['paragraph'] for s in g['sentences']}
                        mismatches=[e for e in outgoing if members[e['source']]!=members[e['target']]]
                        lines.append(f"  표시 {pub}: {outgoing}; source targets={sorted(original_targets)}")
                        lines.append(f"  원래 target 보존: {bool(original_targets) and current_targets==original_targets}; 문단 간 위치 표시: {mismatches or '없음'}. 이 표시는 원래 위치를 직접 알려주지 않는다.")
                if r['op']=='G_OFFTOPIC':
                    target={saved['sentence_ids'].get(s) for s in r['sids']}
                    linked=[e for e in g['sentence_edges'] if e['source'] in target or e['target'] in target]
                    lines.append('  삽입 노드 edges: '+json.dumps(linked,ensure_ascii=False))
                    lines.append('  고립 노드로 표시됨: '+str(not linked)+'. 고립 여부만으로 무관한 문장임을 확정할 수는 없다.')
                if r['op']=='G_PARA_SWAP':
                    lines.append('  source와 sentence-edge set 일치: '+str(se is not None and edge_set(saved)==se))
                    lines.append('  그래프는 현재 문단 순서와 추출 관계를 보여주지만, 바뀐 두 문단이나 정답 순서를 명시하지 않는다. 위 불일치는 재추출 변동성도 포함한다.')
            lines.extend(['','위의 문단 간 위치 표시는 오류 판정이 아니다. 연결이 없는 노드도 실제로 무관한 문장이라는 보장은 없다.',''])
        act=next((a for a in ep['actions_by_role'].get('global',[]) if a['valid'] and action_type(a) in {'MOVE','sentence_insert','sentence_delete'}),None) if ep else None
        if act:
            notice=next((x for x in hist['changes'] if x['role']=='global' and x['t']==act['t']),{'messages':[]})
            lines.extend(['실제 (c) 구조 행동:','','```json',json.dumps({k:act[k] for k in ('t','thought','action','args')},ensure_ascii=False,indent=2),'```','','graph-change notice:','','```text','\n'.join(notice['messages']) or '없음','```',''])
        else:
            lines.extend(['실제 (c) 구조 행동: 없음 — '+('실행 대상이 아님' if kind!='real' else '구조 행동 전에 종료 또는 추출/실행 실패')+'.',''])
    return lines


def report(config):
    from transformers import AutoTokenizer
    design,train,dev,_=prepare(config)
    root=config['paths'][PHASE+'_output']
    tokenizer=AutoTokenizer.from_pretrained(str(config['paths']['policy_base']),local_files_only=True)
    groups=defaultdict(list)
    for setting,cohort,row,_ in episode_files(config):groups[(cohort,setting)].append(row)
    summary={cohort:{s:stats(groups[(cohort,s)],tokenizer) for s in SETTINGS} for cohort in ('corrupted','real')}
    shared=set.intersection(*[{r['corpus_episode_id'] for r in groups[('corrupted',s)] if r['completed']} for s in SETTINGS])
    common={s:stats([r for r in groups[('corrupted',s)] if r['corpus_episode_id'] in shared],tokenizer) for s in SETTINGS}
    differences={s:paired(groups[('corrupted',s)],groups[('corrupted','current')],
        ['global.R','korean.R','combined.R','combined.R_over'],config) for s in SETTINGS[1:]}
    by_level={s:{l:stats([r for r in groups[('corrupted',s)] if r['level']==l],tokenizer) for l in ('L1','L2','L3','L4')} for s in SETTINGS}
    quality=graph_quality(config,design,dev)
    markers=marker_summary(config)
    api=Phase6API(config,12000,phase=PHASE);budget=api.accounting()
    requests=[read_json(p) for p in (root/'api/requests').glob('*.json')]
    usage=Counter()
    for r in requests:usage.update(r.get('cost',{}))
    validation={'prior_92_unchanged':all(file_sha(p)==h for p,h in design['reused_files'].items()),
        'corpora_unchanged':all(file_sha(p)==h for p,h in design['corpus_files'].items()),
        'no_CHECK':all(not any(r['checks'].values()) for rows in groups.values() for r in rows),
        'budget_ok':budget['confirmed_usd']<=8 and budget['pending']==0,
        'same_marker_prompt':None,
        'teacher_inputs_within_7168':all(len(tokenizer.apply_chat_template(r['messages'],tokenize=True,add_generation_prompt=True))<=7168
            for r in requests if r['stage'].startswith('teacher_'))}
    from .markers import JUDGE_PROMPT
    old=read_json(config['paths']['repo']/'verak/v3/outputs/marker_examples/judge_design.json')
    validation['same_marker_prompt']=JUDGE_PROMPT==old['prompt']
    from .audit import audit
    integrity=audit(config,tokenizer)
    validation['trajectory_integrity']=integrity['passed']
    blocked=Counter()
    for p in (root/'graph/blocked').glob('*.json'):
        blocked['corrupted' if read_json(p)['id'] in train else 'real']+=1
    attempts_complete=all(len(groups[(cohort,s)])+(blocked[cohort] if s=='graph' else 0)==n
                          for cohort,n in (('corrupted',92),('real',30)) for s in SETTINGS)
    status='results_collected' if attempts_complete and budget['pending']==0 and markers['judged_cases']==markers['requested_cases'] else 'partial_or_running'
    run_errors={name:read_json(path).get('errors',[]) for name in ('graph','teacher','spot','marker')
                if (path:=root/(name+'_run_status.json')).exists()}
    if status!='results_collected' and budget['pending']==0 and any(run_errors.values()):
        status=('stopped_at_shared_budget' if any('cap:' in e.get('message','') or 'budget reached' in e.get('message','')
                for errors in run_errors.values() for e in errors) else 'stopped_with_missing_results')
    data={'design':design,'summary':summary,'common_completed_ids':sorted(shared),'common_completed':common,
          'paired_differences':differences,'by_level':by_level,'graph_quality':quality,'markers':markers,
          'budget':budget,'usage':dict(usage),'validation':validation,'integrity':integrity,'status':status,'graph_blocked':dict(blocked),
          'run_errors':run_errors}
    write_json(root/'metrics.json',data)
    lines=['# V3 Observation Test','',f'집계 상태: **{status}**. 미수행/미판정은 아래 분모와 상태에 명시했다.','',
        f"손상 글 완료 수: current {summary['corrupted']['current']['completed']}/92, text_only {summary['corrupted']['text_only']['completed']}/92, graph {summary['corrupted']['graph']['completed']}/92. 세 조건 모두 완료된 {len(shared)}편에서 combined R은 "+', '.join(f"{s} {num(common[s]['roles']['combined']['R'])}" for s in SETTINGS)+'.',
        'Graph의 실행 실패와 완료 표본에서의 보상 비교를 구분해야 한다. 문맥 한도를 넘긴 실행은 글이나 필수 관측을 잘라 이어가지 않았다. 아래 paired 신뢰구간은 실행이 완료된 글만 포함하므로 전체 표본의 우열로 일반화할 수 없다.','',
        '## 1. 실험 범위와 해석','',
        '기존 Pilot 2의 **agent_train 손상 글 92편**에서 관찰 형식을 비교했다. **학습 글을 이용한 설계 비교이며 held-out 평가가 아니다.** 실제 글은 Phase 6의 agent_dev 30편이며, 그래프 품질 점검도 agent_dev만 사용했다. 채점기 train/test 원본은 사용하지 않았다.','',
        '교사: `gpt-6-luna`, reasoning low, two_stage, no CHECK, context 8,192 / 생성 1,024. (a)는 기존 92회 Luna low 기록을 그대로 재사용했다(91편 완료, 1편 IncompleteResponse). 실제 글의 (a)는 새 Luna 실행이다. (b)는 글+문장 ID와 GLOBAL 행동만, (c)는 담화/바른 graph와 graph-change notices를 본다. 행동·보상·주기적 전체 갱신·작업 일지 규칙은 공유한다.','',
        '**55% 알림 발생률은 슬라이드에 사용하지 않는다.** 알림과 의미상 손상을 동일시하지 않는다.','',
        '새 코드: `verak/v3/observation/`, CLI `python -m verak.v3.cli.observation_test STAGE --max-api-calls 12000`. 모든 모델/단계가 단일 SQLite ledger의 $8 한도를 공유한다. 12000은 호출 상한이며 비용 한도가 먼저 적용된다. 신규 bulk generation/SFT/RFT는 실행하지 않았다.','',
        '문장 역할은 claim/support/example/contrast/conclusion/elaboration/other로 고정했다. supports/example_of/concludes outgoing은 합계 최대 하나로 검증했다. 위반 응답은 재요청하거나 edge를 임의로 고치지 않고 실패로 기록했다. 담화 edges는 입력에서 한 번 추출하고, 삭제 시 incoming은 dangling으로 보존하며 UNDO는 노드와 원래 edge를 되살린다. 절은 EC/ETM/ETN/JKQ 표면 경계이며 완전한 내포절 구문 분석은 아니다.','',
        '독립 추출 seed 차이: 설치된 SDK와 [공식 Responses API](https://developers.openai.com/api/reference/python/resources/responses/methods/create)에 seed 인자가 없어, 같은 프롬프트로 서로 다른 두 요청을 보냈다. API sampling seed를 직접 통제한 실험으로 주장하지 않는다. 표본 선정 seed=83이다.','',
        '예비 그래프 표기는 긴 글에서 문맥을 초과해 절 경계를 정보 손실 없이 짧게 표기하도록 바꿨다. 예비 (c) 2건은 `serialization_preflight/`에 분리하고 최종 비교에서 제외했다. 그 호출 비용은 $8에 포함된다. 초기 Bareun 환경변수 전달 누락 4건은 교사 호출 전 실패했고 `pre_api_environment_errors/`에 보관했다.','',
        '빈 문단의 마지막 문장 삭제 후에도 문단 노드는 유지해야 한다. 이 갱신 버그에 영향을 받은 (c) 한 편은 `empty_paragraph_pre_fix/`에 원본을 보존한 뒤 수정본으로 재실행했다. 다른 완료 사례는 그대로 유지했고, 동일한 이전 요청은 캐시로 재생했다. `empty_paragraph_fix.json`에 대상·해시를 저장했으며 이 비용도 공동 예산에 포함된다.','',
        '## 2. 손상 글 비교','',
        '| 조건 | 시도/92 | 완료 | GLOBAL R | KOREAN R | combined R | R_over | 형태소 | 순서 | G steps | K steps |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for s,x in summary['corrupted'].items():
        r=x['roles'];c=r['combined']
        lines.append(f"| {s} | {x['attempted']} | {x['completed']} | {num(r['global']['R'])} | {num(r['korean']['R'])} | {num(c['R'])} | {num(c['R_over'])} | {num(c['morpheme'])} | {num(c['order'])} | {num(r['global']['steps'])} | {num(r['korean']['steps'])} |")
    lines+=['','보상은 완료 사례만, steps는 시도 사례 기준이다. 입력 graph 검증 실패는 교사 실행 전에 중단되므로 위 시도 수에서 빠지며 graph 파일에 별도 보관된다. 누락을 0점으로 채우지 않았다.',
        f"Graph 입력 검증으로 실행하지 못한 글: 손상 {blocked['corrupted']}편, 실제 글 {blocked['real']}편.",'',f'세 조건 모두 완료한 동일 **{len(shared)}편**의 비교:','',
        '| 조건 | GLOBAL R | KOREAN R | combined R | R_over | 형태소 | 순서 |','|---|---:|---:|---:|---:|---:|---:|']
    for s,x in common.items():
        r=x['roles'];c=r['combined'];lines.append(f"| {s} | {num(r['global']['R'])} | {num(r['korean']['R'])} | {num(c['R'])} | {num(c['R_over'])} | {num(c['morpheme'])} | {num(c['order'])} |")
    lines+=['','같은 글에서 current 대비 paired 차이(완료 교집합, level-stratified bootstrap 95% CI):','',
        '| 비교 | 측정값 | n | 평균 차이 | 95% CI |','|---|---|---:|---:|---|']
    for s,values in differences.items():
        for metric,v in values.items():
            interval=' / '.join(num(x) for x in v['ci95']) if v.get('ci95') else '—'
            lines.append(f"| {s} − current | {metric} | {v['n']} | {num(v['difference'])} | {interval} |")
    lines+=['','Bootstrap 단위는 손상 episode이며 같은 source/question의 군집 상관을 보정한 구간이 아니다. 기존 한 번의 current 실행을 재사용한 설계 진단으로 해석한다.']
    lines+=['','연산자별 주 복원율(main; 완료 사례, 괄호는 record 수). coupled 포함 recovery는 metrics.json에 함께 저장했다.','',
        '| 연산자 | current | text_only | graph |','|---|---:|---:|---:|']
    ops=sorted({op for x in summary['corrupted'].values() for op in x['recovery']})
    for op in ops:
        vals=[]
        for s in SETTINGS:
            v=summary['corrupted'][s]['recovery'].get(op);vals.append(f"{num(v['main'])} ({v['n']})" if v else '—')
        lines.append('| '+op+' | '+' | '.join(vals)+' |')
    lines+=['','세 조건 공통 완료 집합에서의 연산자 복원율:','',
        '| 연산자 | records | current | text_only | graph |','|---|---:|---:|---:|---:|']
    for op in sorted(common['current']['recovery']):
        n=common['current']['recovery'][op]['n']
        lines.append('| '+op+f' | {n} | '+' | '.join(num(common[s]['recovery'][op]['main']) for s in SETTINGS)+' |')
    lines+=['','역할별 invalid actions 및 실제 전달 문맥 길이(정책 tokenizer):','',
        '| 자료/조건/역할 | invalid/행동 | 입력 median / p90 / max | 입력+답변 median / p90 / max | 압축 턴 |','|---|---:|---|---|---:|']
    for cohort,ss in summary.items():
        for s,x in ss.items():
            for role in ('global','korean'):
                r=x['roles'][role];ins=r['input_tokens'];tot=r['total_tokens']
                fmt=lambda d:' / '.join(f"{d[k]:.1f}" if d.get(k) is not None else '—' for k in ('median','p90','max'))
                lines.append(f"| {cohort}/{s}/{role} | {r['invalid']}/{r['actions']} | {fmt(ins)} | {fmt(tot)} | {r['truncations']} |")
    lines+=['','실제 글은 reward를 계산하지 않았다. 완료율과 실행량:','',
        '| 조건 | 시도/30 | 완료/30 | GLOBAL steps | KOREAN steps |','|---|---:|---:|---:|---:|']
    for s,x in summary['real'].items():
        lines.append(f"| {s} | {x['attempted']} | {x['completed']} | {num(x['roles']['global']['steps'])} | {num(x['roles']['korean']['steps'])} |")
    lines+=['','종료 사유, 역할별 R_over 두 항, L1–L4별 보상, paired difference 및 bootstrap 95% CI는 `metrics.json`에 저장했다.','',
        '## 3. 최종 글의 표지 부적합 (LLM-verified)','',
        '모든 유효한 MOVE/문장 삽입/삭제를 분모로 삼았다. 앞 문장이 달라진 문장·삽입 문장과 내부 Bareun 알림의 대상 문장을 합쳐, **KOREAN 종료 후** 실제 앞 문장과 대상 문장만 Sol high에 제공했다. 세 조건에서 같은 추출 규칙과 이전 marker report의 동일 prompt를 사용했다. 인접 위치 변경 대상에는 conjunction/subject_omission/ending_style을 함께 물었고, 추가 내부 표지 알림이 있으면 해당 field도 검증했다. 대상이 최종 삭제되면 남은 표지 문제는 없으며, 같은 문장에 여러 행동이 관련되면 각 행동에 귀속한다.','',
        '이는 최종 문맥의 잔여 부적합이지 구조 행동이 새 오류를 만들었다는 인과 추정이 아니다. 초기 손상의 잔존도 포함할 수 있다. 두 문장만으로 확정할 수 없으면 true를 반환하는 보수적 검출 규칙이다. 인간 정확도로 해석하지 않는다.','',
        '| 자료 | 조건 | 구조 행동 | 최종 부적합 행동 | 미판정 행동 | 비율 / 가능한 범위 |','|---|---|---:|---:|---:|---|']
    for cohort,ss in markers['rates'].items():
        for s,v in ss.items():
            rate=pct(v['share']) if v['share'] is not None else pct(v['lower_bound'])+'–'+pct(v['upper_bound'])
            lines.append(f"| {cohort} | {s} | {v['structural_actions']} | {v['flagged_actions']} | {v['unknown_actions']} | {rate} |")
    lines+=['',f"평가 완료 {markers['judged_cases']}/{markers['requested_cases']} final sentence cases. 실행 미완료/미판정은 통과로 간주하지 않는다.",'',
        '## 4. agent_dev 그래프 품질','',
        '품질 점검은 원시 추출 edges를 사용한다. outgoing 개수 제한을 위반한 응답도 해당 edge의 의미 정확성·안정성은 검증하되 정책 실행에는 사용하지 않았다. 원시 edges를 임의 삭제·수정하거나 재추출하지 않았다. 따라서 이 절의 평가 가능 수와 (c) 교사 실행 가능 수는 다르다.',
        '전체 그래프 추출 검증 상태: '+json.dumps(quality['raw_extraction_status'],ensure_ascii=False)+'.', '',
        '| 손상 검사 | 전체 records | 평가 가능 | 통과 | 비율 |','|---|---:|---:|---:|---:|']
    for op,v in quality['corruption_checks'].items():
        lines.append(f"| {op} | {v['cases']} | {v['evaluable']} | {v['passes']} | {pct(v['rate'])} |")
    m=quality['corruption_checks']['G_SENT_MOVE'];st=quality['stability'];sp=quality['spot']
    lines+=['',f"G_SENT_MOVE source에 outgoing target이 있는 {m['source_has_target']}건에서 target 보존율 {pct(m['retention_given_source_target'])}. 양쪽 모두 target 없음 {m['both_empty']}건은 복원 증거가 없어 위 통과에 포함하지 않았다. 빈 집합도 일치로 세면 {pct(m['exact_including_empty'])}.",'',
        '다른 local 연산자가 함께 적용된 복합 손상도 포함한다. source와 corrupted의 공개 S/P 번호는 각각 입력 순서에서 부여하며, 비교 때 원래 stable ID로 역매핑한다. graph 추출에 원문·손상 record를 제공하지 않는다.','',
        f"Stability: 요청 100편, 두 원시 응답 비교 가능 {st['evaluable']}편; 두 응답 모두 정책용 검증 통과 {st['policy_valid_pairs']}편. 같은 source/target/label edge를 일치로 센다.",'',
        '| edge 종류 | micro F1 | micro Jaccard | 글별 exact-set 일치 |','|---|---:|---:|---:|']
    for k in ('sentence_edges','paragraph_edges'):
        v=st[k];lines.append(f"| {k} | {pct(v['micro_f1'])} | {pct(v['micro_jaccard'])} | {pct(v['exact_set_rate'])} |")
    lines+=['',f"동일 원문을 반복 추출한 sentence-edge exact-set 일치율도 {pct(st['sentence_edges']['exact_set_rate'])}다. 따라서 G_PARA_SWAP의 exact-set 검사만으로 순서 변경의 영향이나 그래프의 의미 정확성을 분리해 판단할 수 없다. 추출 자체의 변동성과 복합 손상을 함께 고려해야 한다.",'',
        f"Node-role agreement: {pct(st['node_role_agreement'])}, {st['role_nodes']} sentence nodes.",'',
        f"Sol high spot check (LLM-verified): 요청 50 source essays, 상태 {sp['essay_status']}. 전체 correct {sp['overall']['correct']}/{sp['overall']['n']} = {pct(sp['overall']['correct_share'])}; incorrect {sp['overall']['incorrect']}, unclear {sp['overall']['unclear']}.",'',
        '| edge type | correct | incorrect | unclear | correct share |','|---|---:|---:|---:|---:|']
    for k,v in sp['by_type'].items():lines.append(f"| {k} | {v['correct']} | {v['incorrect']} | {v['unclear']} | {pct(v['correct_share'])} |")
    lines+=['']+examples_section(config,design)
    lines+=['## 6. 비용·검증·제한','',f"신규 확정 비용 **${budget['confirmed_usd']:.6f} / $8**, 잔여 예약 ${budget['reserved_usd']:.6f}; API 시도 {budget['calls']}회.",
        f"Input {usage['input']:,}, output {usage['output']:,} (reasoning {usage['reasoning']:,} 포함), cache read {usage['cache_read']:,}, cache write {usage['cache_write']:,}. Reasoning 비용을 output에 이중 합산하지 않았다.",'',
        f"과거 (a) 92편 비용 ${summary['corrupted']['current']['cost_usd']:.8f}는 이번 신규 예산에 포함하지 않는다. 가격은 기존 ledger의 Luna input/output $0.10/$0.50 per M, Sol $2/$10 per M 및 cache 요율을 유지했다.",'',
        '단계별 비용:','','```json',json.dumps(budget['by_stage'],ensure_ascii=False,indent=2),'```','',
        '자동 검증:','','```json',json.dumps(validation,ensure_ascii=False,indent=2),'```','']
    if any(run_errors.values()):
        lines+=['단계 중단 사유:','','```json',json.dumps({s:[{'type':e.get('type'),'message':e.get('message')}
            for e in errors] for s,errors in run_errors.items() if errors},ensure_ascii=False,indent=2),'```','']
    test_path=root/'tests.txt'
    if test_path.exists():lines+=['테스트:','','```text','\n'.join(test_path.read_text().splitlines()[-8:]),'```','']
    lines+=['실행 오류(누락을 숨기지 않음):','','```json',json.dumps({c:{s:x['errors'] for s,x in ss.items()} for c,ss in summary.items()},ensure_ascii=False,indent=2),'```','',
        '데이터/출력 경로: `verak/v3/outputs/observation_test/`. `design.json`, `graphs/`, 조건별 `episodes/`, `events/`, `graph/graph_history/`, `spot/`, `marker_cases.jsonl`, `marker_judgments/`, `metrics.json`, `api/ledger.sqlite`에 원시 결과와 집계를 보관했다.','',
        '이번 비교는 중단 시점까지 확보한 결과만 보고한다. 표본 수/실패율과 paired 완료 집합을 함께 봐야 하며, 학습 표본 설계 비교에서 일반화 성능을 주장하지 않는다.']
    target=config['paths']['repo']/'imple/reports/V3_OBSERVATION_TEST.md'
    target.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(target),'cost':budget['confirmed_usd'],'validation':validation},ensure_ascii=False))
