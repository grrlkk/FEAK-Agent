"""Frozen-output comparison: operations, independent quality, and data hygiene."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from pathlib import Path
import random
import statistics
import threading
import time

from .cooperate import ROOT, REPORT, RUNS, output_dir, verify_design
from .common import REPO, read_json, ref, verified, sha_text, safe_id
from .check_judge import freeze
from .check_outputs import flat, positions
from .judge import JUDGE_PROMPT, paragraphs, validate
from .api import PaidAPI, BudgetStop
from .repair1_eval import record_metrics, transitions
from verak.v4.content_env import dumps
from verak.v4.fix1_infer import summarize
from verak.v4.fix1_taxonomy import rejection_rows, summary as rejection_summary
from verak.v4.scale2_projection import restore_document

CONDITIONS = tuple(['old_run'+str(i) for i in range(1,4)]+
                   ['new_run'+str(i) for i in range(1,4)]+['onepass','luna'])


def results(case):
    values = {label: verified(r) for label,r in case['C_outputs'].items()}
    paths = {label: r for label,r in case['C_outputs'].items()}
    for setting in RUNS:
        run = str(setting['run'])
        path = output_dir(setting,case)/'attempts'/(safe_id(case['case_id'])+'.json')
        values['new_run'+run] = read_json(path); paths['new_run'+run] = ref(path)
        values['old_run'+run] = verified(case['baselines'][run]); paths['old_run'+run] = case['baselines'][run]
    return values, paths


def documents(values):
    return {'original':values['onepass']['initial'], **{
        label:value['final_token_document'] if label.startswith(('new','old')) else value['final']
        for label,value in values.items()}}


def selected_sites(case_id, docs, alignment_failed):
    """A bounded public-text diagnostic, independent of gold or judge decisions."""
    original = paragraphs(docs['original']); old = {s['id']:s for s in flat(original)}
    old_pos = positions(original); candidates = set()
    for name, doc in docs.items():
        if name == 'original' or name == 'onepass' and alignment_failed:
            continue
        view = paragraphs(doc); now = {s['id']:s for s in flat(view)}; pos = positions(view)
        changed = {sid for sid in old if now.get(sid) != old[sid] or pos.get(sid) != old_pos[sid]}
        candidates |= changed
        ordered = list(old)
        candidates |= {ordered[i+1] for i,sid in enumerate(ordered[:-1]) if sid in changed}
    ordered = sorted(candidates)
    rng = random.Random(int(sha_text(case_id+':cooperate1-link-sites')[:16],16))
    return sorted(rng.sample(ordered, min(6,len(ordered))))


def judging_manifest():
    marker = ROOT/'judge_manifest.json'
    if marker.exists(): return read_json(marker)
    if not (ROOT/'inference_complete.json').exists():
        raise RuntimeError('Finish all three inference runs before freezing quality requests')
    gate_path = REPO/'verak/v5/outputs/check/validation/final.json'
    gate = read_json(gate_path)
    if not gate['continue'] or gate['validated_setting'] != 'full':
        raise RuntimeError('This comparison requires the validated full-context setting')
    cases = []; requests = {}
    for case in verify_design()['cases']:
        values, paths = results(case); docs = documents(values)
        failed = values['onepass'].get('alignment',{}).get('failed',False)
        sites = selected_sites(case['case_id'],docs,failed)
        orig = paragraphs(docs['original']); old = positions(orig)
        row = verified(case['row']); aliases = {}; maps = {}
        for condition,doc in docs.items():
            view = paragraphs(doc); pos = positions(view)
            measurable = [s for s in sites if s in pos] if not (condition=='onepass' and failed) else []
            notices = [{'id':f'N{i+1}', 'sid':sid, 'type':'candidate_link_check',
                'before':old.get(sid), 'after':pos[sid],
                'scope':[s for s in (sid,pos[sid]['predecessor']) if s]}
                for i,sid in enumerate(measurable)]
            payload = {'question':row['question'], 'original':orig, 'revised':view,
                       'notices':notices, 'variants':[{'id':'V1','paragraphs':view}]}
            messages = [{'role':'system','content':JUDGE_PROMPT},{'role':'user','content':dumps(payload)}]
            digest = sha_text(dumps(messages)); aliases[condition] = digest
            maps[condition] = {n['id']:n['sid'] for n in notices}
            requests[digest] = {'id':digest, 'messages':messages, 'notice_ids':[n['id'] for n in notices],
                'variant_ids':['V1'], 'max_output':4096}
        cases.append({'case_id':case['case_id'],'cohort':case['cohort'],'aliases':aliases,
            'sid_maps':maps,'selected_original_sites':sites,'onepass_alignment_failed':failed,'provenance':paths})
    order = list(requests)
    random.Random(713031).shuffle(order)
    value = {'cases':cases,'requests':[requests[h] for h in order], 'context':'full',
        'prompt_sha256':sha_text(JUDGE_PROMPT),'validated_gate':ref(gate_path),
        'evaluation_code':ref(Path(__file__)), 'api_contract':ref(ROOT/'api_contract.json'),
        'condition_names_in_API':False,'gold_or_scores_in_API':False,
        'same_saved_text_and_sites_share_judgment':True,
        'global_quality':'Each final output is revised and its identical V1. Exact C judge; all failure outputs included.',
        'link_scope':'Up to 6 seeded original sentence sites per essay from union of changed text/position and successors across arms. '
                     'Newly inserted sentences are outside this bounded link sample; whole-essay quality still sees them. '
                     'Absent sites and failed onepass alignment are unknown, not fine.',
        'judged_control':'Original input also judged for paired broken/fixed/new-link transitions; no private pristine source.',
        'validation_limit':'V5_CHECK calibrates link classification, not global writing preference or fabrication detection.',
        'API_cap_usd':5,'no_new_Luna_calls':True,'test_cohort_untouched':True}
    freeze(marker,value)
    return value


def judge():
    if (ROOT/'judge_complete.json').exists(): return
    start = time.monotonic(); manifest = judging_manifest(); stop = threading.Event()
    def one(req):
        path = ROOT/'judgments'/(req['id']+'.json')
        if path.exists(): return 'saved'
        if stop.is_set(): return 'budget_not_dispatched'
        api = PaidAPI('claude',root=ROOT)
        try:
            value = {'valid':False,'attempts':[],'prompt_sha256':manifest['prompt_sha256']}
            for number in (1,2):
                trial_path = ROOT/'judgments'/f'{req["id"]}_try{number}.json'
                if trial_path.exists(): trial = read_json(trial_path)
                else:
                    trial = {'valid':False}
                    try:
                        response = api.request(req['messages'],item_id=f'COOPERATE1:{req["id"]}:{number}',
                                               max_output=req['max_output'])
                        trial['response'] = response
                        trial['verdict'] = validate(response['raw'],req['notice_ids'],req['variant_ids'])
                        trial['valid'] = True
                    except BudgetStop:
                        stop.set(); return 'budget_not_dispatched'
                    except Exception as exc:
                        trial['error'] = type(exc).__name__+': '+str(exc)
                    freeze(trial_path,trial)
                value['attempts'].append(ref(trial_path))
                if trial['valid']:
                    value.update(valid=True,verdict=trial['verdict']); break
                if trial.get('response',{}).get('status') != 'completed': break
            freeze(path,value)
            return 'valid' if value['valid'] else 'invalid'
        finally: api.close()
    with ThreadPoolExecutor(max_workers=3) as pool:
        counts = Counter(pool.map(one,manifest['requests']))
    api = PaidAPI('claude',root=ROOT); api.close()
    freeze(ROOT/'judge_complete.json',{'counts':dict(counts),'budget_stop':stop.is_set(),
        'wall_seconds':time.monotonic()-start,'model':api.model,'unique_requests':len(manifest['requests'])})
    print('cooperate Claude complete',dict(counts),flush=True)


def verdict(jcase,condition):
    path = ROOT/'judgments'/(jcase['aliases'][condition]+'.json')
    value = read_json(path) if path.exists() else {}
    raw = value.get('verdict',{}) if value.get('valid') else {}
    labels = {n['id']:n['broken_after_revision'] for n in raw.get('notices',[])}
    sites = {sid:labels.get(nid,'unjudged') for nid,sid in jcase['sid_maps'][condition].items()}
    quality = raw.get('variants',[{}])[0]
    return quality, {sid:sites.get(sid,'unavailable') for sid in jcase['selected_original_sites']}


def spread(values):
    return {'mean':statistics.mean(values),'min':min(values),'max':max(values)} if values else None


def text_changes(before, after):
    a={u.sid:u.text for u in restore_document(before).units}
    b={u.sid:u.text for u in restore_document(after).units}
    return {'source_sentences':len(a),'changed_source_sentences':sum(b.get(s)!=t for s,t in a.items()),
        'deleted_source_sentences':len(a.keys()-b.keys()),'inserted_sentence_ids':len(b.keys()-a.keys()),
        'edited_surviving_sids':[s for s in a.keys()&b.keys() if a[s]!=b[s]]}


def measure():
    manifest = judging_manifest(); jcases={c['case_id']:c for c in manifest['cases']}
    rows=[]; operations={}; rejections=[]; attempts=defaultdict(list)
    for case in read_json(ROOT/'design.json')['cases']:
        values,paths=results(case); docs=documents(values); jc=jcases[case['case_id']]
        before_q=read_json(output_dir(RUNS[0],case)/'quality_scores'/(safe_id(case['case_id'])+'.json'))['original']['mean']
        previous_scores=read_json(ROOT/'comparison_scores'/(safe_id(case['case_id'])+'.json'))
        original_quality, original_labels=verdict(jc,'original')
        for condition in CONDITIONS:
            value=values[condition]; quality,labels=verdict(jc,condition)
            if condition.startswith('new'):
                setting=next(s for s in RUNS if str(s['run'])==condition[-1])
                q=read_json(output_dir(setting,case)/'quality_scores'/(safe_id(case['case_id'])+'.json'))['final']['mean']
            else: q=previous_scores[condition]['mean']
            changes=text_changes(docs['original'],docs[condition])
            if condition.startswith(('new','old')):
                protocol=value['completion']['no_protocol_error'];done=value['completion']['all_stages_STOP_done']
                attempts[case['cohort']+'/'+condition].append(value)
                rejections.extend(dict(condition=condition,cohort=case['cohort'],**r)
                    for r in rejection_rows(value,paths[condition]['path'],near=False))
            else: protocol=value['no_protocol_error'];done=None
            records=None
            if not (condition=='onepass' and jc['onepass_alignment_failed']):
                records=record_metrics(case,{'initial':docs['original'],'final':docs[condition],
                    'lineage':value.get('final',{}).get('lineage',value.get('lineage',{}))})
            rows.append({'case_id':case['case_id'],'cohort':case['cohort'],'condition':condition,
                'protocol':protocol,'all_done':done,'Q_original':before_q,'Q_final':q,'Q_delta':q-before_q,
                'quality':quality,'original_control_quality':original_quality,'link_labels':labels,
                'paired_links':transitions(original_labels,labels,set(changes['edited_surviving_sids'])),
                'text_changes':changes,'record_metrics':records,
                'onepass_alignment_failed':condition=='onepass' and jc['onepass_alignment_failed'],
                'artifact':paths[condition]})
    def operation_summary(values):
        out=summarize(values,len(values))
        out['valid_action_counts']={role:dict(Counter(c['action']['action'] for a in values
            for c in a['calls'] if c['role']==role and c['action']['valid']))
            for role in ('orchestrator','revision','korean')}
        return out
    for name,values in attempts.items(): operations[name]=operation_summary(values)
    for condition in CONDITIONS[:6]:
        values=[v for name,items in attempts.items() if name.endswith('/'+condition) for v in items]
        operations['all/'+condition]=operation_summary(values)
    aggregates={}
    for cohort in ('all','smoke20','agent_dev10'):
        for condition in CONDITIONS:
            group=[r for r in rows if r['condition']==condition and (cohort=='all' or r['cohort']==cohort)]
            a={'n':len(group),'protocol':sum(r['protocol'] for r in group),
               'mean_Q_delta':statistics.mean(r['Q_delta'] for r in group),
               'global_quality':{f:dict(Counter(r['quality'].get(f,'unjudged') for r in group))
                  for f in ('better_than_original','meaning_preserved','fabrication')},
               'safe_improvement':sum(r['quality'].get('better_than_original')=='yes' and
                   r['quality'].get('meaning_preserved')=='yes' and r['quality'].get('fabrication')=='no' for r in group),
               'link_labels':dict(Counter(v for r in group for v in r['link_labels'].values())),
               'paired_links':dict(sum((Counter(r['paired_links']) for r in group),Counter())),
               'alignment_failures':sum(r['onepass_alignment_failed'] for r in group)}
            aggregates[cohort+'/'+condition]=a
    repeat={}
    for cohort in ('all','smoke20','agent_dev10'):
        for family in ('old','new'):
            names=[cohort+'/'+family+'_run'+str(i) for i in range(1,4)]
            repeat[cohort+'/'+family]={f:spread([operations[n][f] for n in names])
                for f in ('no_protocol_error','all_stages_STOP_done','mean_model_delegations')}
            repeat[cohort+'/'+family]['mean_Q_delta']=spread([aggregates[n]['mean_Q_delta'] for n in names])
    result={'rows':rows,'operations':operations,'quality':aggregates,'repeat_summary':repeat,
        'rejection_taxonomy':{c:rejection_summary([r for r in rejections if r['condition']==c]) for c in CONDITIONS[:6]}}
    freeze(ROOT/'measurement_rows.json',rows);freeze(ROOT/'rejections.json',rejections)
    freeze(ROOT/'metrics.json',result)
    return result


def report():
    if not (ROOT/'inference_complete.json').exists() or not (ROOT/'judge_complete.json').exists():
        raise RuntimeError('Inference and budget-bounded judging must finish first')
    start=time.monotonic(); m=measure(); design=read_json(ROOT/'design.json')
    lines=['# V5_COOPERATE1 — 세 에이전트 협업 안정화','',
        '고정 모델·프롬프트로 실행 제어와 인계를 수정한 진단 실험이다. 학습 없음; test cohort 미사용. '
        'smoke20 + agent_dev10에서 저장된 FIX1 old 3회와 새 통합 실행 3회를 비교했다.','',
        '## 고정 계약과 수정','',
        '- Orchestrator epoch2 / Revision epoch1 / Korean epoch2, temperature 0, seeds 47/73/101. '
        '3회는 같은 30편의 반복이며 90개의 독립 표본이 아니다. 시스템 프롬프트·가중치·행동/문맥 예산은 유지했다.',
        '- 실제 작업자 보고(변경 요약·미해결 사유)와 현재 상태를 다음 작업자·총괄에게 전달한다. '
        '자동 마지막 Korean pass 후에는 총괄이 그 결과를 보고 FINISH한다.',
        '- Korean 전체 글 범위와 단계 단위 STOP을 유지한다. 같은 상태의 거절 행동에는 구체적인 재시도 정보를 준다. '
        '문장 전체 금지나 강제 건너뛰기 성공 처리는 하지 않는다. 원래 연속 3거절 실패 규칙은 유지한다.',
        '- a4 규칙 위임, 별도 gate/queue, 새 교사 호출 없음. 총괄은 공적 입력만으로 계획한다.',
        '- 묶음 수정 비교이므로 인계 한 요소의 인과 효과로 해석하지 않는다. 관찰 형식 변화는 기존 SFT와 분포 차이가 있다.',
        '- 완료(a)는 강제 종료·step/context/형식 실패가 없는 실행이다. 이유가 있는 모델 STOP(blocked)는 (a)에 포함되지만 '
        '모든 단계 STOP(done)인 완료(b)에는 포함하지 않는다. STOP(done) 자체가 수정 품질의 증거는 아니다.','',
        '## 실행 결과','',
        '| Cohort | Condition | 완료(a) | 완료(b) | 위임/편 |',
        '|---|---|---:|---:|---:|']
    for name,a in m['operations'].items():
        cohort,c=name.split('/');lines.append(f'| {cohort} | {c} | {a["no_protocol_error"]}/{a["planned"]} | '
            f'{a["all_stages_STOP_done"]}/{a["planned"]} | {a["mean_model_delegations"]:.2f} |')
    lines+=['','| Cohort/family | 완료(a) 평균 [최소,최대] | 완료(b) 평균 [최소,최대] |','|---|---|---|']
    for name,a in m['repeat_summary'].items():
        def fmt(k):
            x=a[k];return f'{x["mean"]:.2f} [{x["min"]}, {x["max"]}]'
        lines.append(f'| {name} | {fmt("no_protocol_error")} | {fmt("all_stages_STOP_done")} |')
    lines+=['','| Condition | Role | Valid/actions | STOP done/blocked | Forced stop | Context failures |',
            '|---|---|---:|---:|---:|---:|']
    for c in CONDITIONS[:6]:
        for role,a in m['operations']['all/'+c]['roles'].items():
            lines.append(f'| {c} | {role} | {a["valid_actions"]}/{a["returned_actions"]} | '
                f'{a["model_STOP_done"]}/{a["model_STOP_blocked"]} | {a["controller_STOP_blocked"]} | {len(a["context_limits"])} |')
    lines+=['','총괄은 STOP 대신 FINISH를 쓴다. 총괄 종료 유형, 개별 실패·거절 후 행동은 metrics.json/rejections.json에 보존했다.','',
        '| Condition | Role | 거절 유형별 수 |','|---|---|---|']
    for c,roles in m['rejection_taxonomy'].items():
        for r,types in roles.items():lines.append(f'| {c} | {r} | '+str({k:v['count'] for k,v in types.items()})+' |')
    lines+=['','| Condition | Role | Accepted actions |','|---|---|---|']
    for c in CONDITIONS[:6]:
        for role,counts in m['operations']['all/'+c]['valid_action_counts'].items():
            lines.append(f'| {c} | {role} | {counts} |')
    lines+=['','## 품질 — 실행 성공과 별도','',
        '검증된 full-context C Claude 프롬프트를 그대로 사용했다. 모델·조건명·정답 레코드·채점 점수는 감췄다. '
        '원문 및 최종 글 전체를 제공하고 동일 텍스트/위치 판정은 공유했다. 실패 실행도 포함한다. '
        'Claude 링크 분류 검증이 전체 글 선호·사실성 판단까지 검증한 것은 아니다. LLM 평가이며 사람 평가가 아니다.','',
        '| Cohort | Condition | ΔQ (GPU) | Better yes/no/uncertain/unjudged | Meaning yes | Fabrication yes | Safe improvement |',
        '|---|---|---:|---|---:|---:|---:|']
    for name,a in m['quality'].items():
        cohort,c=name.split('/');g=a['global_quality']; b=g['better_than_original']
        counts='/'.join(str(b.get(k,0)) for k in ('yes','no','uncertain','unjudged'))
        lines.append(f'| {cohort} | {c} | {a["mean_Q_delta"]:+.4f} | {counts} | '
            f'{g["meaning_preserved"].get("yes",0)}/{a["n"]} | {g["fabrication"].get("yes",0)}/{a["n"]} | '
            f'{a["safe_improvement"]}/{a["n"]} |')
    lines+=['','Safe improvement = better=yes, meaning=yes, fabrication=no. 불확실/미판정은 성공으로 세지 않았다. '
        'ΔQ는 동일 GPU reference scorer만 사용하며 비교 기준은 동일한 입력 글이다. '
        'agent_dev10의 기준 글은 훼손 입력이며 비공개 pristine 원문이 아니다.','',
        '| Condition (all30) | 판정 가능 paired sites | Broken before | Fixed | New broken | Broken after | Uncertain/missing pairs |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        a=m['quality']['all/'+c]['paired_links']
        lines.append('| '+c+' | '+' | '.join(str(a.get(k,0)) for k in ('measurable_pairs','broken_before','fixed','new_broken','broken_after','uncertain_or_unjudged_pairs'))+' |')
    lines+=['','링크 표는 모든 조건의 변경 문장/다음 문장 합집합에서 편당 최대 6개 원래 문장 위치를 seed로 뽑은 진단이다. '
        '새 INSERT 문장 자체는 이 위치 표의 대상이 아니며 전체 글 판정에는 포함된다. 삭제·정렬 실패는 unknown이다. '
        '각 행은 원문/해당 결과가 모두 확정 yes/no인 쌍이며 서로 분모가 다를 수 있다. '
        '같은 글의 3회 반복과 공유 판정으로 불확실성의 독립 표본 수가 늘어나지는 않는다.','',
        '## agent_dev10 기록 복원 (보조 진단)','',
        '역사적 진단 입력·레코드는 바꾸지 않았다. L_CONN 정리본은 이후 학습용으로 별도다. '
        '학습에 가까운 훼손 복원 지표를 일반 작문 품질의 대용으로 쓰지 않는다.','',
        '| Condition | Operator | Full / measurable | Mean main recovery |','|---|---|---:|---:|']
    for c in CONDITIONS:
        ops=defaultdict(list)
        for r in m['rows']:
            if r['condition']==c and r['record_metrics'] and r['record_metrics']['final']['status']=='exact':
                for rec in r['record_metrics']['final']['records']:ops[rec['op']].append(rec['main'])
        for op,values in sorted(ops.items()):
            lines.append(f'| {c} | {op} | {sum(x==1 for x in values)}/{len(values)} | {statistics.mean(values):.4f} |')
    lines+=['','| Condition | Exact record projections / 10 | Mean R_over (exact) | Aligned essays / 30 | Changed source sentences | Deleted source sentences |',
            '|---|---:|---:|---:|---:|---:|']
    for c in CONDITIONS:
        group=[r for r in m['rows'] if r['condition']==c]
        exact=[r['record_metrics']['final'] for r in group if r['record_metrics'] and r['record_metrics']['final']['status']=='exact']
        aligned=[r['text_changes'] for r in group if not r['onepass_alignment_failed']]
        over=statistics.mean(v['R_over']['value'] for v in exact) if exact else None
        source_n=sum(v['source_sentences'] for v in aligned)
        lines.append(f'| {c} | {len(exact)}/10 | {over} | {len(aligned)}/30 | '
            f'{sum(v["changed_source_sentences"] for v in aligned)}/{source_n} | '
            f'{sum(v["deleted_source_sentences"] for v in aligned)} |')
    lines+=['','문장 변경/삭제는 ID 정렬 가능 글만 집계한다. one-pass 정렬 실패 글의 전체 글 판정과 ΔQ는 그대로 포함한다. '
              '복원·R_over는 정확히 투영된 결과만이며 미투영은 0으로 채우지 않는다.']
    cleanup=read_json(ROOT/'lconn_cleanup/complete.json')
    lines+=['','## Stage 3 이전 L_CONN 정리','',
        '저장 텍스트를 Codex가 점검했다. 사람 gold가 아니며 불확실한 형태는 유지했다. '
        '명확한 비문만 해당 문장 원문을 복원하고 기록을 뺐다. 다른 기록과 겹치는 문장은 없었다.','',
        '| Split | L_CONN before | Restored/dropped | Reclassified L_CONJ | Uncertain retained | L_CONN after | Zero-record tasks retained |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for split,a in cleanup['corpora'].items():
        lines.append(f'| {split} | {a["operators_before"]["L_CONN"]} | {a["dropped_records"]} | {a["relabeled_records"]} | '
            f'{a["uncertain_retained"]} | {a["operators_after"]["L_CONN"]} | {a["zero_record_tasks"]} |')
    lines+=['','| Role/partition | Original turns | Excluded turns | Kept turns |','|---|---:|---:|---:|']
    for name,a in cleanup['SFT_exports'].items():
        lines.append(f'| {name} | {a["before_turns"]} | {a["excluded_turns"]} | {a["kept_turns"]} |')
    lines+=['','제외는 훼손 episode 단위로 네 역할/대상을 모두 포함한다. 다른 연습이나 원문 기반 B2/MAP은 유지했다. '
        '남은 SFT 행은 원래 토큰·loss mask·5% source split 그대로 복사했다. 기존 어댑터/자료는 바꾸지 않았다.',
        '정리된 corpus는 재준비 입력이다. 오래된 profile/score/reverse snapshot을 제거하고 reward_ready=false로 표시했다. '
        '다음 수집 전 profile 및 GPU 기준 점수 재계산이 필요하다. 문두 아니면 1건은 L_CONJ로 분류했으나 '
        'v1 폐쇄집합 밖이라 재준비 시 명시적으로 다뤄야 한다. v1 보상·학습 코드는 변경하지 않았다.','',
        '## 시간·비용·산출물','']
    judge_done=read_json(ROOT/'judge_complete.json'); cost=read_json(ROOT/'api/accounting.json')
    lines += [f'- 새 GPU 실행/비교 채점: {read_json(ROOT/"inference_complete.json")["wall_seconds"]:.1f} s.',
        f'- Claude: {judge_done["wall_seconds"]:.1f} s; model `{judge_done["model"]}`; '
        f'confirmed ${cost["confirmed_usd"]:.6f}, reserved ${cost["reserved_usd"]:.6f}, cap $5; budget stop={judge_done["budget_stop"]}.',
        f'- L_CONN export CPU: {cleanup["wall_seconds"]:.1f} s (manual inspection time not included).',
        '- 새 학습·Luna 호출·test cohort 사용 없음.',
        f'- 설계/모델/코드/프롬프트 hashes: `{ROOT}/design.json`.',
        f'- 블라인드 judge 입력 및 입력별 provenance: `{ROOT}/judge_manifest.json`.',
        f'- 모든 실행·품질·실패 표: `{ROOT}/metrics.json`; 원시 거절: `rejections.json`.',
        f'- 정리 원문/정확한 제외 목록/필터된 SFT: `{ROOT}/lconn_cleanup/`.', '']
    REPORT.write_text('\n'.join(lines))
    freeze(ROOT/'report_complete.json',{'report':ref(REPORT),'wall_seconds':time.monotonic()-start})
    return m
