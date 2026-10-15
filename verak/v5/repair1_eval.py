"""Blinded, bounded evaluation of the small repair-controller experiment."""
from collections import Counter, defaultdict
from copy import deepcopy
from difflib import SequenceMatcher
import json
import random
import time

from .repair1 import ROOT, REPORT, CONDITIONS, CHECK_ROOT, PIPE_ROOT
from .common import read_json, ref, verified, safe_id, sha_text, atomic_new, write_json, REPO
from .check_judge import freeze
from .judge import JUDGE_PROMPT, paragraphs, validate
from .detector import phi
from .api import PaidAPI, BudgetStop
from verak.v4.content_env import dumps
from verak.v4.scale2_projection import restore_document, project_document, private_document, ProjectionUnknown


def case_dir(case):
    return ROOT/'episodes'/case['cohort']/safe_id(case['case_id'])


def changed_sids(before, after):
    a = {u.sid:u.text for u in restore_document(before).units}
    b = {u.sid:u.text for u in restore_document(after).units}
    return {sid for sid,text in b.items() if a.get(sid) != text}


def judge_sites(case, episodes, candidates):
    before = episodes['baseline']['initial']
    ids = [u.sid for u in restore_document(before).units]
    flagged = sorted({c['sid'] for c in candidates})
    rng = random.Random(int(sha_text(case['case_id']+':repair1-sites')[:16],16))
    selected = set(rng.sample(flagged, min(6,len(flagged))))
    edited = set().union(*(changed_sids(before, e['final']) for e in episodes.values()))
    selected |= edited
    selected |= {ids[i+1] for i,sid in enumerate(ids[:-1]) if sid in edited}
    return [sid for sid in ids if sid in selected]


def judging_manifest():
    marker = ROOT/'judge_manifest.json'
    if marker.exists():
        return read_json(marker)
    if not (ROOT/'inference_complete.json').exists():
        raise RuntimeError('All local arms must finish before comparison-site selection')
    cases = []
    for case in read_json(ROOT/'design.json')['cases']:
        base = case_dir(case)
        episodes = {c:read_json(base/c/'episode.json') for c in CONDITIONS}
        suggestions = read_json(base/'candidates.json')['candidates']
        sids = judge_sites(case,episodes,suggestions)
        original = verified(case['revision'])['initial']
        snapshots = {'before_repair':episodes['baseline']['initial'],
                     **{c:e['final'] for c,e in episodes.items()}}
        requests, aliases = {}, {}
        for condition,document in snapshots.items():
            current = phi(restore_document(document))
            orig = phi(restore_document(original))
            notices = [{'id':f'N{i+1}', 'sid':sid, 'type':'candidate_link_check',
                'before':orig.get(sid), 'after':current[sid],
                'scope':[sid]+([current[sid]['antecedent']['sid']] if current[sid]['antecedent']['sid'] else [])}
                for i,sid in enumerate(sids)]
            payload = {'question':verified(case['row'])['question'], 'original':paragraphs(original),
                'revised':paragraphs(document),'notices':notices,'variants':[]}
            messages = [{'role':'system','content':JUDGE_PROMPT},{'role':'user','content':dumps(payload)}]
            digest = sha_text(dumps(messages))
            aliases[condition] = digest
            requests[digest] = {'messages':messages,'notice_ids':[n['id'] for n in notices],
                               'sid_map':{n['id']:n['sid'] for n in notices}}
        order = list(requests)
        random.Random(int(sha_text(case['case_id']+':repair1-judge')[:16],16)).shuffle(order)
        cases.append({'case_id':case['case_id'],'cohort':case['cohort'],'sites':sids,
            'aliases':aliases,'requests':[dict(id=h,**requests[h]) for h in order]})
    value = {'cases':cases,'prompt_sha256':sha_text(JUDGE_PROMPT),'context':'full',
        'validated_gate':ref(CHECK_ROOT/'validation/final.json'),'no_condition_names_in_API':True,
        'site_contract':'Up to 6 seeded candidate sites plus all repaired sites and their successors, common to all arms.',
        'variants':[],'quality_scope':'Link-label transitions only; no global preference verdicts.'}
    freeze(marker,value)
    return value


def judge():
    if (ROOT/'judge_complete.json').exists():
        return
    started = time.monotonic(); manifest = judging_manifest()
    api = PaidAPI('claude',root=ROOT); counts = Counter(); budget_stop = False
    try:
        for case in manifest['cases']:
            for req in case['requests']:
                path = ROOT/'judgments'/(req['id']+'.json')
                if path.exists():
                    counts['saved'] += 1; continue
                if not req['notice_ids']:
                    freeze(path,{'valid':True,'verdict':{'notices':[],'variants':[]},'no_API_needed':True})
                    continue
                result = {'valid':False,'prompt_sha256':manifest['prompt_sha256'],'attempts':[]}
                for attempt in (1,2):
                    trial_path = ROOT/'judgments'/f'{req["id"]}_try{attempt}.json'
                    if trial_path.exists():
                        trial = read_json(trial_path)
                    else:
                        trial = {'valid':False}
                        try:
                            response = api.request(req['messages'], item_id=f'REPAIR1:{req["id"]}:{attempt}',
                                max_output=min(8192,max(4096,800+200*len(req['notice_ids']))))
                            trial['response'] = response
                            trial['verdict'] = validate(response['raw'],req['notice_ids'],[])
                            trial['valid'] = True
                        except BudgetStop:
                            budget_stop = True; break
                        except Exception as exc:
                            trial['error'] = type(exc).__name__+': '+str(exc)
                        freeze(trial_path,trial)
                    result['attempts'].append(ref(trial_path))
                    if trial['valid']:
                        result.update(valid=True,verdict=trial['verdict']); break
                    if trial.get('response',{}).get('status') != 'completed':
                        break
                if budget_stop:
                    break
                freeze(path,result); counts['valid' if result['valid'] else 'invalid'] += 1
            if budget_stop:
                break
            print('repair1 judged',case['case_id'],dict(counts),flush=True)
    finally:
        api.close()
    freeze(ROOT/'judge_complete.json', {'counts':dict(counts),'budget_stop':budget_stop,
        'wall_seconds':time.monotonic()-started,'model':api.model})


def record_metrics(case, episode):
    if case['cohort'] != 'agent_dev10':
        return None
    from .pilot import cases
    from verak.v3.reward.recovery import recover_records
    from verak.v3.reward.overedit import overedit
    private = next(c for c in cases() if c['case_id']==case['case_id'])
    prepared = verified(private['prepared']); gold = prepared['corpus']
    source,initial = (restore_document(prepared[k]) for k in ('reward_source','reward_initial'))
    result = {}
    for label,document in [('before_repair',episode['initial']),('final',episode['final'])]:
        try:
            final = private_document(project_document(restore_document(document),prepared['policy_row']['normalization'],
                episode['lineage']),prepared['stable_id_aliases'])
            result[label] = {'status':'exact','records':recover_records(source,final,gold['records'],corrupted=initial),
                'R_over':overedit(source,initial,final,gold['records'])}
        except (ProjectionUnknown, KeyError) as exc:
            result[label] = {'status':'unknown','reason':str(exc)}
    return result


def rejection_type(action):
    error = action.get('error','')
    if '바꾸지 않는 행동' in error:
        return 'no_op'
    if '부분 문자열' in error or '매칭' in error:
        return 'span'
    if '종결' in error or '문장 경계' in error or '두 문장' in error:
        return 'boundary'
    if action.get('scope_rejection') or '범위' in error or '위치 밖' in error:
        return 'scope'
    return 'other'


def read_labels(jcase, condition):
    digest = jcase['aliases'][condition]
    req = next(r for r in jcase['requests'] if r['id']==digest)
    path = ROOT/'judgments'/(digest+'.json')
    result = read_json(path) if path.exists() else {}
    values = {r['id']:r['broken_after_revision'] for r in result.get('verdict',{}).get('notices',[])} if result.get('valid') else {}
    return {sid:values.get(nid,'unjudged') for nid,sid in req['sid_map'].items()}


def transitions(before, after, edited):
    value = Counter()
    for sid,old in before.items():
        new = after.get(sid,'unjudged')
        if old not in {'yes','no'} or new not in {'yes','no'}:
            value['uncertain_or_unjudged_pairs'] += 1; continue
        value['measurable_pairs'] += 1
        value['broken_before'] += old == 'yes'
        value['broken_after'] += new == 'yes'
        value['fixed'] += old == 'yes' and new == 'no'
        value['new_broken'] += old == 'no' and new == 'yes'
        value['edited_previously_fine'] += old == 'no' and sid in edited
    return dict(value)


def report():
    started = time.monotonic(); design = read_json(ROOT/'design.json')
    manifest = judging_manifest(); jcases = {c['case_id']:c for c in manifest['cases']}
    rows = []; examples = []
    for case in design['cases']:
        base = case_dir(case); jc = jcases[case['case_id']]
        labels = {name:read_labels(jc,name) for name in ('before_repair',)+CONDITIONS}
        # Paired table uses the same sentence sites in all four snapshots.
        common = {sid for sid in jc['sites'] if all(l.get(sid) in {'yes','no'} for l in labels.values())}
        for condition in CONDITIONS:
            episode = read_json(base/condition/'episode.json'); scores = read_json(base/condition/'scores.json')
            edited = changed_sids(episode['initial'],episode['final'])
            stages = episode['stages']; actions = episode['actions']
            row = {'case_id':case['case_id'],'cohort':case['cohort'],'condition':condition,
                'protocol':episode['korean_no_protocol_error'],'pipeline_protocol':episode['pipeline_no_protocol_error'],
                'all_done':episode['all_done'],'requests':episode['worker_requests'],'gate_requests':episode['gate_requests'],
                'actions':len(actions),'valid_actions':sum(a['valid'] for a in actions),
                'rejections':dict(Counter(rejection_type(a) for a in actions if not a['valid'])),
                'stop_done':sum(s['ending']=='STOP' and s['stop_status']=='done' for s in stages),
                'stop_blocked':sum(s['ending']=='STOP' and s['stop_status']=='blocked' for s in stages),
                'endings':dict(Counter(s['ending'] for s in stages)),
                'forced_skips':sum(len(s.get('forced_events',[])) for s in stages),
                'unresolved_tasks':len(episode['unresolved_tasks']),
                'gate_decisions':dict(Counter(r['decision'] for r in (episode['gate'] or {}).get('decisions',[]))),
                'gate_valid':(episode['gate'] or {}).get('valid'),
                'changed_sentences':len(edited),'total_sentences':len(restore_document(episode['initial']).units),
                'Q_delta_repair':scores['final']['mean']-scores['before_repair']['mean'],
                'Q_delta_original':scores['final']['mean']-scores['original']['mean'],
                'record_metrics':record_metrics(case,episode),
                'label_counts':dict(Counter(labels[condition].values())),
                'paired_links':transitions({s:labels['before_repair'][s] for s in common},
                    {s:labels[condition][s] for s in common},edited),
                'all_sites':len(jc['sites']), 'paired_sites':len(common)}
            rows.append(row)
        examples.append({'case_id':case['case_id'],'cohort':case['cohort'],'original':paragraphs(verified(case['revision'])['initial']),
            'before_repair':paragraphs(read_json(base/'baseline/episode.json')['initial']),
            'variants':{c:paragraphs(read_json(base/c/'episode.json')['final']) for c in CONDITIONS},
            'labels':labels,'common_measurable_sites':sorted(common)})
    freeze(ROOT/'measurement_rows.json',rows)
    freeze(ROOT/'examples.json',examples)
    aggregates = {}
    for cohort in ('all','smoke20','agent_dev10'):
        for condition in CONDITIONS:
            subset = [r for r in rows if r['condition']==condition and (cohort=='all' or r['cohort']==cohort)]
            aggregate = {'n':len(subset)}
            for field in ('protocol','pipeline_protocol','all_done','requests','gate_requests','actions','valid_actions',
                          'stop_done','stop_blocked','forced_skips','unresolved_tasks','changed_sentences','total_sentences','paired_sites','all_sites'):
                aggregate[field] = sum(r[field] for r in subset)
            for field in ('rejections','endings','gate_decisions','paired_links','label_counts'):
                aggregate[field] = dict(sum((Counter(r[field]) for r in subset),Counter()))
            for field in ('Q_delta_repair','Q_delta_original'):
                aggregate[field] = sum(r[field] for r in subset)/len(subset)
            aggregates[cohort+'/'+condition] = aggregate
    operator_rows = []
    for condition in CONDITIONS:
        relevant = [r['record_metrics'] for r in rows if r['condition']==condition and r['record_metrics']]
        for op in sorted({rec['op'] for r in relevant for rec in r.get('final',{}).get('records',[])}):
            before,after = [],[]
            for row in relevant:
                if all(row[k]['status']=='exact' for k in ('before_repair','final')):
                    b = {r['record_id']:r for r in row['before_repair']['records']}
                    for record in row['final']['records']:
                        if record['op']==op and record['record_id'] in b:
                            before.append(b[record['record_id']]['main']); after.append(record['main'])
            operator_rows.append({'condition':condition,'operator':op,'n':len(after),
                'before_full':sum(x==1 for x in before),'after_full':sum(x==1 for x in after),
                'mean_recovery':sum(after)/len(after) if after else None})
    freeze(ROOT/'metrics.json',{'conditions':aggregates,'operators':operator_rows})
    cost = read_json(ROOT/'api/accounting.json') if (ROOT/'api/accounting.json').exists() else {'confirmed_usd':0,'reserved_usd':0}
    lines = ['# V5_REPAIR1 — 항목별 수리와 수정 필요성 판단','',
        '학습 없이 고정된 smoke20 + agent_dev10에서 한국어 수리 방식 세 가지를 비교했다. '
        '모든 조건은 저장된 동일 Revision 결과·MAP·관찰에서 출발한다. test cohort는 사용하지 않았다.','',
        '## 실험 계약','',
        '- baseline: 모든 후보를 한 번에 제시, 기존 14-step 전체 글 한국어 단계.',
        '- queue: 문장별 작업, 항목당 최대 3요청, 같은 상태의 거절 행동 반복은 미해결로 넘김. '
        '목표 작업에 최대 10요청을 쓰고 전체 글 점검에 최소 4요청을 남김. 총 14요청.',
        '- queue_gate: 같은 큐에 고정 Korean epoch2의 repair/keep/uncertain 일괄 판단 1회를 추가. '
        '판단 호출은 별도 비용·요청 수로 보고. repair만 목표 작업으로 배정하며 전체 글 점검은 유지.',
        '- baseline→queue는 작업 분할·범위 제한·반복 제어의 묶음 비교이다. 반복 차단 하나의 효과로 해석하지 않는다.',
        '- 문체·띄어쓰기 등의 공적 규칙 후보와 phi 변화 알림을 문장별로 묶고 원문 순서로 배정했다. '
        '정답 레코드는 추론·판단·후보 선정에 제공하지 않았다.',
        '- 순수 프로토콜 완주는 강제 건너뛰기, 미처리 항목, 형식/문맥/step 제한을 모두 실패로 포함한다. '
        '근거 있는 모델 STOP(blocked)는 유효 보고이지만 all-done은 아니다. 강제 종료는 성공으로 바꾸지 않았다.',
        '- 이전 V4_FIX1 보정 어댑터는 사용하지 않았다. 고정 Korean epoch2, v4.6, temperature=0/top_p=1/seed47. '
        '이 진단은 한국어 수리 제어를 검증하며 구조 수정 능력 개선을 검증하지 않는다.','',
        '## 실행·종료','',
        '| Cohort | Condition | Korean protocol | Pipeline protocol | All done | Valid/actions | Worker/gate requests | Forced skips | Pending tasks |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|']
    for key,a in aggregates.items():
        cohort,condition=key.split('/')
        lines.append(f'| {cohort} | {condition} | {a["protocol"]}/{a["n"]} | {a["pipeline_protocol"]}/{a["n"]} | '
            f'{a["all_done"]}/{a["n"]} | {a["valid_actions"]}/{a["actions"]} | {a["requests"]}/{a["gate_requests"]} | '
            f'{a["forced_skips"]} | {a["unresolved_tasks"]} |')
    lines += ['','Pipeline protocol includes failures already present in the shared saved Revision. '
              'All-done here refers to the Korean repair stages; no claim that a model report proves task quality.','',
              '| Condition | STOP done / blocked stages | Rejections | Stage endings | Gate decisions |',
              '|---|---:|---|---|---|']
    for c in CONDITIONS:
        a=aggregates['all/'+c]
        lines.append(f'| {c} | {a["stop_done"]} / {a["stop_blocked"]} | {a["rejections"]} | {a["endings"]} | {a["gate_decisions"]} |')
    lines += ['','## 오류 수정·새 오류·채점기','',
        '검증된 C의 Claude 프롬프트를 그대로 사용하고 full essay를 제공했다. variants=[]로 링크 판정만 받았다. '
        '조건명·정답 레코드·점수·gate 판단은 숨겼으며 같은 입력 스냅샷은 판정을 공유했다. '
        '후보 중 최대 6문장을 seed로 뽑고, 어느 조건이든 실제 수정한 문장 및 그 다음 문장을 추가한 공통 집합이다. '
        '아래 전후 전이는 네 상태 모두 yes/no인 공통 문장만 포함한다. uncertain/unjudged를 정상으로 간주하지 않는다. '
        '정상이던 문장 수정은 불필요한 수정의 후보 지표이며 모든 표현 개선이 불필요했다는 뜻은 아니다.','',
        '| Cohort | Condition | Paired sites / selected | Broken before | Fixed | New broken | Broken after | Edited previously fine | Mean ΔQ repair | Mean ΔQ original |',
        '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for key,a in aggregates.items():
        cohort,c=key.split('/'); m=a['paired_links']
        lines.append(f'| {cohort} | {c} | {a["paired_sites"]}/{a["all_sites"]} | {m.get("broken_before",0)} | '
            f'{m.get("fixed",0)} | {m.get("new_broken",0)} | {m.get("broken_after",0)} | {m.get("edited_previously_fine",0)} | '
            f'{a["Q_delta_repair"]:.4f} | {a["Q_delta_original"]:.4f} |')
    lines += ['','ΔQ uses only the unchanged GPU reference scorer and compatible saved GPU cache. '
        'A single paired diagnostic run does not establish general writing quality. Claude link decisions have '
        'the validation limitations described in V5_CHECK; judge stochasticity can affect transitions.','',
        '## agent_dev10 복원 및 과잉 수정','',
        '| Condition | Operator | Before full | After full | Mean recovery |','|---|---|---:|---:|---:|']
    for r in operator_rows:
        mean='n/a' if r['mean_recovery'] is None else f'{r["mean_recovery"]:.4f}'
        lines.append(f'| {r["condition"]} | {r["operator"]} | {r["before_full"]}/{r["n"]} | {r["after_full"]}/{r["n"]} | {mean} |')
    lines += ['','| Condition | Exact projections / 10 | Mean R_over before | Mean R_over final | Changed sentences / total (all30) |',
              '|---|---:|---:|---:|---:|']
    for c in CONDITIONS:
        rs=[r['record_metrics'] for r in rows if r['condition']==c and r['record_metrics']]
        exact=[r for r in rs if all(r[k]['status']=='exact' for k in ('before_repair','final'))]
        means=[sum(r[k]['R_over']['value'] for r in exact)/len(exact) if exact else None for k in ('before_repair','final')]
        a=aggregates['all/'+c]
        lines.append(f'| {c} | {len(exact)}/10 | {means[0]} | {means[1]} | {a["changed_sentences"]}/{a["total_sentences"]} |')
    contract=read_json(ROOT/'api_contract.json')
    lines += ['','## 시간·비용·산출물','',
        f'- GPU inference/scoring: {read_json(ROOT/"inference_complete.json")["wall_seconds"]:.1f} seconds.',
        f'- Claude judging: {read_json(ROOT/"judge_complete.json")["wall_seconds"]:.1f} seconds.',
        f'- New paid cost: ${cost["confirmed_usd"]:.6f}; reservations ${cost["reserved_usd"]:.6f}; '
        f'cap ${contract["total_cap_usd"]:.6f}. Previous CHECK + new: ${contract["prior_Claude_confirmed_usd"]+cost["confirmed_usd"]:.6f} / $8.',
        f'- Budget stop: {read_json(ROOT/"judge_complete.json")["budget_stop"]}. Missing coverage remains explicit.',
        '- Luna/API editing calls: 0. No training. Test cohort untouched.',
        f'- All 30 before/after/labels: `{ROOT}/examples.json`.',
        f'- Immutable design and prompt/code hashes: `{ROOT}/design.json`.',
        f'- Per-episode audit: `{ROOT}/measurement_rows.json`.','']
    REPORT.write_text('\n'.join(lines))
    freeze(ROOT/'report_complete.json',{'report':ref(REPORT),'measurement_seconds':time.monotonic()-started})
