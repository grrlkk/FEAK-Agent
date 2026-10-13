"""Paired, source-aware C30 quality and delegation audit for Luna Orchestrator."""
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import random
import statistics
import time

from .orchestrator1 import (ROOT, REPORT, case_dir, verify_design,
    accounting, ref, verified, read_json, write_json, safe_id, sha_text,
    SEARCH_ENABLED, control_artifacts, audit_no_search)
from .api import PaidAPI, BudgetStop
from .check_judge import freeze
from .recovery1_eval import PROMPT, validate, evidence_only
from verak.v4.content_env import dumps
from verak.v4.fix1_infer import summarize


def source_pair(case):
    return {'control':control_artifacts(case)['episode'],
            'luna':ref(case_dir(case)/'attempts'/(safe_id(case['case_id'])+'.json'))}


def make_judge_case(case, episodes, order, order_index):
    row = verified(case['row']); aliases = dict(zip(('A','B'), order))
    payload = {'question':row['question'], 'original':row['text'], 'variants':[]}
    for label, condition in aliases.items():
        episode = episodes[condition]
        if episode['initial']['text'] != row['text']:
            raise ValueError('Quality comparison has different source essays')
        payload['variants'].append({'id':label, 'text':episode['final']['text'],
            'source_evidence':evidence_only(episode.get('source_citations', {}))})
    messages = [{'role':'system', 'content':PROMPT}, {'role':'user', 'content':dumps(payload)}]
    return {'case_id':case['case_id'], 'cohort':case['cohort'], 'order':order_index,
            'aliases':aliases, 'messages':messages, 'id':sha_text(dumps(messages))}


def judge_manifest():
    path = ROOT/'judge_manifest.json'
    if path.exists(): return read_json(path)
    if not (ROOT/'inference_complete.json').exists(): raise RuntimeError('Inference not complete')
    design = verify_design()
    if design['evaluation']['prompt_sha256'] != sha_text(PROMPT):
        raise ValueError('Frozen quality prompt changed')
    rng = random.Random(design['evaluation']['seed']); requests = []
    for case in design['cases']:
        artifacts = source_pair(case); episodes = {c:verified(r) for c,r in artifacts.items()}
        if not SEARCH_ENABLED:
            for episode in episodes.values():
                audit_no_search(episode)
        order = ['control','luna']; rng.shuffle(order)
        for index, sequence in enumerate((order, list(reversed(order)))):
            request = make_judge_case(case, episodes, sequence, index)
            request['artifacts'] = artifacts
            requests.append(request)
    freeze(path, {'cases':requests, 'design':ref(ROOT/'design.json'),
        'evaluator_code':ref(Path(__file__)),
        'primary_order':0, 'reverse_order_is_sensitivity_only':True,
        'scores_actions_models_hidden':True, 'source_evidence_included':True})
    return read_json(path)


def identity_errors(verdict, messages):
    payload = json.loads(messages[1]['content'])
    texts = {v['id']:v['text'] for v in payload['variants']}
    errors = []
    for variant in verdict['variants']:
        if texts[variant['id']] == payload['original']:
            if variant['better_than_original'] == 'yes': errors.append('unchanged_original_rated_better')
            if variant['meaning_preserved'] == 'no': errors.append('unchanged_original_lost_meaning')
            if variant['new_specific_claims'] != 'none': errors.append('unchanged_original_has_new_claims')
    if texts['A'] == texts['B'] and verdict['preference'] in {'A','B'}:
        errors.append('identical_texts_strict_preference')
    return errors


def judge():
    if (ROOT/'judge_complete.json').exists(): return
    manifest = judge_manifest(); started = time.monotonic()
    # Identical input hashes share one verdict, while every planned pair remains.
    unique = {}
    for case in manifest['cases']:
        unique.setdefault(case['id'], case)
    def one(case):
        path = ROOT/'judgments'/(case['id']+'.json')
        if path.exists(): return read_json(path)['status']
        api = PaidAPI('claude', root=ROOT)
        try:
            for attempt in (1,2):
                response = api.request(case['messages'],
                    item_id=f'ORCHESTRATOR1-quality:{case["id"]}:{attempt}', max_output=3072)
                try:
                    verdict = validate(response['raw'])
                except (ValueError, KeyError, TypeError) as exc:
                    freeze(ROOT/'invalid_judgments'/f'{case["id"]}_{attempt}.json',
                           {'response':response, 'error':str(exc)})
                    continue
                errors = identity_errors(verdict, case['messages'])
                status = 'identity_contradiction' if errors else 'valid'
                freeze(path, {'status':status, 'verdict':verdict, 'identity_errors':errors, 'response':response})
                return status
            freeze(path, {'status':'invalid_json'}); return 'invalid_json'
        except BudgetStop: return 'budget_not_dispatched'
        except Exception as exc:
            write_json(ROOT/'judge_failures'/(case['id']+'.json'), {'error':str(exc)})
            return 'api_error'
        finally: api.close()
    # Primary requests first; the reverse-order check must not consume their budget.
    counts = Counter()
    with ThreadPoolExecutor(max_workers=2) as pool:
        for order in (0,1):
            selected = [c for c in unique.values() if c['order']==order]
            counts.update(pool.map(one, selected))
    freeze(ROOT/'judge_complete.json', {'unique_requests':len(unique), 'counts':dict(counts),
        'wall_seconds':time.monotonic()-started, 'accounting':accounting()})
    print(dumps(counts), flush=True)


def delegation_audit(episode):
    rows = []
    for number, stage in enumerate(episode['stages'],1):
        calls = [c for c in episode['calls'] if c.get('stage')==number and c['role']==stage['role']]
        rows.append({'role':stage['role'], 'automatic':stage['automatic'],
            'tasks':stage.get('assigned_tasks', []), 'ending':stage['ending'],
            'stop_status':stage['stop_status'], 'report':stage.get('report'),
            'valid_actions':dict(Counter(c['action']['action'] for c in calls if c['action']['valid'])),
            'invalid_actions':dict(Counter(c['action'].get('error','unknown') for c in calls if not c['action']['valid'])),
            'changed_sids':stage.get('changed_sids', []),
            'STOP_is_not_verified_task_completion':True})
    return rows


def report():
    design = verify_design()
    if not (ROOT/'inference_complete.json').exists(): raise RuntimeError('Inference incomplete')
    episodes = defaultdict(list); rows = []; private_cases = None
    from .repair1_eval import record_metrics
    from .measurement import a1_planning
    from .pilot import cases as pilot_cases
    conditions = ('control1','control2','control3','luna') if SEARCH_ENABLED else ('control1','luna')
    for case in design['cases']:
        for condition in conditions:
            if condition == 'luna':
                artifact = ref(case_dir(case)/'attempts'/(safe_id(case['case_id'])+'.json'))
                scores = read_json(case_dir(case)/'quality_scores'/(safe_id(case['case_id'])+'.json'))
            else:
                saved = control_artifacts(case, condition[-1])
                artifact = saved['episode']; scores = verified(saved['scores'])
            episode = verified(artifact); episodes[condition].append(episode)
            if not SEARCH_ENABLED:
                audit_no_search(episode)
            row = {'case_id':case['case_id'], 'cohort':case['cohort'], 'condition':condition,
                'artifact':artifact, 'completion':episode['completion'],
                'Q_delta':scores['final']['mean']-scores['original']['mean'],
                'delegations':delegation_audit(episode), 'record_metrics':None, 'scope_coverage':None}
            if not SEARCH_ENABLED:
                row.update(deferred_tasks=episode.get('deferred_search_tasks', []),
                           writer_notes=episode.get('writer_notes', []))
            if case['cohort'] == 'agent_dev10':
                row['record_metrics'] = record_metrics(case, {
                    'initial':episode['initial_token_document'], 'final':episode['final_token_document'],
                    'lineage':episode.get('lineage', {})})
                if private_cases is None: private_cases = {c['case_id']:c for c in pilot_cases()}
                prepared = verified(private_cases[case['case_id']]['prepared'])
                plan = a1_planning(prepared, episode)
                row['scope_coverage'] = [{k:r[k] for k in ('record_id','operator','in_explicit_delegated_scope')}
                                        for r in plan['records']]
            rows.append(row)
    operations = {c:summarize(a, len(design['cases'])) for c,a in episodes.items()}
    metrics = {}; recovery = {}
    for condition, values in episodes.items():
        selected = [r for r in rows if r['condition']==condition]
        metrics[condition] = {'episodes':len(values), 'no_protocol_error':operations[condition]['no_protocol_error'],
            'all_stages_STOP_done':operations[condition]['all_stages_STOP_done'],
            'mean_Q_delta':statistics.mean(r['Q_delta'] for r in selected),
            'mean_delegations':operations[condition]['mean_model_delegations'],
            'worker_rejections':sum(not a['valid'] for e in values for role in ('revision','korean') for a in e['actions'][role]),
            'orchestrator_ending':operations[condition]['orchestrator_endings']}
        if not SEARCH_ENABLED:
            metrics[condition].update(
                deferred_tasks=sum(e.get('deferred_task_count', 0) for e in values),
                essays_with_deferred_tasks=sum(bool(e.get('deferred_task_count')) for e in values),
                search_attempts=sum(a['action']=='SEARCH' for e in values for acts in e['actions'].values() for a in acts),
                successful_searches=0, retrieved_passages=0)
        by_op = defaultdict(list)
        for r in selected:
            if r['record_metrics'] and r['record_metrics']['final']['status']=='exact':
                for rec in r['record_metrics']['final']['records']: by_op[rec['op']].append(rec['main'])
        recovery[condition] = {op:{'full':sum(x==1 for x in v), 'measurable':len(v),
            'mean_main':statistics.mean(v)} for op,v in by_op.items()}
    judgments = []; preferences = {0:Counter(), 1:Counter()}; quality = {}
    if (ROOT/'judge_manifest.json').exists():
        for request in read_json(ROOT/'judge_manifest.json')['cases']:
            path = ROOT/'judgments'/(request['id']+'.json')
            value = read_json(path) if path.exists() else {'status':'unjudged'}
            entry = {k:request[k] for k in ('case_id','cohort','order','id')}
            entry['status'] = value['status']
            if value['status']=='valid':
                verdict = value['verdict']
                entry.update(preference=request['aliases'].get(verdict['preference'], verdict['preference']),
                    reason=verdict['preference_reason'],
                    variants={request['aliases'][v['id']]:v for v in verdict['variants']})
            else: entry['preference']='unjudged'
            preferences[request['order']][entry['preference']] += 1
            judgments.append(entry)
    primary = [j for j in judgments if j['order']==0 and j['status']=='valid']
    for condition in ('control','luna'):
        variants = [j['variants'][condition] for j in primary]
        quality[condition] = {'judged':len(variants),
            **{field:dict(Counter(v[field] for v in variants)) for field in
               ('better_than_original','meaning_preserved','new_specific_claims')},
            'mean_overall':statistics.mean(v['scores']['overall'] for v in variants) if variants else None}
    reverse = {j['case_id']:j for j in judgments if j['order']==1 and j['status']=='valid'}
    comparable = [j for j in primary if j['case_id'] in reverse]
    order_check = {'paired_cases':len(comparable),
        'same_preference':sum(j['preference']==reverse[j['case_id']]['preference'] for j in comparable),
        'opposite_strict_preferences':sum({j['preference'],reverse[j['case_id']]['preference']}=={'control','luna'} for j in comparable),
        'both_prefer_luna':sum(j['preference']==reverse[j['case_id']]['preference']=='luna' for j in comparable),
        'both_prefer_control':sum(j['preference']==reverse[j['case_id']]['preference']=='control' for j in comparable)}
    by_cohort = {}
    for cohort in ('smoke20','agent_dev10'):
        group = [r for r in rows if r['cohort']==cohort]
        paired = [j for j in judgments if j['cohort']==cohort and j['order']==0]
        by_cohort[cohort] = {'primary_preference':dict(Counter(j['preference'] for j in paired)),
            'conditions':{}}
        for condition in metrics:
            selected = [r for r in group if r['condition']==condition]
            variants = [j['variants']['control' if condition=='control1' else 'luna']
                        for j in paired if j['status']=='valid'] if condition in {'control1','luna'} else []
            by_cohort[cohort]['conditions'][condition] = {
                'episodes':len(selected),
                'no_protocol_error':sum(r['completion']['no_protocol_error'] for r in selected),
                'all_stages_STOP_done':sum(r['completion']['all_stages_STOP_done'] for r in selected),
                'mean_Q_delta':statistics.mean(r['Q_delta'] for r in selected),
                'judged':len(variants),
                **{field:dict(Counter(v[field] for v in variants)) for field in
                   ('better_than_original','meaning_preserved','new_specific_claims')}}
    dispatch = {role:dict(Counter((c.get('generation') or {}).get('model', 'missing')
                    for e in episodes['luna'] for c in e['calls'] if c['role']==role))
                for role in ('orchestrator','revision','korean')}
    result = {'metrics':metrics, 'operations':operations, 'rows':rows, 'recovery':recovery,
        'judgments':judgments, 'primary_preference':dict(preferences[0]), 'reverse_preference':dict(preferences[1]),
        'quality':quality, 'by_cohort':by_cohort, 'dispatch':dispatch,
        'evaluator_code':ref(Path(__file__)), 'order_check':order_check, 'accounting':accounting(),
        'scope_is_not_correct_diagnosis':True, 'limits':design['limits']}
    write_json(ROOT/'metrics.json', result)
    if SEARCH_ENABLED:
        intro = ['# V5_ORCHESTRATOR1 — 총괄만 Luna로 교체', '',
        '기존 COOPERATE1과 같은 smoke20 + agent_dev10이다. Revision epoch1 / Korean epoch2, '
        '프롬프트·관찰·검색·행동 예산을 유지하고 총괄의 생성 모델을 gpt-6-luna로 바꿨다. '
        'API는 Chat Completions와 JSON 객체 형식을 사용했다. '
        'Luna reasoning none / temperature0, 작업자 temperature0 / top_p1 / seed47. 새 학습과 test cohort 사용 없음.', '',
        '대조군 run1을 주 비교로 고정하고 run2/3은 실행 변동 범위로 보존했다. 새 조건은 30편 1회이다. '
        '기존 Kanana 토큰 기준 입력 제한은 유지하고 API 토큰 사용량은 별도로 기록했다.', '']
    else:
        intro = ['# V5_ORCHESTRATOR1_NO_SEARCH — 검색 없는 총괄 비교', '',
            '사용자 지적에 따라 검색 조건을 바로잡았다. 같은 smoke20 + agent_dev10에서 기존 총괄과 '
            'Luna 총괄을 모두 SEARCH 없이 새로 실행했다(30편 × 2조건). 검색을 켰던 이전 결과는 이 비교의 대조군이 아니다.', '',
            'Revision epoch1 / Korean epoch2, 원래 system prompt·문맥·수정 예산·한국어 최종 점검 순서는 같다. '
            '검색이 필요한 항목은 실행 전 작성자 메모로 보류하고, 총괄과 작업자 관찰에 검색 불가를 표시한다. '
            '검색 백엔드·인덱스 점검·실제 검색 모두 비활성화했다. '
            '보류한 항목은 완료 작업으로 세지 않는다.', '',
            'Luna는 Chat Completions JSON 객체 형식, reasoning none / temperature0이다. '
            '로컬 모델은 temperature0 / top_p1 / seed47. 학습과 test cohort 사용 없음.', '']
    lines = intro + [
        '| 조건 | 정상 종료 /30 | 모든 단계 done /30 | 작업자 거절 | 위임/편 | 평균 ΔQ |',
        '|---|---:|---:|---:|---:|---:|']
    for condition, m in metrics.items():
        lines.append(f'| {condition} | {m["no_protocol_error"]} | {m["all_stages_STOP_done"]} | '
            f'{m["worker_rejections"]} | {m["mean_delegations"]:.2f} | {m["mean_Q_delta"]:.4f} |')
    if not SEARCH_ENABLED:
        lines += ['', '## 검색 차단 확인', '',
            '| 조건 | 검색 시도 | 실제 검색 | 검색 근거 | 보류 항목 | 보류 있는 글 |',
            '|---|---:|---:|---:|---:|---:|']
        for condition, m in metrics.items():
            lines.append(f'| {condition} | {m["search_attempts"]} | 0 | 0 | {m["deferred_tasks"]} | {m["essays_with_deferred_tasks"]} |')
    lines += ['', '## 글 전체 블라인드 평가', '',
        '원문과 두 최종본, 각 조건이 실제로 검색한 근거를 제시했다. 모델·행동·점수는 숨겼다. '
        '기존 RECOVERY1 전체 글 평가 프롬프트를 그대로 사용했다. 첫 무작위 순서가 주 집계이며 역순은 민감도 확인이다. '
        '같은 Claude 모델의 순서 변경 검사이며 독립 평가자 둘로 세지 않는다. '
        '두 순서의 입력이 완전히 같으면 응답을 공유한다.', '',
        '주 선호: `'+dumps(dict(preferences[0]))+'`', '',
        '역순 선호: `'+dumps(dict(preferences[1]))+'`', '',
        '순서 일관성: `'+dumps(order_check)+'`', '',
        '| 조건 | 판정 수 | 원문보다 개선 yes | 의미 보존 yes | 출처 불일치 | 추가 사실 확인 불가 | 평균 전체 /5 |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for condition, q in quality.items():
        lines.append(f'| {condition} | {q["judged"]} | {q["better_than_original"].get("yes",0)} | '
            f'{q["meaning_preserved"].get("yes",0)} | {q["new_specific_claims"].get("contradicted",0)} | '
            f'{q["new_specific_claims"].get("unknown",0)} | {q["mean_overall"]} |')
    lines += ['', '표본별 주 선호:']
    for cohort, value in by_cohort.items():
        lines += ['', '- '+cohort+': `'+dumps(value['primary_preference'])+'`']
    lines += ['', '## agent_dev10 복원', '',
        '| 연산자 | control1 완전 복원/측정 가능 | Luna 완전 복원/측정 가능 |', '|---|---:|---:|']
    for op in sorted(set(recovery['control1'])|set(recovery['luna'])):
        cells = [recovery[c].get(op, {'full':0,'measurable':0}) for c in ('control1','luna')]
        lines.append(f'| {op} | {cells[0]["full"]}/{cells[0]["measurable"]} | {cells[1]["full"]}/{cells[1]["measurable"]} |')
    lines += ['', '투영 실패는 0점으로 채우지 않는다. 범위 포함은 문제를 정확히 진단했다는 뜻이 아니며, '
        '작업자의 STOP(done)도 수정 품질의 증거가 아니다. 전체 위임 지시·변경 문장·실패 사유와 '
        '문제 위치 범위 포함은 metrics.json에 보존했다.', '',
        '## 비용·제약', '', 'API 비용: `'+dumps(accounting())+'`', '',
        '작은 재사용 진단 표본이며 새 조건 1회다. 같은 설정의 로컬 추론도 변동할 수 있다. '
        '모델 평가는 사람 정답이 아니며 최종 일반화 성능을 뜻하지 않는다.', '',
        ('Responses의 첫 실행은 여러 메시지의 행동 JSON이 연결되는 오류로 중단했다. '
        'JSON 모드 확인 요청 2건도 복수 메시지를 반환했다. '
        '그 출력·실패·소스는 별도 보존했고 비용은 위 총액에 포함했다. '
        '따라서 이 결과는 Luna + Chat Completions/JSON 설정의 결과다. 원래 응답에서 일부 행동만 뽑아 성공으로 처리하지 않았다.'
        if SEARCH_ENABLED else '이전 검색 포함 실험·형식 확인 비용과 미확정 예약금도 합산하여 기존 $3 상한 안에서 실행했다. '
        '검색 차단이 EDIT/INSERT로 새 사실을 만들어 내는 오류까지 막는 것은 아니다. '
        '이번 비교만으로 총괄 또는 작업자가 유일한 병목이라고 단정하지 않는다.'), '',
        f'자료: `{ROOT}`. 기본 실행 모델과 기존 결과는 바꾸지 않았다.']
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text('\n'.join(lines)+'\n')
    write_json(ROOT/'report_complete.json', {'report':ref(REPORT), 'at':time.time(), 'accounting':accounting()})
    print(dumps({'metrics':metrics, 'primary_preference':dict(preferences[0]), 'order_check':order_check}), flush=True)
