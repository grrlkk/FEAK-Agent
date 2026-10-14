"""Render aggregate Phase 6 evidence; never copy essay text into reports."""
from datetime import datetime
from zoneinfo import ZoneInfo

from ..common import read_json
from ..phase2 import read_jsonl
from .summary import table, fmt


def pct(value):
    return f'{100*value:.2f}%'


def interval(v):
    return 'NA' if v['ci95'] is None else f"[{fmt(v['ci95'][0])}, {fmt(v['ci95'][1])}]"


def render_report(config):
    output = config['paths']['phase6_output']
    data = read_json(output/'metrics.json')
    validation = read_json(output/'validation.json')
    if not validation['passed']:
        raise ValueError('Full report requires passing final validation')
    conditions, design = data['conditions'], data['design']
    date = datetime.now(ZoneInfo('Asia/Seoul')).strftime('%Y-%m-%d')
    text = [f'# VERAK v3 Phase 6 — 학습 전 구조 비교와 기준선\n\n작성일: {date}. **상태: 완료. Phase 7은 시작하지 않았다.**',
        'Addendum을 우선 적용했다. 구조를 선택하거나 LoRA를 학습하지 않았다. 원천 데이터는 valid에서 만든 agent_train/dev뿐이며, scorer train/test 원천은 열지 않았다. 아래 결과는 dev의 탐색적 비교와 **LLM-verified** 판정이다. 인간 평가나 test 성능을 뜻하지 않는다.',
        '## 1. 내용 범위와 삭제 복구 가능성 필터',
        '`imple/VERAK_V3_SPEC_ADDENDUM_1.md`에 §9 Content scope를 추가하고 GLOBAL/KOREAN/single 프롬프트에 반영했다. 글 안에 있는 내용의 이동·연결·재표현은 허용하고, 글 어디에도 없는 이유·사실·사례·숫자·출처·경험은 일반 상식이어도 본문에 추가하지 않는다. 부족한 내용은 STOP summary에 글쓴이의 보충 과제로 적으며, 모든 STOP summary는 실제 수정 내역을 설명하도록 지시했다. 이는 프롬프트 제약이며 새 내용 생성을 기계적으로 보장하는 검증기를 추가한 것은 아니다.',
        'G_DELETE_SUPPORT의 각 record를 gpt-6.1-sol/high로 한 번씩 판정했다. 판정자만 삭제 문장을 보고, 복구 가능성의 근거는 남은 손상 글 전체로 제한했다. false record가 하나라도 있는 essay를 제거했으며 대체하거나 재균형화하지 않았다. 원래 최종 코퍼스는 보존했다.']
    rows = []
    for split, stats in data['filter'].items():
        for stage in ('before', 'after'):
            v = stats[stage]
            rows.append([split, stage, v['essays'], v['sources'], v['operators'].get('G_DELETE_SUPPORT', 0),
                         *(pct(v['local_shares'][k]) for k in ('WORD', 'SENTENCE', 'TEXT'))])
    text.append(table(['Split', '시점', 'Essays', 'Sources', '삭제 records', 'WORD', 'SENTENCE', 'TEXT'], rows))
    judgments = read_jsonl(output/'filter/judgments.jsonl')
    text.append(table(['Split', '판정 records', 'recoverable=true', 'false', '제외 essays'], [
        [s, len(v), sum(r['judgment']['recoverable'] for r in v), sum(not r['judgment']['recoverable'] for r in v),
         len(data['filter'][s]['dropped_ids'])] for s in ('agent_train', 'agent_dev')
        for v in [[r for r in judgments if r['split'] == s]]]))
    text += ['필터 후 파일: `verak/v3/data/corrupt_recoverable/agent_train.jsonl`, `agent_dev.jsonl`. `config.yaml`의 `paths.active_corrupt`가 이 경로를 가리킨다. 원문과 판단 근거는 로컬 JSONL에만 두고 보고서에는 옮기지 않았다.',
        '## 2. 고정 실험 설계',
        'seed 53으로 L1–L4 각 15편, 총 60편을 뽑았다. 60편의 source는 모두 다르며 두 구조와 두 재작성 기준선에 동일하게 사용했다. CHECK 실험은 그중 30편(seed 54, L1/L2/L3/L4=8/8/7/7)이다. 실제 글은 기존 Phase 3/3b 후보·최종 코퍼스와 Phase 4 복구 표본의 source를 모두 제외한 뒤 seed 57로 장르별 10편을 선택했다. 국소 분석은 필터 후 dev source 100편(seed 59)이다.',
        f"60편 장르: `{design['paired_genres']}`. 국소 분석 100편 장르: `{design['local_genres']}`. ID·원천 hash·프롬프트 hash는 `outputs/phase6/design.json`에 동결했다.",
        'Sol teacher와 Sol one-shot은 **gpt-6.1-sol / low**, 별도 판정은 **gpt-6.1-sol / high**다. Kanana는 GPU0의 기존 vLLM에서 `kakaocorp/kanana-1.5-8b-instruct-2505`, revision `c963a5f4f6496c749f94064a20b33028b0db9f19`를 adapter 없이 사용했다. 점수는 GPU1의 Phase 1 고정 scorer, k=1이다. 입력 한도 3,072와 compact view 한도 3,000을 유지하고 자르지 않았다.',
        'two_stage는 GLOBAL 14/2 CHECK + KOREAN 14/2 CHECK, single은 24/4 CHECK다. 따라서 구조 비교에는 이 기존 예산 차이도 포함된다. 보상은 Phase 4/5의 `R = R_rec + 0.3 R_q − 0.5 R_over − 0.01 steps` 그대로다. R_rec는 알려진 손상의 복구값이며, R_over는 손상·coupled record가 참조하지 않는 원문 문장의 형태소 편집거리 비율이다. R_over를 불필요한 수정의 인간 정답 비율로 해석하지 않는다. steps에는 STOP·CHECK·오류도 포함한다. 장르별 dead zone을 적용한다. 정답·손상 record·삭제 원문·원문 점수는 수정 모델에 제공하지 않았다.',
        '## 3. two_stage와 single: 같은 60편의 비교']
    text.append(table(['조건', '완료', 'R', 'R_rec', 'R_over', 'ΔQ', 'steps', '비용/편'], [
        [name, f"{conditions[name]['completed']}/{conditions[name]['n']}",
         *(fmt(conditions[name]['reward']['combined'][k]['mean']) for k in ('R', 'R_rec', 'R_over')),
         fmt(conditions[name]['delta_q']['mean']), fmt(conditions[name]['steps']['mean']), '$'+fmt(conditions[name]['cost']['mean'])]
        for name in ('two_stage', 'single')]))
    comparisons = data['two_stage_minus_single']
    fields = ['combined.R', 'combined.R_rec', 'combined.R_over', 'steps', 'cost']+['recovery_'+k for k in ('GLOBAL', 'WORD', 'SENTENCE', 'TEXT')]
    text.append(table(['지표', 'paired n', 'two_stage − single', 'bootstrap 95% CI'], [
        [k, comparisons[k]['n'], fmt(comparisons[k]['difference']), interval(comparisons[k])] for k in fields]))
    text.append(f"이 표본의 combined R 차이는 {fmt(comparisons['combined.R']['difference'])}, R_over 차이는 {fmt(comparisons['combined.R_over']['difference'])}다. 복구값 R_rec의 차이 구간은 {interval(comparisons['combined.R_rec'])}이므로 combined 보상과 복구 자체를 구분해 해석한다. 비교 조건은 역할 분리·허용 행동·예산이 함께 다르며, 역할 분리만의 인과 효과를 분리한 실험은 아니다.")
    text += ['CI는 L1–L4 안에서 **글 쌍을 함께** 10,000회 복원추출한 percentile 구간(seed 61)이다. 해당 수준 record가 없는 글을 0점으로 넣지 않았다. `recovery_GLOBAL`은 coupled까지 포함한 record 복구값이며, GLOBAL main만의 진단은 `paired/metrics.json`의 `rec_GLOBAL`에 별도로 있다. 수준별 차이는 글별 해당 수준 평균의 paired 차이다. 문항 단위 cluster CI가 아니므로 문항 내 상관을 별도로 보정한 추론은 아니다.',
        '**사용자 지시대로 two_stage/single 중 하나를 선택하지 않았다.**',
        '## 4. CHECK 한 번 사용을 요청한 조건']
    text.append(table(['지표', 'paired n', 'CHECK 조건 평균', '동일 30편 기본 평균', '차이', '95% CI'], [
        [k, v['n'], fmt(v.get('left_mean')), fmt(v.get('right_mean')), fmt(v['difference']), interval(v)]
        for k, v in data['check_once_minus_default'].items()]))
    text.append(table(['조건', '역할', '평균 CHECK', '정확히 한 번 사용 / n'], [
        [name, role, fmt(value['checks']['mean']), f"{value['check_exactly_once']}/{stats['n']}"]
        for name, stats in [('기본 paired subset', data['check_default_subset']), ('CHECK 요청', conditions['check_once'])]
        for role, value in stats['roles'].items()]))
    check_r = data['check_once_minus_default']['combined.R']
    text.append(f"CHECK 요청의 combined R 차이는 {fmt(check_r['difference'])}, 95% CI는 {interval(check_r)}다. 역할별 성분과 비용의 차이는 위 표에 함께 제시했다.")
    text += ['이 조건은 각 역할에 STOP 전 CHECK를 정확히 한 번 쓰도록 **프롬프트로 요청**했다. 행동을 강제하거나 숨은 CHECK를 추가하지 않았다. 따라서 실제 사용률과 함께 해석해야 한다. 효과의 채택 여부는 결정하지 않았다.',
        '## 5. 실제 글 30편: 전이와 새 내용 추가',
        '원래 글과 최종 글을 별도 요청으로 비교했다. 판정자는 agent의 thought·수정 내역·점수·보상을 보지 않았다. real 모드는 reward=None이며 아래 ΔQ는 종료 후의 측정이다.']
    real = data['real_content']
    text.append(table(['LLM-verified 항목', '건수', '비율'], [
        ['new_content', f"{real['new_content']}/{real['n']}", pct(real['new_content_rate'])],
        ['meaning_changed', f"{real['meaning_changed']}/{real['n']}", pct(real['meaning_changed_rate'])]]))
    text.append(table(['장르', 'n', 'new_content', 'meaning_changed'], [[g, v['n'], v['new_content'], v['meaning_changed']]
        for g, v in real['by_genre'].items()]))
    text.append(f"new_content IDs: `{real['new_content_ids']}`. meaning_changed IDs: `{real['meaning_changed_ids']}`. 정확한 구절과 note는 `real_judgments/`에 저장했다. 이는 같은 Sol 계열을 별도 문맥에서 사용한 첫 추정치이며 사람의 정확도나 독립 모델 합의가 아니다.")
    measure = conditions['real']
    text.append(table(['비LLM 측정', '값'], [
        ['평균 Kanana ΔQ', fmt(real['delta_q']['mean'])],
        ['평균 변경 문장 비율', pct(measure['measurement']['changed_share']['mean'])],
        ['평균 텍스트 변경·삭제 비율', pct(measure['measurement']['text_changed_share']['mean'])],
        *[[k+' 행동 수', measure['action_types'].get(k, 0)] for k in ('MOVE', 'sentence_insert', 'sentence_delete', 'in_sentence_EDIT')],
        *[[k+' 변화/추가 수', v] for k, v in measure['cohesion_counts'].items()],
        ['off-style 문장 합계: 시작 → 종료', f"{round(measure['measurement']['off_style_before']['mean']*30)} → {round(measure['measurement']['off_style_after']['mean']*30)}"]]))
    text += ['변경 비율의 분모는 시작 글의 문장 수이며 표현 변경·삭제·상대 순서/문단 변경을 합친다. 새 문장 삽입은 별도 수로 보존한다. 행동 수는 유효 행동의 누적 수여서 UNDO 뒤 순변화와 다를 수 있다. 명시 주어 삽입은 Bareun의 새 주어 표지가 insertion diff에 포함되는 경우만 센다. 극성·양태·접속 관계 비교는 대응 문장 대상이며, 문장 전체 삭제는 삭제 수로 별도 집계한다. off-style은 각 시점의 대표 문체 기준이다. 이들은 동결한 분석기의 관찰이며 의미 오류라고 단정하지 않는다.',
        '## 6. 국소 한국어 변형과 채점 점수의 관계',
        '100개 dev source에 각 연산자를 **한 번에 하나씩 원문에서 독립적으로** 적용했다. 기존 surface table과 Bareun 재검증을 사용했다. 패턴이 없는 글에는 변형을 만들어 넣지 않았으며 새 LLM QC는 호출하지 않았다. 따라서 모든 변형이 인간에게 나쁜 수정이라는 보장은 없고, 이 표는 연산자 변화에 대한 scorer의 반응이다. Δ는 corrupted − source다.']
    text.append(table(['연산자', '패턴 있음', '검증 통과', 'discard 시도', '평균 ΔQ', 'SD', 'ΔQ<0'], [
        [op, v['applicable'], v['verified'], v['discarded_applications'], fmt(v['delta_q']['mean']), fmt(v['delta_q']['std']), pct(v['negative_delta_share'])]
        for op, v in data['local_link']['operators'].items()]))
    text.append(table(['연산자']+[f'R{i}' for i in range(1, 9)], [[op]+[fmt(d['mean']) for d in v['delta_rubric']]
        for op, v in data['local_link']['operators'].items()]))
    text += ['R1–R8은 scorer의 고정 출력 위치다. 장르별 루브릭 이름이 달라도 같은 위치끼리 비교했다. 표본별 원점수·변형 점수·record·discard 이유는 `local_link/essays/`에 있다.',
        '## 7. 한 번 전체 재작성 기준선',
        '동일한 60편의 손상 글과 문항만 제시하고 “개선하되 글쓴이의 기존 내용만 유지”하도록 했다. Sol/low 한 번, 미학습 Kanana base 한 번이며 후보 선택이나 후처리 LLM을 사용하지 않았다. 전체 생성문을 그대로 채점했다. single-mode combined reward를 적용하고 한 번의 생성을 1 step으로 셌다.']
    text.append(table(['조건', 'n', 'R', 'R_rec', 'R_over', 'ΔQ', '변경 문장 비율', 'API 비용/편'], [
        [name, conditions[name]['n'], *(fmt(conditions[name]['reward']['combined'][k]['mean']) for k in ('R', 'R_rec', 'R_over')),
         fmt(conditions[name]['delta_q']['mean']), pct(conditions[name]['measurement']['changed_share']['mean']), '$'+fmt(conditions[name]['cost']['mean'])]
        for name in ('two_stage', 'single', 'rewrite_sol', 'rewrite_kanana')]))
    text.append(table(['조건', '극성', '양태', '접속 관계', '주어 삽입', 'off-style 시작→끝'], [
        [name, *(conditions[name]['cohesion_counts'][k] for k in ('polarity', 'modality', 'conjunction_relation', 'explicit_subject_inserted')),
         f"{round(conditions[name]['measurement']['off_style_before']['mean']*60)} → {round(conditions[name]['measurement']['off_style_after']['mean']*60)}"]
        for name in ('two_stage', 'single', 'rewrite_sol', 'rewrite_kanana')]))
    text.append(table(['조건', '난이도', 'n', 'R_rec', 'R_over'], [[name, level, stats['n'],
        fmt(stats['reward']['combined']['R_rec']['mean']), fmt(stats['reward']['combined']['R_over']['mean'])]
        for name in ('two_stage', 'single', 'rewrite_sol', 'rewrite_kanana') for level, stats in conditions[name]['by_level'].items()]))
    text.append('재작성에는 행동 기반 stable ID가 없으므로 **손상 입력과 출력만** Hungarian 문자열 유사도 정렬(임계값 .35)로 대응했다. 원문 정답이나 record는 정렬에 쓰지 않았다. 새 문장·분할·큰 재표현의 정렬 오류는 복구와 과수정 수치에 영향을 줄 수 있다. 각 대응의 유사도와 미대응 수를 저장했으며 실제 행동 로그를 만들어낸 것은 아니다. API 비용 0인 Kanana에도 로컬 GPU 계산 비용은 발생한다.')
    text.append(table(['기준선', '대응', '유사도 < .6 대응', '새 단위', '입력 미대응'], [[name, *[conditions[name]['baseline_alignment'][k]
        for k in ('matched', 'matched_below_0_6', 'new_units', 'unmatched_input')]] for name in ('rewrite_sol', 'rewrite_kanana')]))
    text += ['agent의 EDIT/MOVE 직후 변화 알림도 각 trajectory 및 조건별 `metrics.json`의 `per_action_cohesion_notices`에 집계했다. 시작부터 존재하던 상태 대신 각 행동 직전과 직후의 변화를 센다. 개선을 위한 변화도 포함되므로 알림 수를 “오류 개수”로 부르지 않는다.',
        '## 8. 비용·완료·검증']
    api = data['api']
    text.append(table(['단계', 'API 시도', '확정 usage 기반 USD'], [[k, v['calls'], f"${v['confirmed_usd']:.6f}"]
        for k, v in api['by_stage'].items()]))
    text.append(f"총 API 요청 **{api['calls']}회**, 확인된 usage 기준 비용 **${api['confirmed_usd']:.6f} / $35**. 남은 timeout 예약 ${api['reserved_usd']:.6f}. input {api['usage'].get('input',0):,}, output {api['usage'].get('output',0):,}, reasoning {api['usage'].get('reasoning',0):,}, cache-read {api['usage'].get('cache_read',0):,}, cache-write {api['usage'].get('cache_write',0):,} tokens. reasoning은 output에 포함되므로 중복 과금하지 않았다. 단가는 input $2/M, output $10/M, cache-read $0.10/M, cache-write $2.50/M이며 청구서가 아닌 사용량 추정치다. [공식 Sol 모델 문서](https://developers.openai.com/api/docs/models/gpt-6.1-sol).")
    text.append(table(['조건', '완료', '유효하지 않은 행동/전체', '역할 위반', 'cache-read/input'], [[name,
        f"{v['completed']}/{v['n']}", f"{v['invalid_actions']}/{v['action_count']}", v['role_violations'], pct(v['cached_share'])]
        for name, v in conditions.items() if not name.startswith('rewrite_')]))
    text.append(table(['조건', '역할', '평균 steps', '평균 CHECK', '종료 사유'], [[name, role,
        fmt(v['steps']['mean']), fmt(v['checks']['mean']), str(v['terminations'])]
        for name in ('two_stage', 'single', 'check_once', 'real') for role, v in conditions[name]['roles'].items()]))
    tests = (output/'tests.txt').read_text(encoding='utf-8').strip().splitlines()[-1]
    text += [f"`python -m pytest -q tests verak/tests verak/v3/tests`: **{tests}**. 원천 hash, 필터 정확성·무대체, 고정 표본, 모든 측정·보상, 역할별 프롬프트, append-only 대화, 중복 API 완료 요청, call/cost 한도 등 artifact 검증 **{validation['checks']}개 통과**. 최대 동시 API 요청은 {validation['maximum_API_concurrency']}개다. STOP 요약의 비어 있지 않음은 `{validation['stop_summaries']}`로 확인했고 내용의 충분성을 추가 LLM으로 재판정하지 않았다.",
        '## 9. 실행 중 조정과 한계',
        f"- 첫 실행의 클라이언트 시도 {api['blocked_before_send']}개는 샌드박스 DNS 오류로 HTTP 전에 실패했다. 원시 기록과 `api/sandbox_transport_audit.json`을 보존했으며 API 요청 수/비용에서는 제외했다. 이후 DNS 사전 확인과 최대 4개 작업만 미리 제출하는 방식으로 바꿨다. 네트워크 접근 허용 뒤 필터를 실행했다. 실제 전송 단계 오류: `{api['dispatched_error_types']}`. 재시도는 별도 원장에 기록했다.",
        '- 첫 통합 점검에서 Transformers의 동시 lazy import 오류를 발견해 최초 import를 메인 스레드에서 완료하도록 고쳤다. 성공한 재작성 결과는 재생했고 agent 호출은 초기화 이후에 시작했다. 새 테스트와 기존 테스트를 다시 확인했다. 기존 HTTP 테스트의 샌드박스 포트 제한은 접근 허용 후 재검증했다.',
        '- 명시 주어 집계는 새로운 주어 표지 전부가 아닌 실제 insertion diff에 포함된 표현만 세도록 명확히 했다. 저장된 레이아웃으로 파일럿과 본 실행에 동일한 집계식을 적용했으며 모델을 다시 호출하지 않았다.',
        '- 전체 분석기를 더 조정하거나 검증하지 않았다. DEP는 위치 힌트다. 보상은 알려진 손상 복원과 scorer 반응의 대리 지표이며, 글의 전반적 우수성을 독립적으로 입증하지 않는다.',
        '- 미학습 policy의 행동 JSON은 Phase 5에서 18/21=85.71%로 90%에 못 미쳤다. 이번 Kanana one-shot은 행동 JSON 실험이 아니므로 그 수치를 대체하지 않는다. 명세 Phase 6 row에 따라 후속 SFT에서 확인할 항목으로 기록한다. SFT 자체는 실행하지 않았다.',
        '- one-shot 정렬, 같은 모델 계열 판정, 작은 실제 글 표본, dev 탐색 비교의 제한을 유지한다. 논문용 최종 성능과 사람 정확도로 표현하지 않는다. 구조 선택 및 CHECK 채택 여부는 미정이며 **추가 승인을 기다리는 실행 작업은 없다. 요청 범위에서 종료한다.**',
        '## 10. 결과 경로',
        '- 전체: `verak/v3/outputs/phase6/metrics.json`, `validation.json`, `design.json`, `tests.txt`\n- 필터: `filter/manifest.json`, `filter/judgments.jsonl`, `filter/metrics.json`\n- 비교: `two_stage/`, `single/`, `paired/`, `check_once/`, `check_comparison/`\n- 실제 글: `real/`, `real_judgments/`\n- 기준선: `rewrite_sol/`, `rewrite_kanana/`, `local_link/`\n- 비용·원시 응답: `api/ledger.sqlite`, `api/requests/`, `api/accounting.json`\n\n각 실험의 metrics/report와 원시 trajectory는 로컬로 보관했다. Phase 7의 대량 trajectory 생성이나 LoRA 학습은 수행하지 않았다.']
    report = config['paths']['repo']/'imple/reports/V3_PHASE_6.md'
    report.write_text('\n\n'.join(text)+'\n', encoding='utf-8')
    return report
