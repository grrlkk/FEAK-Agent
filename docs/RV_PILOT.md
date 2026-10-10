# Revision Verifier 파일럿

2026-09-28 사용자 지정 `FEAK_RV_PILOT_IMPLEMENTATION.md`의 구현이다.
로컬 원본은 `/home/chanwoo/FEAK_RV_PILOT_IMPLEMENTATION.md`, 사본은
`feak_tc_docs/paper_docs/FEAK_RV_PILOT_IMPLEMENTATION.md`다. 원고는 원격에 업로드하지 않는다.

## 모듈과 판정

| 단계 | 입력 | 출력/동작 |
|---|---|---|
| Scorer | 문항·현재 글 | 기존 Kanana의 8개 rubric |
| Planner | 문항·현재 글·8점수·8기준 설명·점수 우선순위·원문 구간 ID | 기준별 진단, 원문 근거, 수정 범위를 포함한 plan 또는 종료 사유 |
| Reviser | 문항·현재 글·고정된 plan·허용 범위 | 해당 범위의 교체 패치 또는 수정 불가 사유 |
| RV | 문항·전후 글·goal·must_preserve·diff | 네 criterion 및 decision |
| Controller | 네 criterion | ACCEPT / RETRY / STOP |

2026-09-29 확장(`rv-pilot-2026-09-29-grounded-v4.1`): 사용자가 요청한 **Kanana 점수에 따른
목표 선택 → 원문에 근거한 수정 계획**의 연결과 편집 범위를 명시한다. 원래 방법론의 한 문제·한
후보 및 RV에 의한 채택을 유지하고 Planner 내부 처리와 Planner/Reviser의 출력 계약을 확장했다.

- 원래 연속 점수를 변환하지 않고 낮은 순서로 `rubric_priority`를 만든다. 정확히 같은 점수는
  기존 rubric 순서로 정렬한다. 임의의 점수 cutoff나 후보 점수 차이를 채택 조건으로 쓰지 않는다.
- Planner의 첫 API 호출은 점수 없이 문항·원문·8개 기준 설명·구간 ID를 받아 `SourceReview`를
  반환한다. 8개 기준마다 `actionable`, `no_actionable_issue`, `needs_information`과 이유를
  기록하며 actionable에는 실제 문제·원문 인용·독자에게 미치는 영향·앞뒤 문맥 확인을 포함한다.
  모델이 낮은 점수에 맞춰 결함을 만들어 내는 것을 줄이기 위한 분리이며, 편향 제거를 보장하지 않는다.
- 코드는 이 진단 결과를 Kanana 점수 순서로 정렬해 `priority_checks`를 만들고 actionable 중
  가장 낮은 점수의 항목을 선택한다. 둘째 API 호출은 점수·우선순위·선택된 문제를 받아
  `RevisionPlan`을 만든다. 이 호출은 선택된 기준과 진단 내용을 바꿀 수 없다. 같은 진단이라도
  Kanana 점수가 달라지면 선택되는 문제가 달라질 수 있다. 점수는 여전히 목표 선택의 기준이다.
- 기존 plan의 다섯 필드에 `evidence_spans`(원문 인용 1–3개), `reader_impact`, `context_check`,
  `operation`, `edit_scope`(시작·끝 구간 ID)를 추가한다. 점수가 낮거나 설명을 더 붙일 수 있다는
  사실만으로 결함이라고 판단하지 않도록 지시한다. 인용의 존재·순서·범위는 코드로 검증하지만,
  진단 내용의 타당성이 이 형식 검증으로 보장되는 것은 아니다.
- 계획이 없으면 `outcome=no_actionable_issue` 또는 `needs_information`, `plan=null`로 종료한다.
  이 경우 둘째 계획 호출을 생략한다. 정보 부족이 있는 경우와 실제 수정 문제를 확인하지 못한 경우를
  별도 기록한다. 후자는 글의 완전성이나 품질을 보증하는 판정이 아니다.
- Reviser API는 `PatchResponse`를 반환한다. 패치는 허용한 시작·끝 구간 ID, 원문의 정확한
  `expected_text`, 대체할 `replacement`다. 프로그램이 해당 구간만 교체해 내부의
  `Revision.revised_text`를 구성한다. 범위 밖 문자·공백·줄바꿈은 원문 그대로 유지된다.
  `text_units`는 주소를 정하기 위한 결정적 구간 분할이며 한국어 문장 분석기나 품질 판정기가 아니다.
  여러 인접 구간을 하나의 편집 범위로 잡을 수 있어 문장 간·문단 수준 수정도 가능하다.
- 원문 밖 정보를 만들어야 하거나 계획을 타당하게 실행할 수 없으면 Reviser는
  `outcome=cannot_revise`, `patch=null`로 이유를 반환한다. 후보·RV·추가 채점 없이
  `revision_not_feasible`로 STOP하고, 시도에는 `acceptance_decision=NOT_GENERATED`를 남긴다.
  이를 RV의 REJECT나 성공적인 수정으로 세지 않는다.

패치 검증은 수정 범위와 원문 일치의 검증이며 의미 검증이나 별도 경로 가드가 아니다.
편집량으로 의미 보존을 대신 판정하지 않는다. RV의 입력·프롬프트·네 판정 규칙은 변경하지 않았다.
구조화된 출력은 형식을 제한하며 내용의 정확성까지 보장하지 않는다.
[OpenAI 공식 구조화 출력 문서](https://developers.openai.com/api/docs/guides/structured-outputs)

RV 기준은 goal_achievement, necessity, preservation, global_benefit이다.

1. 하나라도 FAIL이면 REJECT. FAIL과 UNCERTAIN이 함께 있으면 FAIL을 우선한다.
2. FAIL 없이 UNCERTAIN이면 같은 입력·fresh context로 한 번 재검증한다.
3. 재검증도 UNCERTAIN이면 REJECT. 모두 PASS일 때만 ACCEPT.
4. REJECT이면 같은 원문과 같은 plan에서 실패 이유와 후보를 Reviser에 주어 한 번 재수정한다.
5. 다시 거절되면 종료한다. 채택 후 새 상태를 채점하고 다음 iteration으로 간다. 최대 3회다.

모델의 decision과 criterion이 모순되면 schema 실패다. Controller는 criterion에서 결정을 다시
계산한다. 원문과 동일한 후보는 no_change로 거절한다. 추가 후보 탐색·경로 가드·rollback은 없다.

## 현재 채점기 유지

문서 예시는 0–10점이나 **사용자의 후속 지시로 현재 채점기의 척도를 유지한다.**
기존 루프처럼 기본 상태 점수는 `rf_corrected_score` 연속값이다. 기반 등급 체계는 1–9이며
RF 잔차 보정 연속값은 구간 밖일 수도 있다. 환산·clipping·소수 첫째 자리 반올림을 하지 않는다.

score_source는 rf_corrected_score(기본), soft_mean, final을 명시적으로 선택할 수 있다.
source, 원래 soft mean/std, RF 보정값, 최종 정수 등급을 기록한다. 보정기 내부의 FEAK 자질은
그대로 계산하지만 Planner와 RV에 전달하지 않는다. rubric 설명은 기존 8개 이름을 풀어 쓴
운영 프롬프트이며 별도 학습 라벨이 아니다.

**후보 점수는 RV 결정 이후에 측정한다.** baseline은 target 증가, 전체 합계 증가, 둘 중 하나 증가를
각각 기록한다. 어떤 점수 임계값도 실제 채택에 적용하지 않는다. 거절 후보의 점수는 scores_candidate,
유지된 글의 점수는 scores_after다. log_score_baseline=false이면 거절 후보 추가 채점을 생략한다.
같은 sample의 동일한 글은 최초 점수를 재사용한다. 채점 잡음을 검증한 것은 아니다.

## 독립 요청과 비용 제한

기본 GPT는 gpt-5-mini, reasoning low, 출력 최대 4096 tokens, 입력 30000 characters,
sample당 24 calls, 전체 120 calls다. SDK 자동 재시도는 끄고 JSON/schema 오류에만 최대 1회
재요청한다. schema 재시도도 호출 예산에 포함한다. API 오류·거절에는 자동 재시도·모델 교체가 없다.

각 역할과 재검증은 별도 Responses 요청이다. previous_response_id나 conversation을 사용하지 않는다.
Planner는 실제 문제가 있으면 원문 진단과 계획 작성에 두 번, 없으면 진단에 한 번 호출한다.
두 호출 모두 동일한 예산에 포함하며, 진단 호출에 점수나 우선순위는 전달하지 않는다.
RV에 점수·자질·이전 판정/근거·iteration·Planner problem·Reviser 변경 요약을 전달하지 않는다.
앞선 답변을 수정하라는 지시도 추가하지 않는다. 같은 모델이므로 통계적으로 독립된 평가자라는 뜻은 아니다.

설정 근거: [GPT-5 Mini](https://developers.openai.com/api/docs/models/gpt-5-mini),
[Structured Outputs](https://developers.openai.com/api/docs/guides/structured-outputs).
로그 usage에는 reasoning을 포함한 API 사용량이 남는다. 모델 가격과 계정 접근 권한은 실행 시 확인한다.

## 실행

```bash
conda activate feak_agent
python scripts/check_env.py
python scripts/run_pilot.py --input examples/pilot_samples.jsonl --output-dir experiments/results/rv_pilot_run1
python scripts/run_pilot.py --text-file examples/local_agent_essay.txt --question "학교의 휴대전화 사용 규칙" --sample-id phone_1 --output-dir experiments/results/rv_pilot_one
python scripts/run_pilot.py --input examples/pilot_samples.jsonl --offline-smoke --output-dir experiments/results/rv_pilot_offline
```

입력 JSONL은 `{"sample_id":"...","writing_prompt":"...","text":"..."}`다.
중복 ID나 빈 글은 모델 호출 전에 거절한다. 기본 5편, --limit은 1–10이다.
매번 새 출력 폴더를 지정한다. --kanana-device, --kanana-m, --max-iterations를 지원한다.
기본 m=3은 연결 확인용이며 논문 측정 조건을 확정하는 값이 아니다.

## 로그와 실패 처리

| 파일 | 내용 |
|---|---|
| config.json | 실제 설정, prompt version, 코드 SHA256, Git commit, 시작 시각, real/offline 구분 |
| inputs.jsonl | 선택한 문항·원문 |
| trajectories.jsonl | 후보별 점수·계획·후보·모든 RV 판정·제어 결정·유지된 상태 |
| events.jsonl | 생성 직후부터 남기는 사건: 판정/채점 중 실패한 후보도 복구 가능 |
| llm_calls.jsonl | 프롬프트/schema, raw 응답, response/model ID, 토큰, API 오류 종류 |
| summaries.jsonl | 글별 최종 상태·종료 사유·전체 경로 |
| report.json | 채택/거절/재시도/재검증/점수와 RV 불일치 건수, API 사용량 |

iteration은 1부터, attempt는 첫 후보 0, 재수정 1이다. acceptance_decision은 후보 판정,
controller_decision은 ACCEPT/RETRY/STOP이다. 재검증도 불확실하면 마지막 rv.decision은
REVERIFY지만 acceptance_decision은 REJECT다. 두 판정 모두 rv_checks에 남는다.

점수 검토 이유와 점수를 보지 않은 진단(`source_review`)은 plan 이벤트와 시도의 `planner_review`에,
실제 교체는 `edit_patch`에 기록한다. 둘째 계획 호출이 실패해도 첫 진단 응답은 `llm_calls.jsonl`에 남는다.
이 추가 기록도 RV에 보내지 않는다. report의 `candidate_attempts`는 실제 후보가 생성된 수이고,
`attempts`는 수정 불가·오류를 포함한 시도 수다. 원문 유지 종료와 정보 부족·수정 불가를 별도 집계한다.

API/schema 오류는 품질 FAIL과 구분한다. 마지막 채택 글을 반환하며 status=error로 기록하고
남은 batch를 멈춘다. RV 채택 후 상태 채점만 실패하면 채택을 되돌리지 않고 final_scores=null로
종료한다. 점수 오류가 채택 여부를 바꾸지 않는다. 각 JSONL 행은 flush한다.
강제 종료 후에는 완료되지 않은 summary 대신 events/llm_calls를 확인한다.

## 검증 범위

점수 차단과 편집 범위 제한은 의미 판단의 정확성을 보장하지 않는다. 양보·대조를 불필요하게
고치거나, 실제 자료가 없는 목표에 내용을 만들어 넣고, RV가 이를 통과시킬 가능성을 별도로
검증해야 한다. 정보 부족 종료도 모델이 이를 올바로 감지해야 작동한다.

2026-09-29의 실제 실행·통제 검사·실패 사례는 저장소 지침에 따라 로컬 결과로만 보관한다.
CLI에서 다음 파일로 원문과 수정본, 검증 조건과 남은 한계를 확인할 수 있다.

```bash
less experiments/results/rv_pilot_grounded_v41_20260929/VALIDATION.md
less experiments/results/rv_pilot_grounded_v41_20260929/comparison.md
```

`python -m pytest -q`는 점수와 채택의 분리, 재수정 원문 유지, 재검증 한도, 판정 이력 차단,
호출 예산, 거절 후보 저장, 오류 시 상태 보존을 검증한다.
합성 예제와 stub은 실제 held-out 평가나 사람 판정 일치도·학습 효과의 근거가 아니다.
baseline 로그는 **같은 RV 경로의 후보별 결정 비교**이며 별도 점수 기반 정책의 궤적이 아니다.
