# Revision Verifier 파일럿

2026-09-28 사용자 지정 `FEAK_RV_PILOT_IMPLEMENTATION.md`의 구현이다.
로컬 원본은 `/home/chanwoo/FEAK_RV_PILOT_IMPLEMENTATION.md`, 사본은
`feak_tc_docs/paper_docs/FEAK_RV_PILOT_IMPLEMENTATION.md`다. 원고는 원격에 업로드하지 않는다.

## 모듈과 판정

| 단계 | 입력 | 출력/동작 |
|---|---|---|
| Scorer | 문항·현재 글 | 기존 Kanana의 8개 rubric |
| Planner | 문항·현재 글·8점수·8기준 설명 | target_rubric, target_span, problem, goal, must_preserve |
| Reviser | 문항·현재 글·고정된 plan | revised_text, summary_of_change |
| RV | 문항·전후 글·goal·must_preserve·diff | 네 criterion 및 decision |
| Controller | 네 criterion | ACCEPT / RETRY / STOP |

Planner의 `plan=null`은 actionable issue가 없다는 종료 응답이다. target_span은 원문에서
인용한다. Reviser는 전체 수정본을 반환하되 한 문제에 필요한 변경만 지시받는다.
문자 diff로 과도한 수정을 점검한다. 편집량으로 의미 보존을 대신 판정하지 않는다.

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

API/schema 오류는 품질 FAIL과 구분한다. 마지막 채택 글을 반환하며 status=error로 기록하고
남은 batch를 멈춘다. RV 채택 후 상태 채점만 실패하면 채택을 되돌리지 않고 final_scores=null로
종료한다. 점수 오류가 채택 여부를 바꾸지 않는다. 각 JSONL 행은 flush한다.
강제 종료 후에는 완료되지 않은 summary 대신 events/llm_calls를 확인한다.

## 검증 범위

`python -m pytest -q`는 점수와 채택의 분리, 재수정 원문 유지, 재검증 한도, 판정 이력 차단,
호출 예산, 거절 후보 저장, 오류 시 상태 보존을 검증한다.
합성 예제와 stub은 실제 held-out 평가나 사람 판정 일치도·학습 효과의 근거가 아니다.
baseline 로그는 **같은 RV 경로의 후보별 결정 비교**이며 별도 점수 기반 정책의 궤적이 아니다.
