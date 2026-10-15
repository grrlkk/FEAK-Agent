# RV 파일럿 실행 검증 — 2026-09-28

현재 Kanana 채점기와 GPT-5-mini(reasoning low)를 사용했다. 입력은 직접 작성한 합성 예제 5편이며, held-out 품질 평가가 아니다.

## 자동 검증

- 전체 테스트: 247 passed (기존 SWIG 경고 2개).
- feak_tc의 59개 모듈 import 성공.
- 모델 없는 5편 실행 및 보관한 이전 CLI 실행 성공.
- 실제 실행의 RV 입력 allowlist, 전부 PASS인 후보만 채택, 거절 시 원문 유지, 재시도 시 같은 plan·원문 유지, 실행 중 소스 hash 일치 확인.

## 실제 5편 실행

| 항목 | 결과 |
|---|---|
| samples | 5 |
| completed | 5 |
| candidate_attempts | 16 |
| accepted | 15 |
| rejected | 1 |
| revision_retries | 1 |
| reverifications | 1 |
| aggregate_score_up_but_rv_rejected | 0 |
| api_calls | 48 |

| sample_id | 종료 | 채택 수 |
|---|---|---|
| synthetic_phone | max_iterations | 3 |
| synthetic_library | max_iterations | 3 |
| synthetic_lunch | max_iterations | 3 |
| synthetic_group | max_iterations | 3 |
| synthetic_park | max_iterations | 3 |

사용량: 입력 48,176, 출력 28,063 tokens.
공식 표준 요금과 기록된 사용량으로 추산한 이 5편의 API 비용은 약 $0.0682, 초기 schema 오류 확인 및 아래 대조 사례까지 합한 검증 비용은 약 $0.0770다. 실제 청구서 확인값은 아니다.
[단가 근거: GPT-5 Mini](https://developers.openai.com/api/docs/models/gpt-5-mini) (입력 $0.25, cached 입력 $0.025, 출력 $2 / 1M tokens; 확인 2026-09-28).

네 번째 글에서 불확실 판정 후 같은 허용 입력으로 fresh 재검증이 수행됐다. 또 다른 후보는 새 중복을 만든다는 global_benefit FAIL로 거절됐고, 그 이유를 받은 재수정이 채택됐다.
점수가 하락했어도 RV가 모두 PASS로 판정한 후보가 채택되어 점수 선택과 분리됨을 확인했다.
전체 합계 점수가 상승했으나 RV가 거절한 자연 발생 후보는 이번 실행에서 0건이었다. 이를 얻기 위해 후보를 추가 탐색하거나 결과를 재라벨링하지 않았다.

## 통제한 RV 사례

- 중복 표현만 줄이는 최소 수정: ACCEPT, preservation PASS.
- 같은 수정에 작성자의 주장을 반대로 바꾸는 변경 추가: REJECT, preservation FAIL.
- 두 사례는 자동 판정의 간단한 실행 확인이며 판정 정확도 추정이 아니다.

## 보관 위치와 실행 중 수정

- 전체 실행: experiments/results/rv_pilot_gpt_live_v2_20260928/
- 통제 사례: experiments/results/rv_live_probes_live_20260928.json
- 모델 없는 실행: experiments/results/rv_pilot_offline_20260928/
- 초기에 GPT가 rubric ID에 한국어 이름을 덧붙여 validation에 실패했다. 8개 ID를 JSON schema enum으로 제한하고 회귀 테스트를 추가한 뒤 전체 실행을 다시 완료했다.
- 초기 실패 기록도 보존했다. 논문 원고·원시 출력·키·모델은 커밋하지 않았다.
- 기존 웹은 과거 2축 루프를 실행한다. 새 방법론은 scripts/run_pilot.py로 실행한다.
