# RV 파일럿 3-LLM 블라인드 전수 검증

날짜: 2026-09-01

## 결론

Revision Verifier 데이터의 **구성 방식과 schema는 사용할 수 있지만, 현재 weak label을 그대로
학습에 사용하면 안 된다.** LLM 생성 후보 중 `wrong_target`의 의도 재현율이 낮고,
candidate type마다 4축 label을 고정하는 방식은 실제 후보별 차이를 반영하지 못했다.

현재 판단은 **RV 학습 보류(no-go)** 이다. 후보 생성과 instance-level label을 고친 뒤 같은
블라인드 검증을 다시 통과시켜야 한다.

## 검증 범위

- 대상: `rv_data_pilot_50.jsonl`의 50개 state 전부
- 검증 후보: LLM이 생성한 `wrong_target` 50개와 `over_edit` 50개
- 총 판정: 100개 후보 x 3개 모델 = 300개 독립 판정
- 모델:
  - `gpt-5-2025-08-07`
  - `gpt-5-mini-2025-08-07`
  - `gpt-4.1-2025-04-14`
- 층화: 4개 corruption type x stage 1/2 전체
- 공개 정보: 질문, 현재 state, 정확한 복원 참고문, target rubric, intended action,
  corruption edit, 무작위 A/B 후보
- 숨긴 정보: 실제 candidate type, 기존 4축 weak label, 다른 모델의 판정

`correct_repair`, `further_corruption`, `no_edit`는 trajectory provenance로, `partial_repair`는
deterministic corruption-edit replay로 후보 생성 자체를 이미 검증했으므로 이번 LLM 판정에서는
제외했다. 따라서 이번 결과는
300행 전체 label의 최종 승인이 아니라, 불확실성이 가장 큰 LLM 생성 후보 100행의 전수 검증이다.

## 모델별 결과

| 모델 | 전체 타입 일치 | wrong_target | over_edit | 4축 전부 일치 | usable 판정 |
|---|---:|---:|---:|---:|---:|
| GPT-5 | 48% | 12% | 84% | 12% | 99% |
| GPT-5 mini | 59% | 72% | 46% | 0% | 68% |
| GPT-4.1 | 50% | 24% | 76% | 6% | 32% |

GPT-5 mini는 해당 후보를 생성한 모델과 동일하다. 이 모델만 `wrong_target` 일치율이 유독
높으므로 생성 의도를 다시 인식하는 편향 가능성이 있다. 후보 생성에 참여하지 않은 GPT-5와
GPT-4.1만 비교하면 다음과 같다.

- `wrong_target`: 두 모델이 모두 intended type으로 판정한 후보 2/50, 둘 다 `other` 30/50
- `over_edit`: 두 모델이 모두 intended type으로 판정한 후보 37/50, 둘 다 `other` 7/50

세 모델 모두 같은 OpenAI provider이므로 이 결과는 사람 평가나 provider-independent 검증을
대체하지 않는 LLM proxy이다.

## 다수결 결과

### Candidate type

- 전체 intended type 일치: 53/100 (2개는 세 모델이 모두 달라 다수결 불가)
- `wrong_target`: 14/50 (28%)
- `over_edit`: 39/50 (78%)

`wrong_target` 50개 중 33개는 다수결로 `other`, 1개는 `over_edit`, 2개는 다수결 불가였다.
대표 실패는 target 밖의 문제를 실제로 고친 후보가 아니라, 오염을 그대로 두고 표현 몇 개만
바꾸거나 target을 불완전하게 고친 후보였다.

### 4축 weak label

| 축 | 기존 label과 다수결 일치 | 전체 후보 기준 일치 개수 |
|---|---:|---:|
| target_fulfillment | 77.3% (판정 가능 97개) | 75/100 |
| preservation | 78.2% (판정 가능 87개) | 68/100 |
| edit_appropriateness | 23.3% (판정 가능 86개) | 20/100 |
| action_consistency | 54.7% (판정 가능 95개) | 52/100 |
| 네 축 모두 | 6.9% (판정 가능 72개) | 5/100 |

가장 큰 문제는 `edit_appropriateness`이다. 현재 mapping은 모든 `wrong_target`을 `partial`, 모든
`over_edit`를 `fail`로 고정하지만 실제 다수결은 다음처럼 갈렸다.

- `wrong_target`: fail 28, partial 10, pass 10, undecided 2
- `over_edit`: fail 10, partial 17, pass 11, undecided 12

즉 candidate type은 4축 label의 정답이 아니다. **각 후보 text를 보고 축별로 따로 label해야 한다.**

## 평가자 일치도

| 항목 | Fleiss' kappa | 3자 완전 일치 |
|---|---:|---:|
| intended candidate type | 0.351 | 40% |
| target_fulfillment | 0.637 | 68% |
| preservation | 0.148 | 32% |
| edit_appropriateness | 0.224 | 32% |
| action_consistency | 0.434 | 46% |
| usable_for_weak_supervision | -0.060 | 29% |

100개 중 93개는 여섯 판정 항목 중 하나 이상에서 모델 간 불일치가 있었다. 특히 usable 판정은
GPT-5 99%, GPT-4.1 32%로 calibration 차이가 너무 커서 현재 형태로 gate에 쓰면 안 된다.

## Corruption별 차이

`wrong_target`는 모든 구간에서 낮았다. stage별 성공 수는 0~3개 수준이었다.

`over_edit`는 action에 따라 결과가 명확히 갈렸다.

- `DELETE_SPECIFICS` / `ADD_DETAIL`: 3/13
- 나머지 `INJECT_LEX_REPEAT`, `INSERT_OFFTOPIC`, `SHUFFLE_FLOW`: 36/37

ADD_DETAIL에서는 내용을 더 많이 추가하는 것만으로 over-edit인지 정상적인 detail 보강인지
구분하기 어렵다. 현재의 공통 생성 prompt로는 명확한 negative를 만들지 못한다.

## 진단

1. `wrong_target` 생성 조건이 약하다. target을 해결하지 않는 것만으로는 부족하며, 다른 rubric의
   문제를 실제로 찾아 의미 있는 수정을 성공시켜야 한다.
2. `over_edit`는 operator-conditioned 생성이 필요하다. 특히 ADD_DETAIL은 정상 보강과 과수정의
   경계를 별도로 정의해야 한다.
3. candidate type 기반 고정 label mapping은 폐기해야 한다. 같은 타입 안에서도 4축 결과가 크게
   다르다.
4. LLM judge끼리도 preservation/minimality calibration이 낮다. 명시적 anchor 예시와 소규모 사람
   평가로 rubric을 먼저 보정해야 한다.

## 다음 작업

1. `wrong_target`는 명시적인 non-target rubric/action을 먼저 선택하고, 그 문제를 실제로 개선한
   edit가 없거나 near-no-op이면 생성 단계에서 reject한다.
2. `over_edit`는 correct repair를 시작점으로 target 밖의 의미 손실, 불필요한 재구성, 근거 없는
   추가 중 하나를 의도적으로 결합한다. ADD_DETAIL 전용 규칙을 별도로 둔다.
3. `candidate_type`은 생성 provenance로만 저장하고, 4축 label은 instance별 판정으로 만든다.
4. 2/3 모델이 intended type에 동의하지 않는 synthetic candidate는 폐기하거나 재생성한다.
5. 수정 후 50-state 전수 블라인드 검증을 다시 실행하고, 이후 나머지 4개 trajectory 후보의
   instance-level label도 별도로 audit한다.

## 재현 파일

- 설정: `configs/rv_llm_judge.yaml`
- 실행: `python scripts/evaluate_rv_pilot_llm_judges.py`
- 공개 packet: `experiments/results/rv_data_pilot_50_three_llm_judge_public.jsonl`
- 숨은 key: `experiments/results/rv_data_pilot_50_three_llm_judge_hidden_key.jsonl`
- 모델별 원응답: 같은 prefix의 `_gpt5.jsonl`, `_gpt5_mini.jsonl`, `_gpt41.jsonl`
- 통계: `experiments/results/rv_data_pilot_50_three_llm_judge_report.json`
- 불일치: `experiments/results/rv_data_pilot_50_three_llm_judge_disagreements.jsonl`

평가 스크립트는 모델별 JSONL checkpoint를 재사용한다. 같은 설정으로 재실행하면 API를 다시
호출하지 않고 report를 재생성한다.
