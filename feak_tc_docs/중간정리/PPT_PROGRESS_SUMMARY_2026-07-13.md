# FEAK-TC 현재 진행 상황 요약

> 교수님 보고용 요약  
> 작성일: 2026-07-13  
> 초점: 서론/문제정의가 아니라, 현재까지 구현된 방법론과 실험적으로 확인한 사항

## 1. 한 줄 요약

현재 FEAK-TC는 **학생 글을 진단하고, 여러 수정 action 후보를 만들고, 각 후보를 실제로 적용한 뒤, 재채점 결과로 좋은 수정을 고르는 one-step MVP**까지 구현되어 있다.

다만 최근 실험 결과, 기존의 정수 점수 기반 검증은 Kanana 채점기의 반올림 노이즈에 민감하다는 문제가 확인되어, 현재는 **반올림 전 연속 점수 기반 검증 구조로 전환한 상태**다.

## 2. 현재 MVP 흐름

현재 시스템의 한 step은 다음 순서로 동작한다.

```text
입력 글
  ↓
Kanana diagnoser
  - 8개 rubric 점수
  - FEAK 언어 자질 29개
  - 약한 rubric 선택
  ↓
Action proposer
  - ADD_DETAIL
  - DELETE_OR_FOCUS
  - COMPRESS
  - RESTRUCTURE
  - STYLE_REFINE
  ↓
Patcher
  - 각 action 후보를 실제 수정문으로 적용
  ↓
Kanana re-diagnosis
  - 수정 전/후 점수와 자질 비교
  ↓
Transition feature 계산
  - target_gain
  - non_target_drop
  - target_gap_reduction
  - evidence_match
  - edit_ratio
  - goal_preservation / emb_sim
  ↓
Heuristic selector
  - 현재는 휴리스틱으로 accept/reject 결정
  - 향후 TVM이 이 역할을 대체 예정
```

즉 현재 구조는 LLM이 바로 “좋은 수정”을 고르는 방식이 아니다.  
LLM은 후보를 만들고, 시스템은 **실제 적용 후 재채점한 counterfactual 결과**를 보고 선택한다.

## 3. 구현 완료된 핵심 구성요소

| 구성요소 | 현재 상태 |
|---|---|
| Kanana diagnoser 연동 | 구현 완료 |
| FEAK feature 29개 추출 | 구현 완료 |
| train.jsonl essay 컬럼 기반 batch 실행 | 구현 완료 |
| 발문 question 반영 | 구현 완료 |
| 핵심 키워드 미사용 처리 | 구현 완료 |
| action taxonomy 기반 후보 생성 | 구현 완료 |
| LLM proposer / deterministic fallback | 구현 완료 |
| LLM patcher / deterministic fallback | 구현 완료 |
| patch validity check | 구현 완료 |
| 수정 전/후 transition feature 계산 | 구현 완료 |
| heuristic accept/reject selector | 구현 완료 |
| 실험 로그 JSONL 저장 | 구현 완료 |
| soft score 기반 gain/drop 계산 | 구현 완료 |
| scorer noise 측정 스크립트 | 구현 완료 |
| 전체 pytest / stub smoke test | 통과 |

## 4. 최근 가장 중요한 변경: soft score 전환

기존에는 Kanana의 최종 정수 rubric 점수를 사용해 다음을 계산했다.

```text
target_gain = 수정 후 정수 점수 - 수정 전 정수 점수
non_target_drop = 다른 rubric의 정수 점수 하락량
```

문제는 Kanana가 내부적으로 연속 점수를 만든 뒤 최종적으로 정수로 반올림한다는 점이다.  
그래서 실제 변화가 작아도 반올림 경계 근처에서는 `+1`처럼 보일 수 있다.

이를 해결하기 위해 현재 transition 계산은 다음 방식으로 바뀌었다.

```text
정수 rubric 점수 대신
rf_corrected_score, 즉 반올림 전 연속 점수를 사용
```

단, 안전하게 하기 위해 before/after 양쪽 모두 연속 점수가 있을 때만 soft score를 사용하고, 없으면 기존 정수 점수로 fallback한다.

## 5. scorer noise 측정 결과

Kanana를 같은 글에 반복 적용했을 때 점수가 얼마나 흔들리는지 측정했다.

### 정수 점수 기준

| kanana m | rerun max\|diff\| | 결론 |
|---:|---:|---|
| 3 | 1.0 | 정수 점수는 여전히 1점 흔들림 |
| 10 | 1.0 | 정수 점수는 여전히 1점 흔들림 |
| 20 | 1.0 | 정수 점수는 여전히 1점 흔들림 |

정수 점수만 보면 `target_gain=1`이 실제 개선인지 채점 노이즈인지 분리하기 어렵다.

### 연속 점수 기준

| kanana m | rerun max\|diff\| | rerun mean\|diff\| |
|---:|---:|---:|
| 3 | 0.685 | 0.158 |
| 10 | 0.225 | 0.080 |
| 20 | 0.281 | 0.081 |

연속 점수로 보면 noise가 더 세밀하게 관찰된다.  
특히 m=10과 m=20은 평균 noise가 약 0.08 수준으로 비슷했고, m을 20까지 올려도 큰 추가 안정화는 제한적이었다.

## 6. 20건 before/after 비교 결과

같은 20개 글에 대해 기존 정수 기반 v2와 새 soft score 기반 v3를 비교했다.

| 항목 | v2: 정수 기반 | v3: soft score 기반 |
|---|---:|---:|
| 처리 성공 글 수 | 20 | 20 |
| accept | 13 | 0 |
| reject_all | 7 | 20 |
| 후보 수 | 100 | 100 |
| soft score 사용 후보 | 0 | 100 |
| target_gain 최대값 | 3.000 | 0.780 |
| target_gain 평균 | 0.020 | -0.023 |
| non_target_drop 평균 | 0.840 | 0.315 |

가장 중요한 해석은 다음이다.

기존 정수 점수에서는 `+1`, `+2`, `+3` 개선처럼 보였던 후보들이, 연속 점수로 다시 보면 대부분 `+1.0`보다 작았다.  
따라서 현재 threshold인 `target_gain_min = 1.0`을 그대로 유지하면 soft score 기준에서는 모든 후보가 reject된다.

이것은 구현 실패라기보다, **기존 정수 점수 기반 accept가 채점기 반올림 효과를 과대평가했을 가능성을 보여주는 결과**다.

## 7. 현재 방법론적 의미

현재까지 확인된 방법론적 포인트는 세 가지다.

1. **LLM 단독 수정 방식이 아니다.**  
   LLM이 제안한 수정안을 실제로 적용하고, 채점기와 feature 변화로 검증하는 구조다.

2. **정수 점수 기반 검증은 위험하다.**  
   Kanana의 최종 정수 점수는 반올림 때문에 `+1` 변화가 실제 개선인지 noise인지 불분명하다.

3. **soft score 기반 transition이 더 적절하다.**  
   `rf_corrected_score`를 사용하면 작은 변화와 큰 변화를 더 세밀하게 구분할 수 있다.

## 8. 현재 완성도

현재 상태는 **논문용 최종 프레임워크 완성**이라기보다는, 핵심 아이디어를 검증할 수 있는 **one-step MVP가 돌아가는 단계**다.

완성된 부분:

- 글 입력부터 action 후보 생성, patch 적용, 재채점, transition 계산, accept/reject까지 end-to-end 실행 가능
- train.jsonl 기반 batch 실험 가능
- 정수 점수 noise 문제 확인
- soft score 기반 transition으로 개선
- 실험 로그와 비교 문서 생성 가능

아직 남은 부분:

- soft score 기준 threshold 결정
- TVM 학습/적용
- corruption 기반 학습 데이터 생성
- multi-step 수정 trajectory 실험
- SBERT 기반 semantic similarity 안정 적용
- 논문용 baseline 비교

## 9. 현재 한계

현재 가장 큰 한계는 세 가지다.

### 1. threshold가 아직 정해지지 않음

soft score로 바꾸면서 기존 `target_gain_min=1.0`은 너무 엄격한 기준이 되었다.  
따라서 연속 점수 기준에서 어느 정도 gain을 “실질적 개선”으로 볼지 새로 정해야 한다.

예를 들어 m=10 기준 rerun noise가 다음 정도였다.

```text
rerun max noise: 0.225
rerun mean noise: 0.080
```

따라서 향후 `target_gain_min`은 1.0이 아니라, noise보다 충분히 큰 값으로 다시 실험해야 한다.

### 2. 현재 selector는 아직 TVM이 아님

현재는 heuristic selector가 action을 고른다.  
논문에서 주장하려는 핵심은 향후 TVM이 transition을 평가하는 구조이므로, 현재 MVP는 TVM 학습 전 단계다.

### 3. semantic similarity는 아직 fallback

SBERT 기반 `goal_preservation` / `emb_sim` 구조는 넣었지만, 현재 환경에는 한국어 SBERT 모델이 로컬 캐시에 없어 token fallback으로 동작했다.  
즉 의미 보존 평가는 아직 완성된 상태가 아니다.

## 10. 다음에 해야 할 결정

현재 가장 먼저 결정해야 할 것은 다음이다.

### 1순위: soft score threshold sweep

`target_gain_min`을 여러 값으로 바꿔 보면서 accept/reject가 어떻게 변하는지 확인해야 한다.

예시 후보:

```text
target_gain_min = 0.2
target_gain_min = 0.3
target_gain_min = 0.5
```

이 실험은 threshold를 최종 확정하기 위한 것이 아니라, soft score 기준에서 시스템이 어느 정도 민감하게 작동하는지 보는 과정이다.

### 2순위: reject_all 케이스 질적 검토

현재 v3에서는 20건 모두 reject_all이므로, 실제로 좋은 수정이 있었는데 threshold 때문에 막힌 것인지 확인해야 한다.

봐야 할 것:

- 사람이 보기에도 개선된 후보가 있었는가?
- 있다면 그 후보의 soft target_gain은 얼마였는가?
- non_target_drop이나 edit_ratio는 적절했는가?

### 3순위: TVM 전환 준비

soft score threshold가 어느 정도 정리되면, heuristic selector를 TVM으로 바꾸기 위한 로그 설계를 해야 한다.

TVM에 들어갈 후보 feature:

```text
target_gain
non_target_drop
target_gap_reduction
evidence_match
edit_ratio
goal_preservation
emb_sim
action_type
target_rubric
```

## 11. PPT용 핵심 메시지

PPT에서는 다음 흐름으로 설명하면 된다.

1. 현재는 one-step action generation MVP가 end-to-end로 구현되어 있다.
2. LLM이 수정안을 만들지만, 선택은 재채점 기반 counterfactual evaluation으로 한다.
3. 실험 중 정수 rubric 점수의 noise 문제가 확인되었다.
4. 그래서 transition 계산을 정수 점수에서 `rf_corrected_score` 연속 점수로 바꿨다.
5. 그 결과 기존 accept 13건이 soft score 기준에서는 모두 threshold 미달로 바뀌었다.
6. 이는 기존 방식이 잘못됐다기보다, 정수 반올림이 개선 효과를 과대평가했을 가능성을 보여준다.
7. 따라서 다음 단계는 soft score 기준 threshold sweep과 TVM 학습 준비다.

## 12. 현재 결론

현재까지의 결론은 다음과 같다.

**FEAK-TC의 기본 파이프라인은 구현되어 작동한다.**  
하지만 좋은 action을 검증하는 기준은 정수 점수보다 연속 점수를 사용하는 것이 더 타당하다.  
soft score 전환 후 기존 threshold가 너무 엄격하다는 것이 확인되었으므로, 다음 단계는 threshold를 새 점수 체계에 맞게 실험적으로 정하는 것이다.

