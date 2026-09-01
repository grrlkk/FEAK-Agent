# FEAK-TC 평가 구조 정리

## 1. 핵심 아이디어

FEAK-TC에서는 하나의 평가기로 모든 것을 판단하지 않는다.

글 수정 Agent에서는 서로 다른 수준의 판단이 필요하기 때문에 평가를 다음 세 단계로 나눈다.

```text
State-level Evaluation
→ FEAK

Transition-level Evaluation
→ Revision Verifier

Trajectory-level Evaluation
→ Trajectory Guard
```

쉽게 말하면:

```text
FEAK
= "현재 글이 어떤 상태인가?"

Revision Verifier
= "이번 수정이 제대로 수행됐는가?"

Trajectory Guard
= "여러 번 수정한 전체 경로가 괜찮은가?"
```

---

## 2. FEAK — State Evaluator

FEAK는 현재 글의 상태를 평가하는 Diagnoser이다.

FEAK의 역할은 다음과 같다.

- 현재 글의 8개 rubric 점수 산출
- 29개 linguistic feature 분석
- elite benchmark와의 gap 분석
- 현재 글에서 어떤 부분이 부족한지 진단

즉 FEAK가 답하는 질문은:

> **"현재 글의 품질 상태는 어떠한가?"**

이다.

중요한 점은 FEAK가 잘못된 평가기라서 다른 모델이 필요한 것이 아니다.

FEAK는 **State Quality를 측정하는 역할**을 수행하며, Agent가 실제로 수행한 수정이 적절했는지는 별도의 문제다.

---

## 3. Planner — 어떤 수정을 시도할 것인가?

Planner는 FEAK의 진단 결과를 바탕으로 수정 방향을 제안한다.

예:

```text
FEAK
→ Inter-paragraph Structure가 약함

Planner
→ RESTRUCTURE
→ DELETE_OR_FOCUS
```

Planner가 답하는 질문은:

> **"현재 문제를 해결하기 위해 어떤 action을 시도할까?"**

이다.

Planner는 수정 결과가 좋은지 평가하지 않는다.

---

## 4. Candidate Generator — 실제 수정 결과 생성

Planner가 선택한 action을 실제 글에 적용한다.

예:

```text
Current State

Planner
→ RESTRUCTURE

↓

Generator

Candidate A
→ 문단 순서를 적절하게 변경

Candidate B
→ 문단 순서를 바꿨지만 다른 내용도 과도하게 수정
```

같은 action을 사용하더라도 실제 수정 결과는 크게 달라질 수 있다.

따라서 action 선택과 수정 결과 평가는 구분해야 한다.

---

## 5. Revision Verifier — Transition Evaluator

기존의 TRM(Transition Reward Model)은 역할이 다소 모호해질 수 있기 때문에, 현재 구조에서는 **Revision Verifier**로 재정의하는 것이 더 자연스럽다.

Revision Verifier가 답하는 질문은:

> **"의도했던 수정이 실제로 적절하게 수행됐는가?"**

이다.

즉 수정 전과 수정 후를 함께 본다.

```text
Before State
+ Target Rubric
+ Intended Action
+ Revised Text

↓

Revision Verifier
```

---

## 6. Revision Verifier가 확인하는 항목

### 6.1 Target Fulfillment

의도했던 문제를 실제로 해결했는가?

### 6.2 Content Preservation

수정할 필요가 없었던 핵심 내용이나 원래 주장을 보존했는가?

### 6.3 Edit Minimality

필요 이상으로 많은 부분을 수정하지 않았는가?

### 6.4 Action Consistency

선택한 action과 실제 수행된 수정이 일치하는가?

예:

```text
Action:
RESTRUCTURE

실제 수정:
새로운 사례와 내용을 대량 추가

→ Action mismatch
```

---

## 7. FEAK와 Revision Verifier의 차이

둘은 서로 경쟁하는 평가기가 아니다.

```text
FEAK
= 수정된 글 자체가 얼마나 좋은 상태인가?

Revision Verifier
= 그 상태로 이동한 수정 과정이 적절했는가?
```

예를 들어 Candidate B의 최종 FEAK 점수가 높더라도, Revision Verifier는 의미 보존 실패나 과수정을 감지할 수 있다.

이것은 FEAK가 틀린 것이 아니다.

FEAK와 Revision Verifier가 **서로 다른 문제를 평가하는 것**이다.

---

## 8. FEAK Gain의 역할

FEAK Gain도 그대로 사용할 수 있다.

```text
FEAK Gain
= 수정 후 State가 얼마나 개선되었는가

Revision Verifier
= 그 개선을 만드는 수정 과정이 적절했는가
```

예:

```text
Candidate A

FEAK Gain:
+1.5

Revision Verifier:
Target Fulfillment O
Preservation O
Minimality O

→ 좋은 수정
```

반대로:

```text
Candidate B

FEAK Gain:
+2.0

Revision Verifier:
Target Fulfillment O
Preservation X
Minimality X

→ 높은 점수 상승이 있어도 부적절한 revision
```

따라서 FEAK Gain을 없애는 것이 아니라, **State improvement signal**로 활용한다.

---

## 9. Trajectory Guard — Global Evaluator

Revision Verifier가 한 번의 수정만 평가한다면, Trajectory Guard는 전체 수정 경로를 평가한다.

```text
s0 → s1 → s2 → s3 → ...
```

Trajectory Guard가 답하는 질문은:

> **"각 수정은 괜찮았지만, 여러 수정이 누적되면서 전체 글이 나빠지고 있지는 않은가?"**

확인할 수 있는 요소:

- 원래 Writing Goal 유지
- 이미 개선된 rubric의 재악화 여부
- non-target rubric의 누적 하락
- 핵심 주장과 내용의 장기적 보존
- 반복적인 추가에 따른 과도한 장문화
- 이전에 해결한 문제가 다시 나타나는지 여부

---

## 10. 세 평가 수준의 최종 관계

```text
                         Candidate
                             │
          ┌──────────────────┼──────────────────┐
          ▼                  ▼                  ▼

        FEAK          Revision Verifier    Trajectory Guard

    State Quality       Transition           Trajectory
       Change             Validity              Safety

    "좋아졌나?"          "잘 고쳤나?"          "계속 가도 되나?"

          │                  │                  │
          └──────────────────┼──────────────────┘
                             ▼
                         Controller
```

Controller는 이 세 신호를 종합한다.

---

## 11. Revision Verifier 학습 데이터 생성

Progressive Corruption을 Revision Verifier의 supervision 생성에 활용할 수 있다.

먼저 고품질 글을 준비한다.

```text
x0
= High-quality Essay
```

특정 문제를 의도적으로 만든다.

```text
x0
↓ Inter-paragraph corruption
x1
```

이때 우리는 다음 정보를 알고 있다.

```text
Corruption Type
Target Rubric
Changed Span
Original Content
```

이 `x1`을 현재 State로 보고 여러 revision candidate를 만든다.

```text
x1
│
├─ A. 정확한 복구
├─ B. 부분적인 복구
├─ C. target은 고치지 않고 다른 부분만 수정
├─ D. target은 복구하지만 과도하게 전체를 수정
└─ E. 오히려 더 악화
```

---

## 12. Revision Verification Label

각 candidate에 대해 다음 기준으로 label을 만들 수 있다.

| Candidate | Target Fulfillment | Preservation | Minimality | Action Consistency |
|---|---|---|---|---|
| A. 정확한 복구 | O | O | O | O |
| B. 부분 복구 | △ | O | O | O |
| C. 잘못된 부분 수정 | X | O | △ | X |
| D. 과도한 수정 | O | △/X | X | △ |
| E. 추가 악화 | X | X | X | X |

즉 단순히:

```text
A > B > C
```

라는 preference만 만드는 것보다,

```text
Target Fulfillment
Preservation
Minimality
Action Consistency
```

이라는 **revision-specific supervision**을 만들 수 있다.

---

## 13. Synthetic Pretraining + Real Calibration

### Stage 1. Synthetic Pretraining

```text
High-quality Essay
        ↓
Known Corruption
        ↓
Corrupted State
        ↓
Revision Candidates
        ↓
Revision Verification Labels
        ↓
Revision Verifier Pretraining
```

### Stage 2. Real Revision Calibration

실제 LLM이 생성한 수정은 synthetic 데이터보다 복잡하다.

따라서 실제 글에서 candidate를 생성한다.

```text
Current Essay
        ↓
Planner
        ↓
LLM Revision Candidates
```

그리고 일부 candidate를 사람이 평가한다.

예:

```text
Target Fulfillment
Preservation
Minimality
Action Consistency
Overall Revision Success
```

이 human annotation을 사용해 Revision Verifier를 calibration한다.

---

## 14. 전체 Agent에서의 역할

```text
FEAK
"지금 글에서 무엇이 부족한가?"

        ↓

Planner
"어떤 수정 방법을 시도할까?"

        ↓

Exemplar Retrieval
"잘 쓴 글에서는 어떻게 했는가?"

        ↓

Candidate Generator
"실제로 수정해보자."

        ↓

┌────────────────────────────────────────┐
│ Evaluation                             │
│                                        │
│ FEAK                                   │
│ → 수정 후 State가 좋아졌는가?         │
│                                        │
│ Revision Verifier                      │
│ → 이번 수정이 적절하게 수행됐는가?    │
│                                        │
│ Trajectory Guard                       │
│ → 전체 수정 경로가 안전한가?          │
└────────────────────────────────────────┘

        ↓

Controller

Accept
Reject
Rollback
Stop

        ↓

Agent State Update

        ↺
```

---

## 15. 핵심 메시지

FEAK-TC에서는 평가를 하나의 점수로 해결하지 않는다.

글쓰기 Agent가 안정적으로 반복 수정하려면 다음 세 가지가 서로 다른 수준에서 필요하다.

```text
FEAK
→ State-level Quality

Revision Verifier
→ Transition-level Validity

Trajectory Guard
→ Trajectory-level Safety
```

즉,

> **글의 상태가 좋아졌는지 평가하는 것과, 그 상태로 이동한 수정 과정이 적절했는지 평가하는 것, 그리고 그러한 수정이 누적된 전체 경로가 안전한지를 평가하는 것은 서로 다른 문제이다.**

이 세 수준의 평가를 결합하여 FEAK-TC는 반복적인 Open-ended Writing revision을 제어한다.
