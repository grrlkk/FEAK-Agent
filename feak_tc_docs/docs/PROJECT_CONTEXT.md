# PROJECT_CONTEXT — FEAK-TC 프로젝트 전체 맥락

> CLI 에이전트가 이 프로젝트의 "왜"와 "무엇"을 이해하기 위한 문서.
> 구현 지시는 IMPLEMENTATION_MVP.md, 데이터/학습 스펙은 SPEC_CORRUPTION_TVM.md 참조.

## 1. 연구 배경 (짧게)

- 선행 연구 FEAK: 한국어 글을 한 번 진단하고 피드백하는 파이프라인 (SAC'26 게재).
- 본 연구 FEAK-TC: 이를 **반복 수정을 제어하는 agent**로 확장.
- 핵심 관점: 평가 단위를 최종 글이 아니라 **수정 transition(s→s′)** 으로 옮긴다.
- 핵심 질문: "무엇을 고치고(B1), 그 수정이 좋았는지(B2), 경로가 틀어지지 않았는지(B3)"를
  하나의 closed loop로 통합.

## 2. 시스템 구성 (확정)

```
글 입력
 → ① 진단: 채점기(kanana-8B+LoRA, 학습완료) → rubric 8개
           + 독립 룰베이스 → feature 29개
 → ② 후보 생성: LLM(GPT-5 mini)이 action별 수정안
    action = ADD_DETAIL / DELETE_OR_FOCUS / COMPRESS / RESTRUCTURE / STYLE_REFINE / STOP
 → ③ 평가: TVM이 각 후보 transition의 가치 채점  ← 유일한 신규 학습 모듈
 → ④ 제어: accept / reject(국소) / rollback(전역 drift) / stop
 → 수렴까지 반복
```

### 두 층의 판단 (중요)
- **국소 = TVM**: "이 수정 하나가 좋은가" → accept/reject
- **전역 = Drift 추적**: "경로 전체가 원본 의도에서 틀어졌나" (임베딩 유사도 누적) → rollback 전용
- reject(미시, 적용 전 폐기)와 rollback(거시, 채택된 경로 되돌림)은 다른 층의 판단.

## 3. 데이터 (보유)

- AI-Hub 한국어 논술 평가 데이터 약 4만 건.
- 한 건 = 학생 글 + 8 rubric별 **사람 채점자 2인 점수** + rubric별 전문가 피드백 텍스트
        + morpheme feature + 채점기준.
- 주의: 이 데이터는 "한 글에 대한 평가"이며 **수정 전후 쌍이 아니다.**
  → TVM 학습쌍은 corruption으로 합성한다 (SPEC_CORRUPTION_TVM.md).

## 4. 확정된 설계 결정 (변경 금지, 변경하려면 사용자 승인)

| 결정 | 내용 | 이유 |
|---|---|---|
| 채점기 | kanana-8B+LoRA, 글→rubric만. **자질을 입력에 넣지 않음** | 성능 향상 미미 + 자질 독립성이 순환 방지 근거 |
| 자질 | 채점기와 독립적으로 룰베이스 계산 | rubric 신호와 feature 신호를 독립 신호로 활용 |
| TVM 구조 | **주 모델 Qwen2.5-7B-Instruct** + LoRA + scalar head, **대조군 Kanana-8B** + 별도 LoRA/head | 채점기와 같은 backbone 효과를 분리하고 주류 reward model 구조 유지 |
| TVM 입력 | 수정 전/후 **텍스트** + transition feature (텍스트 프롬프트에 포함) | 수치만으론 수정의 미묘함 못 잡음 |
| TVM loss | pairwise Bradley-Terry (+corruption 단계차를 margin/confidence로) | RM 표준 + corruption의 공짜 이점 |
| TVM 학습 데이터 | FEAK-guided corruption (action 역연산) + 실제 LLM 수정 소량 + 사람 보정쌍 소량 | 라벨 없음 문제 해결, 순환 방지(정답 근거=사람 점수) |
| policy 학습 | 하지 않음 (future work). TVM argmax로 action 선택 유도 | 일정/리스크 |
| tool-use | 넣지 않음. "value-guided control agent"로 포지셔닝 | agent 코스프레 방지 |
| drift 측정 | 고정 임베딩(한국어 SBERT 계열) 유사도, 학습 안 함 | goal_preservation + 전역 drift 공용 |

TVM 모델 결정은 2026-08-21 사용자 승인으로 기존 Kanana 단일 본선에서
cross-backbone 주 모델 + matched-backbone 대조군으로 바뀌었다. Kanana TVM의 adapter/head는
채점기 adapter와 공유하지 않지만, 같은 base backbone 자체가 주는 상관을 확인하기 위해서다.
또한 각 backbone에서 `full`과 Kanana 채점 출력(`target_gain`, `non_target_drop`)을 입력에서
제외한 `scorer_free`를 함께 학습한다. 단, `scorer_free`도 corruption 채택·필터 단계가
Kanana 측정에 의존하므로 완전한 독립 사람 라벨로 해석하지 않는다.

## 5. Transition Feature (TVM 입력, 고정)

```
action_type          무슨 수정 (6종)
target_rubric        노린 rubric
target_gain          target rubric 점수 변화 (채점기)
target_gap_reduction target feature가 elite 기준에 가까워진 정도 (자질, 독립)
non_target_drop      비target rubric 최대 하락 (채점기)
evidence_match       수정 위치가 진단된 약점과 일치하는가 (0~1)
edit_ratio           변경 토큰 비율
goal_preservation    원문 의도 보존 (고정 임베딩 유사도)
emb_sim              수정 전후 의미 유사도 (고정 임베딩)
```
- rubric 계열(채점기)과 feature 계열(독립)을 둘 다 두는 이유: 서로 다른 것을 보며,
  둘의 불일치가 hard negative(점수만 오르고 실질 훼손) 판별에 유용.

## 6. 용어 구분 (혼동 주의)

- **rubric (8개)**: 채점기가 매긴 점수. task/content/organization/expression 계열.
- **feature/자질 (29개)**: 룰베이스로 계산한 언어학 측정값 (국어교육 전문가 정의).
- **transition feature (9개)**: 수정 전후를 비교한 값. TVM 입력. 위 둘의 "변화"로 계산.

## 7. 로드맵

```
[완료] 채점기 학습 / 자질 계산기
[지금] MVP 루프 (학습 없이 한 바퀴)          → IMPLEMENTATION_MVP.md
[다음] corruption 데이터 생성                → SPEC_CORRUPTION_TVM.md
[다음] TVM 학습 (GBM baseline → kanana 본학습)
[다음] drift 추적 + rollback 통합
[다음] 평가 (baseline: FEAK 1회 / 반복 FEAK / LLM self-refine / 휴리스틱 / TVM agent)
```

## 8. 미검증 전제 (실험으로 확인할 것 — 코드가 이를 확인 가능하게 로그 남길 것)

1. 휴리스틱(즉시 점수)만으로 고르면 과수정·회귀 후보를 자주 고르는가 → TVM 필요성
2. corruption 선호쌍이 사람 선호와 일치하는가 → 라벨 타당성
3. corruption으로 학습한 TVM이 held-out 글에 일반화되는가
