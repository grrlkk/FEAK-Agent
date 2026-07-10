# 중간정리 — Action 생성 방식 재설계 구상

> 2026-07-06. propose 단계의 action 생성 방식을 어떻게 바꿀지에 대한 문헌 조사 + 설계 제안.
> 관련 코드: `feak_tc/mvp/propose.py`, `feak_tc/mvp/targeting.py`
> 관련 문서: `docs/PROJECT_CONTEXT.md`(확정 결정), `docs/SPEC_CORRUPTION_TVM.md`(corruption 설계)

## 0. 현재 방식 요약

LLM이 action을 "선택"하지 않는다. 프롬프트에서 **6개 action_type 전부에 대해 각각
후보를 강제 생성**시키고(action-stratified 열거), 6개 후보를 전부 patch → 재진단 →
transition 실측한 뒤 휴리스틱(나중엔 TVM)이 고른다.

- LLM 역할: action이 주어졌을 때의 구체화 (target_rubric, target_span, instruction)
- 선택 주체: 사후 counterfactual 비교 (휴리스틱 → TVM)
- LLM이 action을 빼먹으면 deterministic proposer가 슬롯을 채워 6종 슬레이트 유지
- 설계 의도 반영: "add 편향 방지" + "policy 학습 안 함, 가치평가기가 선택"

현재 구현의 세부 이슈 3가지:

1. 6개 action을 **한 번의 API 호출**로 생성 → 후보 간 오염 가능 (한 action의 아이디어가
   다른 후보에 영향)
2. LLM 프롬프트에 action의 **한국어 정의가 안 들어감** (`_ACTION_INSTRUCTIONS`는
   deterministic 모드 전용, LLM은 영어 enum 이름만 보고 자체 해석)
3. `_RUBRIC_TO_ACTION_HINT`가 rubric→action 1:1 하드코딩 (학습된 prior의 수동 스텁)

## 1. 다른 연구들의 Action 생성 방식 4계열

> 컷오프 2026-01 기준 기억 인용. 수치·세부는 원문 확인 필요.

### 계열 1 — 자유 생성 피드백이 곧 action (taxonomy 없음)

- Self-Refine (Madaan et al., 2023), Reflexion (Shinn et al., 2023)
- LLM이 자기 글에 자유 텍스트 피드백 → 그걸로 수정 → 반복
- 실패 보고 다수: 자기평가 신뢰 불가, 과수정 ("LLMs Cannot Self-Correct Reasoning
  Yet", Huang et al., 2023)
- **FEAK-TC가 taxonomy를 도입한 이유가 이 계열의 실패.** 돌아갈 이유 없음.
  대신 좋은 baseline (로드맵의 "LLM self-refine" baseline과 일치).

### 계열 2 — 고정 edit-intention taxonomy + 학습된 intent 분류기 ★ FEAK-TC와 가장 가까움

- IteraTeR / R3 (Du et al., ACL 2022): 사람 퇴고 코퍼스에 CLARITY/FLUENCY/
  COHERENCE/STYLE/MEANING-CHANGED 5종 intention 주석
- 파이프라인: (1) "어디를, 어떤 intention으로"를 span 단위 예측하는 분류기 →
  (2) 그 intention을 실행하는 수정 생성기
- 핵심 차이: action을 열거하지 않고 **사람 수정 데이터로 학습한 분류기가 먼저 고름**
- "Learning Where to Edit" 계열 (EMNLP 2022)도 같은 구조
- FEAK-TC가 이걸 못 하는 유일한 이유 = 한국어 수정 전후 쌍 데이터 없음
  → **corruption이 이 구멍을 메운다** (아래 Stage B)

### 계열 3 — 실제 편집 이력에서 학습한 plan/instruction-as-action

- PEER (Schick et al., 2022): Wikipedia 편집 이력 학습, "plan(자연어 action) →
  edit → 설명" 생성. **역방향 infilling으로 데이터 증강 = corruption 역연산과 동일 발상**
  → 방법론 서술 시 인용 근거로 유용
- CoEdIT (Raheja et al., 2023): 편집 instruction 카테고리별 fine-tune 실행기

### 계열 4 — propose-then-select (제안은 넓게, 선택은 가치함수) ★ FEAK-TC의 현재 구조

- Tree of Thoughts (Yao et al., 2023), LATS (Zhou et al., 2024), RAP (Hao et al.,
  2023), best-of-N + reward model 일반 관행
- process reward model (Lightman et al., 2023)의 step 단위 가치 평가는
  FEAK-TC의 transition 단위 TVM과 개념적으로 동형
- FEAK-TC와의 차이: 후보 분포를 샘플링이 아니라 **taxonomy 전수 열거**로 만든다는 점뿐

## 2. 제안: 2단계 전환 설계

### Stage A — 지금: 열거 유지 + 생성 품질 수리

전 action 열거를 당장 버리지 않는다. **지금 모으는 로그가 TVM 학습 데이터(②번 소스)인데,
action 선택을 지금 똑똑하게 만들수록 로그의 action 분포가 편향된다.** offline RL의
균등 탐색 정책과 같은 이유로, 데이터 수집 단계의 열거는 결함이 아니라 기능.

수리할 것:

1. **action별 독립 LLM 호출로 분리** (최소한 프롬프트에 action별 한국어 정의 주입).
   gpt-4o-mini 5회 호출 비용은 kanana 재진단 6회 대비 무시 가능.
2. **`n_per_action` 2~3 + temperature**로 action 내 다양성 확보.
   → 같은 action의 좋은 실행 vs 나쁜 실행 쌍이 로그에 생김
   = corruption 스펙의 hard negative("무엇을 ADD했는지가 중요")를 실데이터에서 공짜로 획득
3. **STOP을 생성 후보에서 제거, controller 결정으로 일원화 검토.**
   문헌에서 종료는 가치 임계값(전 후보 threshold 미만 → stop)으로 처리하지
   "STOP 후보 생성"이 아님. 현재는 STOP 후보와 accept_threshold stop이 공존해
   판단 경로가 이중. taxonomy에는 남기되(transition 라벨용) 생성·선택에서는 제외 권장.

### Stage B — corruption 확보 후: 학습된 action prior로 가지치기

**corruption 역연산 라벨 = R3 intent 분류기의 학습 데이터를 공짜로 생성.**
corruption 체인의 각 단계는 "(약점 상태의 진단 결과) → (그걸 고치는 정답 action)" 쌍이므로,
rubric 8 + feature 29 벡터 입력으로 P(action | diagnosis)를 예측하는 가벼운 분류기
(LightGBM — TVM GBM baseline과 인프라 공유)를 사람 라벨 없이 학습 가능.

추론 시:

```
진단 → action prior가 상위 2~3개 action만 선택
     → 그 action들만 후보 생성·patch·재진단
     → TVM이 최종 선택
```

- 재진단(kanana 8B)이 병목이므로 6→2~3 가지치기 = 스텝당 비용 절반 이하
- 현재의 `_RUBRIC_TO_ACTION_HINT` 하드코딩이 이 prior의 수동 스텁 → 학습 분포로 교체

**⚠ 미결 사항 (사용자 판단 필요):** 확정 결정 "policy 학습 안 함"과의 관계.
prior는 action을 최종 결정하지 않고 후보 집합만 줄이는 pruner이며 최종 선택권은 TVM에
있으므로 결정 위반이 아니라고 서술 가능("value-guided control with learned proposal
prior"). 단, 논문 포지셔닝에 걸리는 부분이라 확정 필요.

### Stage C — future work: 다단계 탐색

TVM 확보 후 one-step 선택은 beam/greedy trajectory 탐색(LATS류)으로 자연스럽게
일반화. 로드맵대로 이번 범위 밖.

## 3. 논문 서술 관점

비교 축이 깔끔해진다:

| 방식 | action 결정 | 라벨 필요 |
|---|---|---|
| Self-Refine | 자유 생성 + 자기평가 | 없음 (신뢰 불가) |
| R3 | 학습된 intent 분류기 | 사람 수정 코퍼스 필요 |
| **FEAK-TC** | corruption 합성 라벨로 prior 학습 + transition 가치 평가 | **사람 라벨 불요** |

- 기여 포인트: "사람 수정 코퍼스 없이 R3식 action 선택을 달성"
- Stage A→B 전환 자체가 ablation (전수 열거 vs prior 가지치기의 품질/비용 트레이드오프)

## 4. 다음 액션

- [ ] Stage A-1: propose를 action별 독립 호출로 분리 + 한국어 정의 프롬프트 주입
- [ ] Stage A-2: n_per_action 2~3 + temperature 다양성
- [ ] Stage A-3: STOP 생성 제거 여부 결정 (사용자 확인)
- [ ] Stage B는 corruption 단계 진입 후 (SPEC_CORRUPTION_TVM.md와 함께 진행)
- [ ] "policy 학습 안 함" 결정과 action prior의 관계 포지셔닝 확정 (사용자 확인)
