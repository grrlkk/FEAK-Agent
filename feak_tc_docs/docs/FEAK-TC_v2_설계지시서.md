# FEAK-TC v2 설계 지시서 (CLI 구현용)

> 이 문서는 Claude Code / Codex가 FEAK-TC v2를 구현할 때 따르는 **단일 기준 문서**다.
> 기존 `docs/SPEC_CORRUPTION_TVM.md`, `docs/PROJECT_CONTEXT.md`와 충돌하는 부분은 **이 문서가 우선**한다.
> 고민·논의 기록이 아니라 구현 지시다. 불명확한 것은 추측하지 말고 사용자에게 확인한다.

---

## 0. 프로젝트 제약 (스코프 판단 기준)

- 모든 단계는 **게이트(go/no-go) 통과 후에만** 다음 단계로 진행한다 (§6). 게이트 실패 시 다음 단계 구현 금지, 사용자에게 보고.
- 실험은 간소화 버전(§8)만 구현한다. 그 이상의 baseline/ablation은 사용자 요청 없이 추가하지 않는다.

---

## 1. 아키텍처 개요 (v2 확정)

```text
현재 글 + 과제
  → ① FEAK State Observer        (kanana 채점기 rubric 8 + 독립 룰베이스 feature 29)
  → ② Structured Working Memory  (수정 이력·checkpoint·잔여 예산, 비학습)
  → ③ Learned Revision Planner   (action·target span·intent 결정, 학습)
  → ④ LLM Patch Executor         (계획→국소 reversible patch, 범용 API, 비학습)
  → ⑤ Hard Validity Guard        (형식·범위·명백 오류 차단, 규칙)
  → ⑥ Text-aware TVM             (실제 수정 결과 vs NO_OP 비교, 학습)
  → ⑦ Checkpointed Controller    (accept / reject / replan / rollback / stop, 규칙)
  → 재관찰 후 반복 (closed-loop replanning)
```

| 모듈 | 학습 | 핵심 규칙 |
|---|---|---|
| ① Observer | X (기존) | 채점기에 자질 입력 금지 (독립성 유지 = 순환 방지 근거) |
| ② Memory | X | 세션 한정. 벡터DB·장기기억 금지 |
| ③ Planner | **O** | supervised imitation only. RL·policy gradient 금지 |
| ④ Executor | X | 전체 rewrite 금지. target span 국소 patch만 |
| ⑤ Guard | X | 품질 판단 금지. 형식·안전 위반만 차단 |
| ⑥ TVM | **O** | kanana + 새 LoRA 어댑터B (채점기 어댑터A와 별개) |
| ⑦ Controller | X | NO_OP 기준 통일 (§4) |

### v1 대비 변경 확정 (기존 문서보다 우선)
1. **학습된 Planner 도입** — "policy 학습 안 함" 결정은 폐기. 단 supervised(corruption 역연산 모방)이며 RL 아님. 논문 포지셔닝: "RL 없이 corruption 합성 supervision으로 R3식 plan 학습".
2. **action별 전수 열거는 기본 추론 경로에서 제거** — Planner top-k plan → plan별 후보 생성으로 대체. 단 열거 모드는 (a) baseline, (b) Planner용 real-log 수집기로 **코드에 유지**한다.
3. **STOP 후보 생성 제거** — STOP은 NO_OP 비교로 일원화 (§4).
4. **TVM은 text-aware가 본선** — compact feature GBM/MLP는 baseline이자 게이트 검증기로 유지.

---

## 2. 데이터 스키마 (핵심)

```python
PlanOutput:      {action_type, target_span, revision_intent, preserve_constraints}
Patch:           {action_type, target_span, before_text, after_text, revision_intent, preserve_constraints}
TransitionRecord:{plan, patch, feats_before, feats_after, rubrics_before, rubrics_after,
                  target_gain, non_target_drop, target_gap_reduction, evidence_match,
                  edit_ratio, goal_preservation, emb_sim, decision, step_idx, essay_id}
CorruptionStep:  {essay_id, chain_id, stage_k, corruption_op(=action 역연산), target_rubric,
                  target_features, text, measured_rubrics, measured_features,
                  generator(rule|llm), normalized(bool)}
```

- action_type: `ADD_DETAIL / DELETE_OR_FOCUS / COMPRESS / RESTRUCTURE / STYLE_REFINE` (+ 내부적으로 NO_OP)
- revision_intent(구조화 라벨): `ADD_SUPPORTING_EXPLANATION / REMOVE_REDUNDANCY / MERGE_OVERLAPPING_SENTENCES / RESTORE_LOGICAL_ORDER / IMPROVE_CONNECTIVE_FLOW / REFINE_FORMAL_STYLE`
- rubric 8개·feature 29개의 실제 키 이름: **TODO — 사용자 제공 후 `configs/schema.yaml`에 상수화** (제공 전 하드코딩 금지)

---

## 3. Corruption 파이프라인 (Planner·TVM 공통 supervision)

### 3.1 위계 원칙 (혼동 금지)
```text
corruption의 정의 축 = action 역연산   (무엇을 하는 연산인가)  ← action-grounded
rubric              = 조준 축          (어느 품질 축을 겨냥)     ← rubric-conditioned
feature             = 검증 계기        (의도한 변화가 났는지 실측) ← feature-audited
선호 라벨의 근거     = 재측정된 품질 순서 (생성 순서가 아님) + 독립 사람 평가로 최종 확인
```
- 뭉갠 뒤 **반드시 채점기+자질로 재측정**. 의도한 축이 실측으로 하락하지 않은 단계는 **폐기**.
- transition feature는 전부 실측값으로 계산. "뭉갰으니 나빠졌겠지" 가정 금지.
- 매 단계 `corruption_op`(action 역연산)·`target_rubric`·건드린 feature를 기록 → 역방향 transition에 action/intent/span 라벨 자동 부여.

#### 3.1.1 절대 금지 — feature를 목표값으로 최적화하는 corruption
- **feature 수치를 직접 떨어뜨리거나 목표에서 멀어지게 "최적화"하는 방식 금지.** (예: NN_repRatio를 올리도록, topicConsistency를 낮추도록 글을 생성)
  - 이유 1 (순환): feature로 생성 → feature로 라벨 → TVM 입력에도 같은 feature → TVM이 품질이 아니라 **생성 규칙을 역추적**. GBM이 쉽게 풀어버려 text-aware TVM의 존재 이유가 사라짐.
  - 이유 2 (비단조성): 많은 feature는 "높을수록/낮을수록 좋음"이 아니라 **적정 범위(U자형)**. word_Cnt(짧아도 길어도 문제), 어휘 다양성, 문장 간 overlap(너무 낮으면 단절·높으면 반복), 문장 길이, 종결어미 다양성 등. 극단으로 밀면 학생이 실제로 안 만드는 synthetic artifact가 됨.
- feature는 **생성의 목표가 아니라 사후 감사(audit)에만** 쓴다. "operator를 적용했더니 target rubric이 실제로 떨어졌나"를 확인하는 계기.

#### 3.1.2 rubric도 직접 편집 명령으로 쓰지 않는다 — operator를 거친다
- "organization을 낮춰라"처럼 LLM에 rubric 이름만 주면, 접속어 제거·문장 뒤섞기·문법 파괴·핵심 삭제 중 무엇을 할지 **통제 불능** → 무엇 때문에 점수가 떨어졌는지 알 수 없어 라벨 오염.
- rubric은 **operator를 선택하는 상위 조건**으로만 사용. 실제 훼손은 명시된 operator + **보존 조건**으로 수행:
```text
target_rubric : organization_1
operator      : SHUFFLE_FLOW
구체 동작      : 서로 다른 문장 2개의 위치 이동
보존 조건      : 내용·문법·문장 자체는 유지   ← 필수. 한 번에 한 축만 무너지게
```
- **모든 operator는 보존 조건(preserve)을 명시**해야 한다. 그래야 target 축만 정확히 하락한 깨끗한 corruption이 되고, "무엇 때문에 나빠졌는지"가 통제된다.

### 3.2 Action 역연산 대응표 (골격 — operator·보존조건 명시)
| corruption operator (순방향) | 보존 조건 (건드리지 말 것) | 역방향 action | intent 라벨 | 조준 rubric |
|---|---|---|---|---|
| DELETE_SPECIFICS (구체 사례·근거·수치 삭제) | 문장 문법·전체 논지 | ADD_DETAIL | ADD_SUPPORTING_EXPLANATION | content 구체성 |
| INSERT_OFFTOPIC (주제 무관 문장 삽입) | 기존 문장·문법 | DELETE_OR_FOCUS | REMOVE_REDUNDANCY | 주제 명료성 |
| INFLATE_REDUNDANCY (반복·장황화 주입) | 핵심 주장·사실 | COMPRESS | MERGE_OVERLAPPING_SENTENCES | 간결성 |
| SHUFFLE_FLOW (문장 순서 이동·접속 제거) | 내용·문법·문장 자체 | RESTRUCTURE | RESTORE_LOGICAL_ORDER / IMPROVE_CONNECTIVE_FLOW | organization |
| MONOTONE_STYLE (어미 단일화·어휘 반복) | 내용·문장 구조 | STYLE_REFINE | REFINE_FORMAL_STYLE | expression |

- **operator = 구체적·통제 가능한 편집 연산**(rubric 이름을 LLM에 직접 주지 않음). 각 operator는 target rubric 하나만 겨냥하고 보존 조건으로 나머지 축을 보호한다.
- feature는 이 표에 목표로 넣지 않는다. operator 적용 후 **재측정에서** target rubric이 실제로 떨어졌는지 감사할 때만 참조 (§3.1.1).
- 실제 rubric 키 이름(organization_1 등)으로의 확정은 §configs/schema.yaml에서. **feature 29개를 이 표에 미리 매핑할 필요 없음** — operator가 축을 정의하고 재측정이 검증하므로.

### 3.3 Anti-shortcut 규칙 (필수 — "생성규칙 역추적" 차단)
모델이 "품질"이 아니라 "훼손 생성기의 지문"을 배우는 것을 막는다.

1. **지문 균일화(de-confound)**: 체인의 모든 상태(원본 x0 포함)를 동일한 경량 LLM 정규화 패스(의미 보존 문장 다듬기)에 1회 통과시키는 `normalize` 옵션을 구현하고 **기본 ON**. 목적: "LLM 손길의 양 ∝ 나쁨"이라는 허위 상관 제거. 정규화 후에도 재측정 통과해야 채택.
2. **generator 다양화**: 같은 corruption_op을 (a) 규칙 기반, (b) LLM 프롬프트 변형 2종 이상으로 구현. 한 generator가 한 op을 독점하지 않게 배분 기록.
3. **feature-모호 쌍 의무 포함**: 학습쌍 중 일정 비율(초기 20%)은 compact transition feature만으로 GBM이 신뢰도 있게 구분 못 하는 쌍으로 구성(같은 축·유사 강도, 텍스트 실현 품질만 다른 쌍 — 예: 같은 plan에 대한 잘된 복구 vs 어색한 복구). 선별 방법: G2에서 학습한 GBM의 예측 확신도가 낮은 쌍을 우선 샘플링. 목적: text encoder가 feature 지름길로 학습을 회피하지 못하게 함.
4. **실데이터 2단계 학습(양측 모두)**:
   - TVM Stage-2: 실제 Planner+API 후보쌍 + NO_OP + 소량 human pair (양쪽 다 LLM 문체 → 지문 교란 소멸)
   - Planner Stage-2: 열거 모드 real-log에서 "실측(TVM/휴리스틱)상 이긴 action/span"을 distill (synthetic-to-real 보정)
5. **판정 게이트**: 위 안전장치의 성패는 G2/G5의 **held-out 사람 선호 일치율**로만 판정. 통과 못 하면 corruption 재설계.

### 3.4 데이터 분할
- corruption 생성 **전에** 원본 에세이 단위로 train/val/test 분할. 같은 원문 파생 상태는 같은 split.
- Planner↔TVM cross-fitting: fold A로 Planner 학습 → fold B에서 후보 생성해 TVM 데이터로 (반대도 동일).

---

## 4. NO_OP 통일 규칙 (Controller)

- 매 step, 모든 patch 후보와 함께 **NO_OP(현재 글 유지)** 를 TVM으로 평가.
- accept: 최상위 patch가 NO_OP + margin 초과. reject: Guard 위반 또는 NO_OP 이하.
- replan: 현재 plan의 전 후보 실패 or 예측-실측 괴리 큼 → Planner 재호출(다른 plan).
- rollback: **누적** 판정 전용 — best checkpoint 대비 악화 or 원문 대비 drift(임베딩 유사도 누적 하락) → best checkpoint 복귀. reject와 혼용 금지.
- stop: NO_OP가 연속 k-step 최상위 or 예산 소진.
- 모든 임계값은 `configs/controller.yaml` 상수. val split로만 튜닝.

---

## 5. 학습 스펙

### 5.1 Planner
- **1차(Planner-lite, 필수)**: LightGBM 2개 — action 분류(rubric8+feature29+메모리 요약 feature 입력), 문장 단위 span 랭킹(문장별 feature). intent는 action→intent 규칙 매핑으로 시작.
- **2차(선택 업그레이드)**: 문장 임베딩 + FEAK 상태 입력의 소형 encoder + 3-head(action/span/intent). **G3 통과 + 일정 여유 시에만** 구현. 졸업 일정상 lite로 충분하면 lite가 최종.
- supervision: corruption 역연산 (§3). 모호 사례(복수 정답 action)는 multi-label/soft label.
- Stage-2: real-log distill (§3.3-4).

### 5.2 TVM (text-aware)
- 구조(2026-08-21 사용자 승인 변경): 주 모델 Qwen2.5-7B-Instruct + LoRA + scalar head,
  matched-backbone 대조군 Kanana-8B + 별도 LoRA/head. 기존 Kanana 단일 본선보다
  채점기와 같은 backbone 효과를 직접 측정할 수 있다.
- 입력 프롬프트: `[과제][수정 전(target 주변 문맥)][수정 후][plan: action/span/intent][transition feature 수치 텍스트]`
- Loss: pairwise BT + corruption 단계차 margin: `L = -log σ((V_c - V_r) - m(Δstage))`
- 학습 규칙(위반 금지): **1 epoch 엄수** / LR 1~3e-6, AdamW, warmup ~10 / 재측정 차이 미미한 쌍 필터 / 학습 후 zero-mean 정규화.
- 커리큘럼: Stage-1 corruption 쌍(+feature-모호 쌍 20%) → Stage-2 실후보+NO_OP+human pair.
- baseline: 동일 쌍으로 GBM(feature-only), 휴리스틱 가중합, immediate FEAK gain.
- 입력 ablation: 두 backbone 모두 `full`과 `scorer_free`를 학습한다.
  `scorer_free`는 Kanana score 출력인 `target_gain`, `non_target_drop`을 제외한다.

---

## 6. 구현 게이트 (순서 엄수, 실패 시 정지·보고)

```text
G0  diagnose 출력의 rubric/feature **키 형식만** 확인 → configs/schema.yaml에 상수화.
    (feature 29개를 대응표에 미리 매핑할 필요 없음 §3.2. operator가 축을 정의하고 재측정이 검증.
     rubric 키는 operator의 target_rubric 지정·재측정 판정에 쓰이므로 이름 확인 필요.)
G1  corruption 파이프라인 (재측정·기록·정규화 포함) → 소량(에세이 30~50편) 생성
G2  GBM sanity: corruption 쌍 held-out pairwise acc가 랜덤 대비 유의하게 높음
    + 소표본(50쌍) 사람 선호 일치 ≥ 70% 수준     [실패: corruption 재설계, 진행 금지]
G3  Planner-lite 학습: held-out action acc / span top-k recall 보고
G4  대량 corruption 생성 → TVM Stage-1 → 열거 real-log 수집 → TVM Stage-2
G5  TVM 검증: held-out human pairwise에서 GBM·휴리스틱·immediate-gain 초과
    [실패: text-aware 포기, feature TVM으로 축소 후 사용자와 재논의]
G6  Planner Stage-2(real-log) → Controller 통합 → 간소화 평가(§8)
```

---

## 7. 폴더 구조 (추가분)

```text
src/feak_tc/
  corruption/   ops.py(역연산들) normalize.py chain.py measure.py pairs.py
  planner/      lite.py(GBM) features.py [encoder.py는 2차]
  tvm/          dataset.py model.py(kanana+어댑터B) train.py baselines.py
  controller/   noop.py decisions.py memory.py
configs/        schema.yaml corruption.yaml controller.yaml train_tvm.yaml
scripts/        10_gen_corruption.py 11_gbm_sanity.py 12_train_planner_lite.py
                13_train_tvm.py 14_run_agent.py 15_eval.py
```

---

## 8. 간소화 평가 (이것만 구현)

- **Baselines (5)**: ① LLM self-refine ② LLM planner+executor(비학습) ③ Planner-lite+executor+immediate-FEAK-gain ④ +feature-TVM ⑤ +text-TVM full controller
- **Ablations (4)**: text 입력 제거 / NO_OP 제거 / rollback 제거 / TVM Stage-2(real 보정) 제거
- **지표**: 최종 rubric 개선, non-target 회귀율, 과수정율, stop 적절성(NO_OP 판별 acc), held-out human pairwise 일치, (Planner) action acc·span recall
- human eval: 소규모(수십~백 쌍 pairwise, **blind 제시** 필수)

---

## 9. 금지 목록 (v2에서도 유지)

policy-gradient/RL 학습, 전체 rewrite, 벡터DB·장기기억, multi-agent 토론, tree search,
채점기(어댑터A) 재학습·수정, 자질을 채점기 입력에 주입, 원본 데이터·가중치·.env 커밋,
사용자 승인 없는 git push, 게이트 건너뛰기, §8 외 실험 추가.
