# FEAK-TC 최종본: 서론 + 방법론

> **한 줄 정의**
> FEAK-TC는 FEAK의 진단 신호로 수정 transition을 평가하는 value model(TVM)을 학습하여,
> Open-ended Writing Agent의 반복 수정 경로를 **선택·검증·복구·종료**하는 value-guided control agent다.

---

# 1부. 서론

## X. 분야의 큰 흐름

- LLM 기반 글쓰기 시스템은 단발 생성·단발 피드백을 넘어, 글쓰기 과정에 반복적으로 개입하는 **Writing Agent paradigm**으로 이동하고 있다.
- 글쓰기는 정답과 종료점이 고정되지 않고, 같은 목표를 만족하는 수정 경로가 여러 개인 **open-ended task**다.
- Gooding et al.은 Writing Agent의 핵심 요구를 **exploration → evaluation → goal alignment**의 반복으로 정리한다.
- 따라서 Writing Agent의 성능은 좋은 문장을 생성하는 능력뿐 아니라, **수정 경로를 안정적으로 관리하는 능력**으로 결정된다.

## Y. 본 연구가 자리 잡는 영역

- 본 연구는 Open-ended Writing Agent의 **trajectory-level revision control**에 초점을 둔다.
- 핵심 관점: 평가 단위를 최종 글이나 피드백 문장이 아니라 **revision transition** 으로 옮긴다.

```text
현재 상태 s_t  →  수정 action a_t  →  다음 상태 s_{t+1}
질문: 이 action이 현재 상태를 "더 나은 다음 상태"로 옮겼는가?
```

- Agent는 매 step 다음을 판단한다.
  - 현재 상태에서 어떤 action을 시도할 것인가 (B1)
  - 그 action이 현재 글을 실제로 개선했고 다른 부분을 훼손하지 않았는가 (B2)
  - 후보를 거절할지, 이전 checkpoint로 복귀할지, 종료할지 (B3)
- 핵심은 **generation quality**가 아니라 **revision control quality**다.

## Z. 기존 연구 흐름

| 흐름 | 대표 연구 | 강조점 |
|---|---|---|
| Z1. Planning·Action 탐색 | CRAFT, WriteHERE, SuperWriter | 수정을 planning/search로 구조화, 경로 탐색 |
| Z2. Evaluation Signal | Rationalize and Align, LitBench, APRES, FeedbackWriter | reward model·rubric·rationale·evaluator로 평가 강화 |
| Z3. Intent·Evidence·Grounding | Intention-Tuning, DeepWriter, QRAFT, Beyond Single-shot Writing | intent conditioning·grounding으로 목표 이탈·회귀 완화 |

> 위 분류는 각 연구가 한 문제만 다뤘다는 뜻이 아니라 주된 강조점 기준이다.

각 흐름은 유효한 부품을 만들었다. 그러나 반복 수정에서 이 셋은 **서로의 전제**이므로 분리된 채로는 경로를 제어하지 못한다.

- step마다 수정이 좋았는지 평가할 수 없으면(Z2 부재), 나빠진 경로를 되돌릴 근거가 없다(Z3 불가).
- 되돌릴 수 없으면, agent는 삭제·재구성처럼 위험하지만 필요한 action을 고르지 못하고 안전한 추가에 치우친다(Z1로 회귀, action bias).
- action을 상태에 맞게 고르지 못하면, 평가와 복구는 매 step 뒷수습만 하게 된다.

## B. 기존 연구가 남긴 문제

### B1. 현재 상태에서 후보 action의 상대적 가치를 판단하는 기준 부족
- planning·search는 가능한 경로를 탐색하지만, 현재 글 상태에서 add / delete / compress / restructure / stop 중 무엇이 더 유익한지 비교하는 기준은 제한적이다.
- **핵심 차이**: 구조를 계획하는 것과, 지금 이 상태에서 가장 가치 있는 action을 고르는 것은 다른 문제다.

### B2. Step의 transition 가치를 평가하는 신호 부족
- 기존 평가는 최종 결과나 결과 간 선호에 집중한다. 그러나 즉시 점수가 오른 수정도 비목표 rubric 회귀·과수정·후속 수정 가능성 저하를 일으킬 수 있다.
- 필요한 것은 단순 즉시 score gain이 아니라, **이번 action이 현재 상태를 더 나은 상태로 옮겼는지를 transition 단위로 평가하는 신호**다.
- **핵심 차이**: 좋은 최종 결과를 고르는 것과, 좋은 수정 transition을 고르는 것은 다른 문제다.

### B3. 경로 악화 이후의 복구·종료 부족
- intent conditioning·grounding은 악화를 예방하지만, 이미 채택된 경로가 나빠졌을 때 거절·복귀·종료를 결정하는 것은 별도 문제다.
- 본 연구는 완전한 semantic drift 해결이 아니라, **비목표 rubric 회귀·과수정·trajectory degradation의 감지와 복구·종료**에 범위를 둔다.
- **핵심 차이**: drift를 예방하는 것과, 이미 나빠진 경로를 복구·종료하는 것은 다른 문제다.

### B 한 문장 정리
> 기존 Writing Agent 연구는 planning·evaluation·grounding을 각각 발전시켰지만, 후보 action의 transition 가치를 추정하고 그 결과를 **reject·rollback·stopping**으로 연결하는 trajectory-level closed-loop control은 충분히 확립되지 않았다.

## W. 본 연구의 방향

### W1. FEAK를 Agent의 Diagnoser로 재구성
- 채점기는 **kanana-8B + LoRA**로 학습된 rubric scorer이며, 입력(글) → 8 rubric score를 출력한다.
- linguistic feature(자질)는 채점기와 **독립적으로(규칙 기반)** 계산된다. 채점기에 자질을 입력으로 주입하는 방식도 실험했으나 성능 향상이 미미하여, 채점기는 글→rubric으로 고정하고 자질은 독립 신호로 둔다. 이 분리는 이후 transition 평가에서 rubric 신호와 feature 신호를 **독립 신호로 활용**할 근거가 된다(순환 완화).
- 이 신호들(rubric, feature, elite gap, weak feature, evidence sheet)을 단발 피드백 근거가 아니라 **매 step 상태·변화를 측정하는 dense diagnostic signal**로 재구성한다.
- **왜 FEAK인가**: control loop가 매 step 판단 근거로 삼는 신호는 정량적이어서 transition 간 비교가 가능하고, 검증 가능해서 거절·복구·종료를 감사할 수 있어야 한다. LLM 자기평가만으로는 둘 다 불안정하다(B2). FEAK는 한국어 글쓰기에서 이 조건을 이미 만족하는, 측정값으로 역추적되는 진단 신호다.
- FEAK는 최종 reward나 controller가 아니라 **Diagnoser Tool**이다.

### W2. Action 유형별 Counterfactual 후보 생성
- 같은 상태에서 add / delete·focus / compress / restructure / style refine / stop 후보를 각각 생성한다.
- 각 후보는 전체 rewrite가 아닌 **reversible patch**로 적용한다.
- LLM의 추가·보강 편향을 줄이고, 동일 state에서 서로 다른 action의 효과를 비교한다.

### W3. FEAK-Guided Corruption으로 학습 데이터 생성 (핵심 novelty)
- "좋은 수정"의 도착점(사람 고점수 글)은 이미 데이터에 존재한다.
- 비싼 rollout으로 앞을 내다보는 대신, 좋은 글을 **진단 신호 기준으로 단계적으로 망가뜨려** 라벨을 만든다.
- 같은 글의 corruption 단계에서 **덜 망가진 상태 = preferred, 더 망가진 상태 = rejected** 의 선호쌍을 얻는다.
- corruption 축·강도가 알려져 있어 transition feature가 자동 라벨링되고, 난이도(hard/easy negative)도 통제된다.

### W4. Transition Value Model과 Checkpointed Controller
- 보정된 후보 선호쌍으로 **TVM**을 학습한다. 실제 실행 시 rollout 없이 TVM만으로 후보 transition 가치를 예측한다.
- Controller는 TVM value와 명시적 제약으로 결정한다.
  - **Accept** 유익한 후보 적용 · **Reject** 적용 전 폐기 · **Rollback** 경로 악화 시 이전 best checkpoint 복귀 · **Stop** 유익한 후보 없을 시 종료

### W 한 문장 정리
> 본 연구는 좋은 글을 FEAK 진단 신호 기준으로 corruption하여 transition value 학습 데이터를 만들고, 이를 TVM에 학습하여 Open-ended Writing Agent의 action 선택·검증·복구·종료를 하나의 closed loop로 통합한다.

## 연구 질문
- **RQ1.** 즉시 FEAK gain이 높은 action과 실제로 더 나은 transition 사이에 유의미한 불일치(과수정·회귀)가 존재하는가?
- **RQ2.** corruption으로 만든 선호쌍이 사람의 후보 선호와 일치하는가?
- **RQ3.** corruption 라벨로 학습한 TVM이 휴리스틱·즉시 gain 기준보다 사람 pairwise를 더 잘 맞히고, held-out 글에 일반화되는가?
- **RQ4.** TVM 기반 agent가 self-refinement·즉시 gain controller보다 회귀·과수정을 줄이고 더 나은 종료 시점을 선택하는가?

## 예상 기여
1. Open-ended writing revision을 **transition-level value-guided control** 문제로 정식화하고, action 선택·검증·복구·종료를 하나의 closed loop로 통합
2. FEAK 진단 신호로 수정 transition을 평가하는 **TVM 기반 value-guided action selection**
3. TVM 학습 데이터를, 비싼 rollout 없이 **FEAK-guided corruption**으로 생성하는 방법
4. accept·reject·rollback·stop을 구분하는 **checkpointed revision controller**
5. 최종 품질뿐 아니라 action preference·regression·over-editing·rollback·stopping을 함께 보는 **trajectory-level 평가 설계**

---

# 2부. 방법론

## 2.1 핵심 아이디어
- 글 수정을 단발 생성이 아니라 **순차적 경로 제어(trajectory control)** 문제로 본다.
- 매 step 여러 action 후보를 만들고, 각 후보가 더 나은 transition인지 TVM으로 평가한다.
- 학습 데이터는 비싼 rollout이 아니라 **FEAK-guided corruption**으로 생성한다.

## 2.2 문제 정의 (MDP)
```text
State   s_t : 원본 과제 + 원본 글 + 현재 글 + FEAK 관측 + accepted checkpoint 이력
Action  a_t : (action_type, target_span, instruction)
              action_type ∈ { ADD_DETAIL / DELETE_OR_FOCUS / COMPRESS / RESTRUCTURE / STYLE_REFINE / STOP }
Transition  : s_t --a_t--> s_{t+1}   (평가의 단위)
Value       : V_θ(s_t, a_t, s_{t+1}) — 이 transition이 더 나은 상태로 가는가
```
- action은 이산 타입이 아니라 **parameterized action** (type, target_span, instruction)이다.
  같은 타입이라도 무엇을 어디에 실행했는지에 따라 transition 가치가 달라지며, 이것이
  hard negative 학습("주제 무관 문장을 삽입한 ADD는 나쁜 ADD")이 성립하는 전제다.
- **reward 함수는 명시하지 않는다** (reward 없는 MDP 정식화, MDP\R; Wirth et al., 2017).
  글쓰기 품질은 수치 reward로 명세하기 어렵고, 채점기의 즉시 score gain은 채점 노이즈와
  과수정 불일치(RQ1)를 포함하므로, transition 선호쌍으로 V_θ를 직접 학습한다
  — preference-based RL(Christiano et al., 2017), BT reward model(Stiennon et al., 2020;
  Ouyang et al., 2022), 확신도 기반 margin(Llama 2, Touvron et al., 2023; 우리는 확신도를
  사람 라벨 대신 corruption 단계차로 획득), step 단위 가치 평가(Lightman et al., 2023)와
  같은 계열이다.
- action-generation policy를 직접 학습하지 않는다. 같은 state에서 후보를 펼치고 TVM argmax로 **state-conditioned action selection**을 유도한다: `π(a|s) = argmax_a V_θ(φ)`.

## 2.3 전체 루프
```text
현재 글
 → FEAK Diagnoser (관측)
 → Action별 counterfactual 후보 생성
 → Patch 적용 (reversible)
 → 각 후보를 compact transition vector로 표현
 → TVM 가치 예측
 → Checkpointed Controller (accept / reject / rollback / stop)
 → 종료 조건까지 반복
```

## 2.4 구성요소

### (1) FEAK Diagnoser
- 글 → 8 rubric, 29 feature, elite gap, weak rubric/feature, evidence sheet. 관측·진단 전용.

### (2) Stratified Action Proposer
- 같은 state에서 action 유형별 후보 생성. add 편향 완화. 후보 생성기는 1차에서 학습하지 않음.

### (3) Patch Simulator
- 각 후보를 reversible patch(operation/target_span/before/after/reason)로 적용. 전체 rewrite 금지.

### (4) Compact Transition Representation
```text
φ(s,a,s') = [action_type, target_rubric, target_gain, target_gap_reduction,
             non_target_drop, evidence_match, edit_ratio, goal_preservation,
             emb_sim]   # 수정 전후 텍스트 임베딩 유사도 (고정 임베딩, 학습 안 함)
```
- rubric 신호(target_gain 등, 채점기)와 feature 신호(target_gap_reduction 등, 독립 계산)를 **둘 다** 포함한다. 서로 다른 것을 보므로 둘의 일치/불일치가 hard negative 판별에 유용하다.
- `emb_sim`은 고정 임베딩(예: 한국어 SBERT)으로 계산하며 학습하지 않는다. goal_preservation을 보강하고 텍스트 뉘앙스를 가볍게 포착한다(PGSRM식 임베딩 신호).
- 위 신호 외 추가는 ablation으로 필요성 입증 전까지 금지(feature creep 방지).

### (5) Transition Value Model (TVM) — 국소 평가
- 입력 φ → transition value. pairwise ranking 학습:
```text
L = -log σ( V_θ(φ_preferred) - V_θ(φ_rejected) )
```
- 실제 실행 시 rollout 없이 TVM만 사용.
- TVM은 **한 수정의 국소적 가치**를 본다. 매 step "이 수정이 좋은가"는 잘 잡지만, 여러 step 누적으로 글 전체가 원래 의도에서 틀어지는 것(drift)은 국소 판단만으로 못 잡는다.

### (5-b) Drift 추적 — 전역 평가
- TVM(국소)이 못 보는 **경로 전체의 틀어짐**을 별도로 추적한다.
- 원본 글/과제 대비 현재 상태의 누적 이탈을 측정(임베딩 유사도 + 누적 non-target regression).
- 매 step은 각각 통과했더라도, 누적 이탈이 임계를 넘으면 drift로 판정한다.
- 역할 분담: **TVM = 미시(이 수정 좋은가) / Drift = 거시(경로 전체가 틀어졌나)**.

### (6) Checkpointed Controller
```text
accept   : 유익한 후보를 현재 state로 확정 + checkpoint 저장
reject   : 적용 전, TVM이 나쁘다고 본 후보 폐기 (미시 판단)
rollback : Drift가 누적돼 경로 전체가 틀어지면, 덜 틀어진 이전 checkpoint로 복귀 (거시 판단)
stop     : 유익한 후보가 없는 상태가 일정 step 지속되면 종료
```
- **reject와 rollback의 구분**: reject는 TVM(미시)이 이 수정 하나를 거르는 것, rollback은 Drift(거시)가 경로 전체 틀어짐을 잡아 되돌리는 것. 둘은 다른 층의 판단이다.
- TVM value + hard constraint(non_target_drop 상한, evidence_match 하한, edit_ratio 상한, goal_preservation 하한) 병용.

- TVM 모델 구조: 입력이 compact feature(+고정 임베딩 신호)이므로 **작은 MLP(main) / GBM(LightGBM·XGBoost, baseline)** 로 충분. 텍스트를 무거운 LLM backbone으로 인코딩하지 않는다(입력이 이미 수치 신호라 과함). 임베딩은 고정 모델로 미리 계산.

## 2.5 학습 데이터: FEAK-Guided Corruption

### 절차
```text
좋은 글 x0 (사람 고점수)
 → FEAK로 강한 feature/rubric 식별
 → 그 축을 따라 단계적으로 망가뜨림: x0 → x1 → x2 → ...
 → 각 단계: 어떤 rubric/feature를 얼마나 떨궜는지 기록
```
- **corruption 축**: feature 기준(예: 종결어미 다양성↓) 또는 rubric 기준(예: organization 흐트러뜨리기).
- **선호쌍**: 같은 글에서 덜 망가진 상태 > 더 망가진 상태. 역방향(나쁜→좋은)이 좋은 수정, 순방향이 나쁜 수정의 정답 신호.

### 일반화·분포 보정 (함정 완화)
- corruption은 인위적 분포 → **실제 LLM이 만든 수정 후보 소량을 섞어** train/test 분포 차이를 줄인다.
- corruption은 "나쁜→보통"을 가르치지 "좋은→더 좋은"은 못 가르침 → **소량의 human-calibrated pair**로 상한 영역 보완.
- TVM 입력이 *특정 글*이 아니라 *추상 transition feature*이므로, "이 원본으로 가라"가 아니라 "이런 변화가 좋다/나쁘다"를 학습 → open-ended와 충돌하지 않음(held-out 일반화로 검증).

### 순환 방지
- 라벨의 최종 근거는 **사람 점수(고점수 글)** 이지 채점 모델 자기참조가 아니다. corruption 축·강도만 FEAK feature로 정의한다.

### Hard Negative (corruption + 규칙)
- target_gain↑인데 non_target_drop 큼 / goal_preservation 낮음 / evidence 무관 / edit_ratio 과함 / action이 진단과 반대 / 표현만 좋아지고 핵심 미해결.

## 2.6 서론(B) 대응
| 문제 | 대응 |
|---|---|
| B1 | action 유형별 후보 + TVM value-guided selection |
| B2 | corruption 선호쌍으로 학습한 TVM의 transition 평가 |
| B3 | checkpointed rollback + stop |
| 통합 | 위 셋을 하나의 closed-loop control로 결합 |

## 2.7 먼저 검증할 전제 (Phase A)
1. 즉시 gain만으로 후보를 고르면 과수정·회귀 후보를 자주 고르는가 (→ TVM 필요성)
2. corruption 선호쌍이 사람 후보 선호와 일치하는가 (→ 라벨 타당성)
3. corruption TVM이 휴리스틱보다 사람 pairwise를 잘 맞히고 held-out에 일반화되는가 (→ 학습 가치)
```text
2·3 성립      → FEAK-guided Corruption TVM을 핵심으로 확정
불일치        → corruption 축 재설계 또는 라벨 전략 변경
즉시 gain 충분 → TVM 축소, 단순 controller로
```

## 2.8 평가
- **TVM**: held-out human pairwise accuracy, 휴리스틱·즉시 gain 대비 개선, calibration, hard-negative rejection
- **corruption 라벨**: 사람 선호 일치도, 실제 LLM 수정 분포와의 격차
- **Agent**: 최종 품질, non-target regression rate, over-edit ratio, rollback 효과, stop 적절성, trajectory-level human preference
- **비교군**: Original FEAK / Repeated FEAK / LLM self-refine / 즉시 gain controller / 휴리스틱 TVM / Corruption TVM Agent
- **ablation**: stratified proposal·TVM·evidence_match·non_target_drop·goal_preservation·rollback·stop 각각 제거

## 2.9 핵심 novelty
1. revision을 **transition 단위**로 평가하고, action 선택·검증·복구·종료를 **closed-loop control**로 통합
2. 같은 state에서 **action 유형별 counterfactual 후보** 생성 (add 편향 완화)
3. FEAK 진단 신호로 transition을 평가하는 **TVM 기반 value-guided action selection**
4. TVM 학습 데이터를 비싼 rollout 없이 **FEAK-guided corruption**으로 생성

## 2.10 범위 밖 (future work)
- action-generation policy의 직접 SFT/DPO/RL (TVM 로그 축적 후)
- learned tool router, RAG, multi-agent, full semantic-drift 해결, writer-voice 모델, 장기 user memory

---

## 부록. 남은 검증과 리스크 (정직한 메모)
- **이 전체는 Phase A에 인질로 잡혀 있다**: 즉시 gain으로 충분하면 TVM 불필요, corruption이 사람 선호와 안 맞으면 novelty 기각. 종이로 못 풀고 50~100개 글 실험으로만 답이 나온다.
- **novelty의 본질**: 새 알고리즘이 아니라 "transition control 문제 정식화 + corruption 기반 weak supervision + 탄탄한 검증". problem formulation + empirical evidence 형 기여.
- **agent 성격**: tool-using LLM agent가 아니라 **value-guided control agent**. 정직하게 그렇게 포지셔닝.
- **venue 주의**: WSDM(검색·데이터마이닝)은 주제 적합성 desk-reject 위험. 제출 시 글쓰기 교육이 아니라 method(corruption value learning)를 전면화하거나, 교육AI(AIED/L@S)·NLP(ACL/EMNLP Findings)를 안전망으로 둘 것.
