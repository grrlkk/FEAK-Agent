# SPEC — Corruption 데이터 생성 & TVM 학습

> MVP 루프가 돌아간 뒤의 다음 단계 스펙. action 역연산 corruption 설계(확정)를 포함한다.

## 0. 목적

TVM("이 수정 transition이 좋은가"를 채점하는 모델)의 학습쌍을 만든다.
보유 데이터는 "한 글 + 사람 점수"뿐이고 수정 전후 쌍이 없으므로, **corruption으로 합성**한다.

## 1. 핵심 설계 — Corruption은 action의 역연산이다 (확정)

단순히 글을 뭉개면 안 된다. **뭉개는 방법을 action taxonomy의 역연산으로 정의**해야
역방향 transition에 action 라벨이 자동으로 붙는다.

```
x0 --[구체 사례 삭제]--> x1        (corruption, 순방향)
x1 --[?]--> x0                     (역방향)
      = "구체 사례를 추가해 좋아진 transition"
      = action_type: ADD_DETAIL    ← 라벨 공짜
```

### Corruption ↔ Action 대응표 (설계의 뼈대)

| Corruption (순방향, 뭉개기) | 역방향 action | 주로 건드리는 rubric/feature |
|---|---|---|
| 구체 사례·근거·수치 삭제 → 일반론으로 | **ADD_DETAIL** | content 구체성 |
| 주제 무관 문장/여담 삽입 | **DELETE_OR_FOCUS** | 주제 명료성 |
| 같은 말 반복·장황화 주입 | **COMPRESS** | 간결성, 길이 계열 feature |
| 문단/문장 순서 섞기, 접속 표현 제거 | **RESTRUCTURE** | organization, 응집성 |
| 종결어미 단일화, 어휘 반복 주입 | **STYLE_REFINE** | expression, 다양성 feature |

- 실제 rubric 8개 이름과 feature 29개 목록을 받아 이 표를 세분화할 것 (TODO: 사용자에게 요청).
- 순방향(뭉갬) 자체도 활용: "같은 action_type의 나쁜 버전" = **hard negative**
  (예: 주제 무관 문장을 삽입한 ADD는 나쁜 ADD → "ADD가 항상 좋은 게 아니라 무엇을 ADD했는지가 중요"를 학습).

## 2. Corruption 생성 절차

```
① 소스 선별: 4만 건에서 사람 평균점수 상위 글 추출 (rubric별 고루)
② 체인 생성: 글마다 action 역연산 corruption을 단계적으로 적용
   x0 → x1 → x2 → x3  (단계가 깊을수록 나쁨)
   - 방식: LLM 지시 + 규칙 혼합 ("이 문단의 구체 사례를 제거하되 자연스럽게 이어라")
③ 재측정 (필수): 각 단계를 채점기+자질로 실측
   - 의도한 축이 실제로 떨어졌는지 검증. 안 떨어졌으면 그 단계 폐기.
   - transition feature는 실측값으로 계산 ("뭉갰으니 나빠졌겠지" 가정 금지)
④ 쌍 추출: 체인에서 (K choose 2) 선호쌍
   - 덜 망가진 쪽 = preferred
   - 단계 차이 = confidence (loss의 margin으로 사용)
   - 단계 차이가 너무 작아 재측정 차이가 미미한 쌍은 필터링
```

## 3. 데이터 혼합 (corruption만으로는 안 됨)

| 소스 | 역할 | 규모 |
|---|---|---|
| ① corruption 쌍 | 하한선 학습 ("명백히 나쁜 방향"), 대량 | 주력 |
| ② 실제 LLM 수정 쌍 (MVP 루프 산출) | 분포 보정 (인위적 corruption ≠ 실제 LLM 수정) | 소량 혼합 |
| ③ 사람 보정 쌍 | 상한선 ("좋은 vs 더 좋은" 미세 판단) + held-out 평가 | 수백 쌍 |

- ②는 MVP 루프의 trajectory 로그에서 수집 → 자동 라벨(채점기+자질+의도보존 종합).
- ③은 ②의 후보쌍을 사람이 pairwise 비교. 학습용과 **held-out 평가용을 분리**할 것.

## 4. TVM 학습 스펙

### 구조
```
kanana-8B (LoRA, 새 어댑터B — 채점기 어댑터A와 별개)
→ end-of-response 토큰 hidden state
→ linear head → scalar value
```

### 입력 포맷 (텍스트 프롬프트)
```
[과제 프롬프트]
[수정 전 글(또는 수정 부분±문맥)]
[수정 후 글]
[action: DELETE_OR_FOCUS, target_rubric: ...]
[transition: target_gain=+0.5, non_target_drop=0.1, evidence_match=0.8, ...]
```
- transition feature는 head concat이 아니라 프롬프트 텍스트로 (간단·표준). head concat은 ablation.

### Loss
```
L = -log σ( (V(chosen) - V(rejected)) - m(Δstage) )
```
- 기본 Bradley-Terry + corruption 단계 차이 기반 margin(Δstage 클수록 큰 margin).
- Scaled-BT/margin 계열이 regular BT보다 낫다는 보고 다수.

### 학습 규칙 (함정 회피 — 위반 금지)
- **1 epoch 엄수** (BT reward model은 2 epoch부터 성능 붕괴 보고 다수)
- LR 1~3e-6 탐색, AdamW, warmup ~10 steps
- 재측정 점수차가 미미한 쌍 제외
- 학습 후 출력 zero-mean 정규화 (controller 임계값 안정화)
- 2단계 커리큘럼: Stage1 = corruption+LLM쌍 대량 → Stage2 = 사람 보정쌍 소량

### 순서 (싸게 찔러보고 비싸게 들어가기)
1. **GBM baseline 먼저**: transition feature 9개만 입력 (텍스트 없이) → LightGBM으로
   corruption 쌍이 학습되는지 며칠 안에 확인. 안 되면 데이터 문제 → 8B 학습 전에 발견.
2. 되면 kanana TVM 본 학습.

## 5. 검증 (필수 리포트)

- held-out 사람 선호쌍 pairwise accuracy (메인 지표)
- hard negative rejection rate ("점수 올랐지만 훼손" 후보를 거르는가)
- 휴리스틱 가중합 / GBM / kanana TVM 비교 (TVM이 이겨야 존재 정당화)
- calibration: 점수 차이 클수록 실제로 더 확실히 좋은가
- held-out **글**(학습에 안 쓴 글)에 대한 일반화 — corruption이 특정 글 암기가 아님을 확인

## 6. 알려진 리스크 (문서화된 것, 코드에서 로그로 확인 가능하게)

- corruption은 인위적 분포 → 실제 LLM 수정과 격차 (③ 혼합으로 완화, 격차 측정 리포트)
- corruption은 "나쁜→보통"만 가르침, "좋은→더 좋은"은 사람 보정쌍이 담당
- TVM 입력이 텍스트를 포함하므로 "인위적으로 망친 티"를 외울 위험 → LLM 기반 자연스러운 corruption 우선, 규칙 기반은 보조
