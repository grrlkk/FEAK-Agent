# FEAK-TC MVP 구현 지시서 (Codex용)

## 목표

FEAK-TC의 **한 step revision loop**를 학습 없이 돌아가게 만든다.
TVM 학습, corruption 데이터, drift/rollback, DB는 **이번 범위 밖**이다.
목표는 "글 하나가 진단 → 후보생성 → 수정 → 재진단 → 휴리스틱 선택으로 한 바퀴 도는 것"을 눈으로 확인하는 것.

---

## 이번에 구현하는 것 / 안 하는 것

### 구현한다
- diagnose 래퍼 (채점기 + 자질을 하나의 함수로)
- action별 후보 생성 (LLM)
- patch 적용 (reversible)
- 재진단
- transition feature 계산
- 휴리스틱 선택 (accept/reject/stop)
- 전체를 잇는 one-step loop + 로그

### 구현하지 않는다 (명시적 제외)
- TVM 학습 / corruption 데이터 생성
- drift 추적 / rollback
- MongoDB / 데이터 인제스트 파이프라인
- 다단계(multi-step) trajectory (이번엔 one-step 확인만; 반복은 최소 래퍼만)
- FastAPI / 프론트엔드
- 채점기·자질 계산기 재학습 (이미 있음, 호출만)

---

## 전제 (이미 준비된 것)

호출 가능한 함수가 두 개 있다고 가정한다. **실제 시그니처는 사용자가 채워 넣는다.**

```python
# 채점기: kanana-8B + LoRA (학습 완료, 어댑터A), 글 -> rubric 점수
# 주의: 채점기에 자질을 입력하지 않는다 (확정 결정 — PROJECT_CONTEXT.md 참조)
rubric_scores = score_rubrics(text: str) -> dict[str, float]
# 예: {"task_1": 4.0, "content_1": 3.5, ..., "expression_2": 2.0}  (8개 rubric)

# 자질: 채점기와 독립, 규칙 기반, 글 -> feature
features = extract_features(text: str) -> dict[str, float]
# 예: {"len_word": 320, "ending_diversity": 0.4, ...}  (29개 내외)
```

LLM 후보 생성용 API (예: GPT-4o-mini)도 사용 가능하다고 가정한다.
키/엔드포인트는 `.env`로 주입.

---

## 모듈 구조

```
src/feak_tc/mvp/
  diagnose.py      # 채점기 + 자질 -> Diagnosis
  propose.py       # LLM -> action별 후보 (JSON)
  patch.py         # 후보를 글에 적용 (reversible)
  transition.py    # 수정 전후 -> transition feature
  heuristic.py     # 휴리스틱 점수 + 선택
  loop.py          # 위를 잇는 one-step loop
  schemas.py       # 데이터 클래스
configs/
  action_taxonomy.yaml
  heuristic.yaml   # 임계값·가중치 (상수)
tests/
  test_mvp_loop.py # 하드코딩 글 1개로 스모크 테스트
scripts/
  run_mvp.py       # 글 1개 넣고 한 바퀴 실행 + 로그 출력
```

---

## 데이터 스키마 (schemas.py)

```python
@dataclass
class Diagnosis:
    text: str
    rubrics: dict[str, float]        # score_rubrics 결과
    features: dict[str, float]       # extract_features 결과
    weak_rubrics: list[str]          # 점수 낮은 rubric (규칙으로 선별)

@dataclass
class Candidate:
    action_type: str                 # ADD_DETAIL/DELETE_OR_FOCUS/COMPRESS/RESTRUCTURE/STYLE_REFINE/STOP
    target_rubric: str
    target_span: str                 # 수정 대상 (문장/구절 텍스트 또는 위치)
    instruction: str                 # 수정 지시
    # patch 적용 후 채워짐:
    new_text: str | None = None
    patch: dict | None = None        # {operation, target_span, before, after, reason}

@dataclass
class Transition:
    action_type: str
    target_rubric: str
    target_gain: float               # 수정 후 - 전, target_rubric 점수
    non_target_drop: float           # 비target rubric 중 최대 하락
    target_gap_reduction: float      # target feature가 기준에 가까워진 정도
    evidence_match: float            # 0~1, 수정 위치가 weak 지점과 맞나
    edit_ratio: float                # 변경 토큰 / 전체 토큰
    goal_preservation: float         # 0~1, 원문 의도 유지 (임베딩 유사도)
    # emb_sim 등은 이번 MVP에선 goal_preservation로 대체 가능
```

---

## 각 모듈 상세

### diagnose.py
```python
def diagnose(text: str) -> Diagnosis:
    rubrics = score_rubrics(text)
    features = extract_features(text)
    weak = select_weak_rubrics(rubrics)   # 하위 N개 또는 임계 이하
    return Diagnosis(text, rubrics, features, weak)
```
- `select_weak_rubrics`: 단순 규칙 (점수 낮은 순 상위 N개). 학습 없음.

### propose.py
```python
def propose(diag: Diagnosis, n_per_action: int = 1) -> list[Candidate]:
    # LLM에 진단 결과(weak_rubrics, 관련 feature)를 주고
    # action_type별로 후보를 JSON으로 생성하게 함
    # add 편향 방지: action_type을 명시적으로 펼쳐 각각 요청
```
- **출력은 반드시 스키마 검증된 JSON**. 파싱 실패 시 재시도/스킵.
- STOP도 후보 중 하나로 표현 가능.
- 후보 다양성은 action taxonomy에서 나오게 (자유 생성 아님).

### patch.py
```python
def apply_patch(text: str, cand: Candidate) -> Candidate:
    # cand.instruction을 실제 텍스트 수정으로 반영 (LLM 사용)
    # 전체 rewrite 금지 - target_span 중심 최소 수정
    # before/after 기록 -> cand.patch, cand.new_text 채움
```
- reversible: 원문은 보존, 수정본은 별도.

### transition.py
```python
def compute_transition(before: Diagnosis, after: Diagnosis, cand: Candidate) -> Transition:
    # before/after의 rubrics, features 비교로 각 신호 계산
    # goal_preservation: 고정 임베딩(예: 한국어 SBERT)으로 원문 vs 수정본 유사도
```
- 임베딩 모델은 고정, 로드만. 없으면 이번 MVP에선 간단한 대체(예: 자카드/코사인 on features)로 두고 TODO 표시.

### heuristic.py
```python
def heuristic_score(t: Transition) -> float:
    # 학습 아님. 명시적 가중합.
    return (t.target_gain
            + t.target_gap_reduction
            + t.evidence_match
            + t.goal_preservation
            - t.non_target_drop
            - t.edit_ratio)

def select(cands_with_t: list[tuple[Candidate, Transition]], cfg) -> dict:
    # hard constraint 위반 후보 reject
    # 남은 것 중 최고 점수 accept
    # 최고 점수가 accept 임계 미만이면 stop
    # 반환: {decision: accept/reject_all/stop, chosen, scores, logs}
```
- **이 휴리스틱은 baseline이자 weak-label 생성기다. "학습된 reward model"이 아님.**

### loop.py
```python
def one_step(text: str, cfg) -> dict:
    diag = diagnose(text)
    cands = propose(diag)
    results = []
    for c in cands:
        c = apply_patch(text, c)
        after = diagnose(c.new_text)
        t = compute_transition(diag, after, c)
        results.append((c, t))
    decision = select(results, cfg)
    return {"before": diag, "candidates": results, "decision": decision}
```
- multi-step은 이번 범위 밖. `one_step`만 확실히.

---

## 설정 (configs/heuristic.yaml)
```yaml
weak_rubric_top_n: 3
n_per_action: 1
accept_threshold: 0.0        # 개발 데이터로 나중에 튜닝
hard_constraints:
  non_target_drop_max: 1.0
  edit_ratio_max: 0.5
  goal_preservation_min: 0.7
  evidence_match_min: 0.3
```
- 모든 임계값은 상수로 두고, 나중에 개발 데이터로만 튜닝.

---

## 실행/테스트

```bash
# 하드코딩 글 1개로 한 바퀴
python scripts/run_mvp.py --text-file sample_essay.txt

# 스모크 테스트
pytest tests/test_mvp_loop.py
```

`run_mvp.py`는 다음을 출력:
- 진단 결과 (rubric, weak)
- 생성된 후보들 (action_type, instruction)
- 각 후보의 transition feature + 휴리스틱 점수
- 최종 decision (accept/reject/stop)과 이유

---

## 먼저 사용자에게 확인할 것 (구현 시작 전)

Codex는 코드를 짜기 전에 다음을 사용자에게 확인/요약:
1. `score_rubrics`, `extract_features`의 **실제 함수명·경로·반환형** (스텁을 실제 호출로 교체하려면 필요)
2. LLM API 종류/키 주입 방식
3. 임베딩 모델 사용 가능 여부 (없으면 goal_preservation 임시 대체)
4. rubric 이름 목록 (8개) / feature 이름 목록

확인 전에는 위 인터페이스를 **스텁(dummy 반환)** 으로 두고 loop가 끝까지 도는 것부터 완성한다.

---

## 구현 원칙
- 모든 LLM 출력은 스키마 검증된 JSON.
- 원문은 절대 파괴하지 않음 (reversible patch).
- 각 모듈에 최소 스모크 테스트.
- 학습·DB·drift·rollback은 절대 이번에 넣지 않음 (다음 단계).
- 먼저 스텁으로 loop를 완성하고, 그 다음 실제 함수로 교체.
