# FEAK-TC 데이터 생성 단계 지시 (Codex용)

> 지금 단계 목표: **학습 데이터를 만드는 것**에만 집중한다.
> TVM·Planner의 모델 구조/학습 방식은 이 단계에서 확정하지 않는다 (데이터가 준비된 뒤 별도 논의).
> corruption 생성 방식은 유지한다. 대량 생성은 GBM sanity 통과 후에만. 커밋/푸시는 사용자 승인 시에만.

---

## 확정 사항 (변경 금지)

- corruption 파이프라인(operator 기반 생성 → 재측정 → noise margin 0.3 채택)은 **유지**.
- operator↔rubric 매핑 유지:
  - DELETE_SPECIFICS → content_2
  - SHUFFLE_FLOW → organization_1
  - INSERT_OFFTOPIC → organization_2
  - INJECT_LEX_REPEAT → expression_1
- feature는 생성 목표가 아니라 사후 검증용 (유지).
- G2 AI 감사(94%)는 사람 평가가 아니므로 "사람 선호 일치율"로 보고하지 않는다. 상태 = pending_human_review.

---

## STEP 1 (필수) — 어법 오류를 corruption chain에서 분리

- 이유: 실행 파이프라인에서 기계적 어법(조사·띄어쓰기·철자)은 **Planner 앞단의 맞춤법 검사기 API가 먼저 처리**한다.
  따라서 Planner·TVM이 실제로 보는 글에는 기계적 어법 오류가 거의 없다.
  그런데 corruption chain 매 step에 INJECT_GRAMMAR_ERR를 넣으면, 학습 분포와 실행 분포가 어긋난다(train/test mismatch).
- 조치:
  - **INJECT_GRAMMAR_ERR를 corruption chain에서 제거.** chain은 content/organization/expression(표현) 축 훼손만 포함.
  - 기계적 어법 오류 샘플은 **chain 밖에서 별도**로 소량만 생성 → `D_raw → 맞춤법 API → D0` surface correction 검증용으로만 사용.
  - expression 축에서 chain에 남기는 것은 **고차원 표현 문제**(어휘 반복 INJECT_LEX_REPEAT 등 맞춤법 API가 못 잡는 것)뿐.

## STEP 2 (필수) — 보존검사 2개 강화 후 소량 재생성

G2 감사가 찾은 결함. 방향이 뒤집힌 라벨을 만들 수 있어 noise margin으로 못 거른다.

- **DELETE_SPECIFICS**: 지정 span 외 편집 금지. 삭제 대상 span 밖 문장이 재작성·교정·삭제되면 샘플 거부.
  (구현: 원문 vs corrupt문 문장 단위 diff → span 외 변경 있으면 reject 로그 후 폐기.)
- **SHUFFLE_FLOW**: 단순 위치 변경만으로 채택 금지. 접속어·지시어·시간/인과 의존관계가 실제로 깨진 경우만 유효 훼손으로 채택.
- 위 두 검사 + STEP 1 적용해 **동일 30편으로 재생성** → 채택률 변화 보고.

## STEP 3 (대량 생성 관문) — GBM sanity

"이 corruption 쌍이 학습 가능한 신호인가"를 싸게 확인. 실패 시 대량 생성으로 가지 않는다.

- 입력: 채택된 **인접 transition 쌍**(stage k→k+1)의 transition feature만 (텍스트 없이).
- 모델: LightGBM pairwise. 라벨: 덜 훼손 > 더 훼손.
- 분할: **에세이 단위** train/test (같은 원문 파생 쌍이 split 넘지 않게).
- 판정: held-out pairwise accuracy가 랜덤(50%) 대비 유의하게 높은가.
  - 높음 → STEP 4로. 애매(47쌍이 적을 수 있음) → STEP 4에서 규모 늘려 재시도.
- 함께 보고: operator별·stage gap별 정확도.

## STEP 4 (규모 결정) — 학습곡선으로 목표 규모 산출

목표 데이터 규모는 추측하지 말고 **곡선의 포화 지점**으로 정한다.

- 에세이 수를 늘려가며(예: 원본 100 / 200 / 300편분) 인접쌍 생성 → GBM 성능 기록.
- "쌍 수 vs pairwise accuracy" 곡선에서 평평해지는 지점 = 목표 규모.
- 원본 4만 건이 있으므로 원료는 충분. **몇 편을 돌릴지만** 곡선으로 결정해 보고.

## STEP 5 (병렬) — 사람 블라인드 평가 준비

- `human_review_50.jsonl`(operator·정답 숨긴 A/B view) 유지·정리.
- 사람 2명 이상 독립 A/B 평가 → 불일치는 합의/제3자. 통과 기준 ≥ 70%.
- 이 평가 자체를 AI로 대체하지 않는다. 통과 시 G2 official = passed.

---

## real-log 수집기 (STEP 3와 병렬로 코드만 준비)

- corruption은 "단일 축이 훼손된 깨끗한 상태"만 만든다. 실제 학생 글은 "여러 축이 동시에 약한" 상태다.
- 따라서 나중에 **실제 학생 글에 여러 action을 걸어보고 결과를 기록하는 real-log**가 추가로 필요하다
  (Planner의 우선순위 판단, TVM의 실전 분포 보정용).
- 이 단계에서는 **열거 모드 로그를 남기는 코드만 준비**해 두고, 실제 대량 수집은 데이터 규모 확정 후 진행.
- 지금 구현/논의 대상 아님. 코드 훅만 유지.

---

## 진행 순서 요약

```
STEP 1  INJECT_GRAMMAR_ERR를 chain에서 분리 (surface 검증용으로 chain 밖)     [필수]
STEP 2  DELETE_SPECIFICS / SHUFFLE_FLOW 보존검사 강화 → 30편 재생성           [필수]
STEP 3  GBM sanity (인접쌍 feature-only) → 랜덤 초과 확인                      [대량 생성 관문]
STEP 4  학습곡선(100/200/300편) → 목표 규모 산출                              [몇 건 필요한지]
STEP 5  사람 50쌍 블라인드 평가 (병렬)                                        [G2 공식 통과]
(병렬)  real-log 수집기 코드 훅만 준비 (실수집은 나중)
```

## 이 단계에서 하지 않는 것

- **TVM·Planner의 모델 구조/학습 방식 확정 금지** (데이터 준비 후 별도 논의).
- STEP 3 통과 전 대량 corruption 생성 금지.
- INJECT_GRAMMAR_ERR를 TVM 핵심 학습쌍으로 사용 금지.
- AI 감사(94%)를 사람 선호 일치율로 보고 금지.
- 사용자 승인 없는 커밋/푸시 금지. 커밋 전 git status 보고.
