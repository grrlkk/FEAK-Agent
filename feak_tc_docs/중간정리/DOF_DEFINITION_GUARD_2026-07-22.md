# DELETE_OR_FOCUS 핵심어 정의 문장 보호 가드

작성일: 2026-07-22 (같은 날 out-of-sample 검증으로 규칙 강화)
브랜치: `fix/dof-definition-guard` (PR #5)
관련: `STAGE_A_BGE_100_RESULTS_2026-07-20.md`의 "다음 작업 1. proposer targeting 개선"

## 가드의 위상 (중요)

이 가드는 **corruption/TVM 학습 데이터 클리닝용 임시 비계(scaffolding)**다.
"어떤 문장이 지우기 아까운가"는 본질적으로 TVM이 데이터에서 배워야 할
의미 판단이므로, 이 하드 룰은 영구 시스템 컴포넌트가 아니다:

- **지금 필요한 이유**: Stage A 로그가 TVM 학습 데이터인데, "정의 문장
  삭제가 accept됨" 같은 오염이 섞이면 TVM이 그걸 좋은 transition으로
  학습한다. 가드는 성능 튜닝이 아니라 학습 데이터 오염 방지다.
- **TVM 학습 후**: 하드 룰 대신 "삭제 span의 주제 중심성" 류 transition
  feature로 전환해 TVM이 판단하게 하는 것을 검토한다 (feature 추가는
  사용자 승인 + ablation 필요 — No Feature Creep 규칙). hand-crafted
  guard vs learned TVM이 같은 오류를 잡는지가 자연스러운 ablation.
- validity.py의 구조 가드(문장 파편·숫자 조작·중복)와 구분할 것: 그쪽은
  영구 룰이어도 되는 구조 검사이고, 이 가드는 의미 영역을 침범한 예외.

## 문제

bge 07-20 run에서 train_631이 글 첫 문장(핵심어 "인권"의 정의 문장)을
DELETE_OR_FOCUS target으로 골라 accept됨. 패치 자체는 깨끗해서 validity
guard·자동 스캔에 안 걸리지만 내용상 손실. 경로는 LLM proposer가 직접
정의 문장을 target_span으로 지정한 것 (해당 레코드는 question=None이라
질문 핵심어 fallback도 없었음).

## 구현 (3중 방어)

1. **탐지** (`targeting.py`): 문장이 용어 X를 정의하고 X가 핵심어면 보호.
   - 정의 판정: "X(이)란" head가 있고 문장이 정의 형태(어휘 마커 또는
     "…이다" 종결 / "라고 할 수" 구문)이거나, "X는/은" head + 어휘 마커
     (의미한다/뜻한다/정의한다/개념이다… — `configs/targeting.yaml`의
     `definition_markers`).
   - 핵심어 판정: X가 질문에 등장하거나 본문의 다른 문장
     `definition_min_repeats`(기본 2)개 이상에서 재등장.
2. **휴리스틱 targeting**: DELETE_OR_FOCUS 점수에서 보호 문장에
   `definition_penalty`(기본 4.0) 감점 → 항상 최하순위.
3. **LLM proposer 경로** (`propose.py`): LLM이 보호 문장과 겹치는 span을
   DOF target으로 반환하면 그 후보를 버리고 deterministic fill로 대체
   (`is_protected_deletion_span`). 프롬프트에도 금지 규칙 1줄 추가.

## 검증 1 — in-sample (bge 07-20 로그 100건 소급)

- DOF 후보 100건 중 6건 차단, 전부 핵심어 정의 문장. accept됐던 DOF
  8건 중 차단은 train_631 1건뿐 — 07-20 질적 검토에서 깨끗한 삭제로
  확인된 7건은 통과 (과잉 차단 없음).
- `tests/test_mvp_loop.py` 44개 전부 통과 (신규 4개).

## 검증 2 — out-of-sample (`scripts/validate_definition_guard.py`)

가드 설계에 쓰인 100건은 인권 주제 편중이라, **안 본 레코드 328편
(train.jsonl에서 164개 질문 전체에 걸쳐 층화 표집, 시드 20260722)**으로
오탐을 별도 확인했다.

초기 규칙은 오탐 2패턴이 발견돼 강화함:
- "-란"이 명사 일부/관용구인 경우 (수정**란** 처리, 흑인이**란** 이유로)
  → "X란" head는 문장이 정의 형태일 때만 인정
- 마커 "정의"의 부분문자열 함정 (정의(justice) 주제 글에서 "정의로운
  행동…" 문장이 대량 매칭) → 마커를 정의한다/정의된다 등으로 구체화,
  "라고 할 수"는 결론 문장을 잡길래 X란 패턴의 보조 증거로 강등

강화 후 최종 수치 (5,812문장):

| | question 없음 | question 있음 |
|---|---:|---:|
| 보호 문장 비율 | 0.57% (33) | 0.65% (38) |
| DOF top-1이 바뀌는 에세이 | 2/328 | 11/328 |

flagged 38건 육안 전수 검토: 사실상 전부 진짜 핵심어 정의 문장
(원주각·대푯값·변이·채식·국제기구·법이란…), 경계 사례 1~2건, 명백한
오탐 0건. 개입 범위가 작고(에세이의 3% 미만) 정밀도가 높음을 확인.

## 다음 작업

1. **corruption 데이터 생성 진입** (`docs/SPEC_CORRUPTION_TVM.md`) —
   targeting 개선까지 끝나 Stage A 진입 조건 충족.
2. (선택) 다음 100건 재실행 시 이번에 안 쓴 레코드로 돌려 accept 분포
   재확인.
3. TVM 학습 후 이 가드의 transition feature 전환 여부 재논의 (위 "위상"
   섹션).
