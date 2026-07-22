# DELETE_OR_FOCUS 핵심어 정의 문장 보호 가드

작성일: 2026-07-22
브랜치: `fix/dof-definition-guard`
관련: `STAGE_A_BGE_100_RESULTS_2026-07-20.md`의 "다음 작업 1. proposer targeting 개선"

## 문제

bge 07-20 run에서 train_631이 글 첫 문장(핵심어 "인권"의 정의 문장)을
DELETE_OR_FOCUS target으로 골라 accept됨. 패치 자체는 깨끗해서 validity
guard·자동 스캔에 안 걸리지만 내용상 손실. 경로는 LLM proposer가 직접
정의 문장을 target_span으로 지정한 것 (해당 레코드는 question=None이라
질문 핵심어 fallback도 없었음).

## 구현 (3중 방어)

1. **탐지** (`targeting.py`): 문장이 "X(이)란 …" 패턴 또는 정의 마커
   (의미한다/뜻한다/개념이… — `configs/targeting.yaml`의
   `definition_markers`)로 용어 X를 정의하고, X가 (a) 질문에 등장하거나
   (b) 본문의 다른 문장 `definition_min_repeats`(기본 2)개 이상에서
   재등장하는 핵심어면 보호 대상.
2. **휴리스틱 targeting**: DELETE_OR_FOCUS 점수에서 보호 문장에
   `definition_penalty`(기본 4.0) 감점 → 항상 최하순위.
3. **LLM proposer 경로** (`propose.py`): LLM이 보호 문장과 겹치는 span을
   DOF target으로 반환하면 그 후보를 버리고 deterministic fill로 대체
   (`is_protected_deletion_span`). 프롬프트에도 금지 규칙 1줄 추가.

## 검증

- `tests/test_mvp_loop.py` 44개 전부 통과 (신규 4개: 탐지 양/음성,
  질문 핵심어 경로, DOF 랭킹 최하위, LLM 후보 거부→fill 대체).
- **bge 07-20 로그 소급 재현**: DOF 후보 100건 중 9건 차단.
  - 차단 9건 전부 핵심어(인권 8, 헌법 1) 정의 문장 — 육안 확인 결과
    오탐 없음.
  - accept됐던 DOF 8건 중 차단은 train_631 1건뿐 — 07-20 질적 검토에서
    깨끗한 삭제로 확인된 나머지 7건은 그대로 통과 (과잉 차단 없음).

## 남은 것 / 다음 작업

1. (선택) 100건 재실행으로 accept 분포 변화 확인 — 가드는 DOF 후보를
   deterministic fill로 바꾸므로 accept 수 변화는 크지 않을 것으로 예상.
2. 로드맵 다음 단계: corruption 데이터 생성 진입
   (`docs/SPEC_CORRUPTION_TVM.md`).
