# Stage A span 고정 patcher + guard 100건 재실행 결과

작성일: 2026-07-19
로그: `experiments/results/mvp_stage_a_100_spanfix_m10.jsonl`
비교 대상(pre): `mvp_stage_a_100_soft_m10.jsonl` (2026-07-16 baseline)

## 실행 조건

baseline과 동일 (같은 100건 shard, Kanana m=10, gpt-4o-mini, GPU 1~3).
차이는 두 가지뿐이다.

1. span 고정 patcher (`feak_tc/mvp/patch.py`): 수정 위치를 target_span
   offset으로 코드가 고정. 삭제는 기계적 제거 + 접속어 수선(dangling
   opener 감지 시에만 LLM 1회).
2. validity guard 3종 (`feak_tc/mvp/validity.py`): fabricated_numbers,
   fabricated_claim_marker, near_duplicate_sentence.

100/100 status ok, 오류 0건. 실행 약 2.7시간.

## Decision 비교

| | pre (07-16) | post (07-19) |
|---|---:|---:|
| accept | 52 | 48 |
| reject_all | 48 | 52 |

선택 action 분포:

| action | pre | post |
|---|---:|---:|
| ADD_DETAIL | 24 | 22 |
| DELETE_OR_FOCUS | 12 | 5 |
| COMPRESS | 5 | 7 |
| RESTRUCTURE | 5 | 6 |
| STYLE_REFINE | 6 | 8 |

## 결함 비교 (핵심 결과)

07-17 질적 검토와 동일한 자동 스캔 + 수동 검토 기준.

| 결함 유형 | pre (accept 52 중) | post (accept 48 중) |
|---|---:|---:|
| DOF 위치 오류·비문·문맥 붕괴 | 11~12 / 12 | 0~1 / 5 |
| 통계·수치 환각 | 3 | 0 |
| 문장 중복 | 1 | 0 |
| 자동 스캔 플래그 합계 | 21 | **0** |

- DOF 채택 5건 수동 전수 검토: 4건 깨끗한 성공(train_993, 1035, 1040은
  접속어 수선까지 자연스러움, 816은 구조적으로 정확), 1건 경미한 어색함
  (train_1021 — 삭제 후 "크게 ...로 나뉜다"의 주어 소실, dangling opener
  목록에 없는 시작이라 수선이 발동하지 않음).
- 전체 후보(500) 기준 DOF 후보 100개 중 target_span이 삭제되지 않은
  경우 0건. baseline의 지배적 실패 유형이 구조적으로 소멸했다.
- 접속어 수선은 전체 실행에서 다수 발동했고(중간 체크 시점 40건 중
  14회), 수선 결과는 검토 표본에서 모두 자연스러웠다.

## Guard 발동 내역 (전체 후보 500개 기준)

| reject reason | 건수 |
|---|---:|
| validity:fabricated_numbers | 8 |
| validity:fabricated_claim_marker | 2 |
| validity:near_duplicate_sentence | 1 |

patcher 프롬프트에 수치·통계 생성 금지를 명시했는데도 환각이 8건
발생했다 — 프롬프트 견제만으로는 부족하고 guard가 실제로 필요함이
재확인됐다. 발동한 guard는 전부 해당 후보를 거부시켰고, accept 48건에는
환각·중복이 남지 않았다.

## 해석

- accept 52→48은 겉보기 후퇴지만, pre의 52건 중 최소 15건이 결함이었던
  것을 감안하면 실질 유효 accept는 ~37 → 47~48로 **개선**이다.
- `goal_preservation < 0.9`가 11/48로 늘었는데, 이제는 결함 신호가
  아니라 "정상적인 삭제·압축으로 텍스트가 실제로 변했다"는 신호에
  가깝다. gp 지표를 결함 검출 용도로 쓰지 않아야 한다.
- 남은 품질 이슈는 patcher가 아니라 **proposer의 target 선택** 문제다.
  예: train_816은 질문("인권의 뜻과 특징")의 '뜻'을 정의하는 첫 문장을
  삭제 target으로 골랐다. 패치는 정확했지만 내용상 손해일 수 있다.

## 다음 작업

1. dangling opener 목록에 "크게", "이 중" 등 수량·분류 시작 표현 보강
   또는 삭제 후 다음 문장의 주어 유무 검사 (train_1021 유형)
2. proposer targeting 개선: 질문 핵심어를 정의하는 문장은 삭제 target
   후보에서 제외
3. 한국어 SBERT 로컬 준비 후 token_fallback similarity 교체
4. corruption 데이터 생성 단계로 진행 (`docs/SPEC_CORRUPTION_TVM.md`)
