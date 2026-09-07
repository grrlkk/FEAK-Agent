# FEAK-TC — Agent Guide (CLAUDE.md / AGENTS.md 겸용)

> 이 파일은 Claude Code / Codex CLI가 이 프로젝트에서 작업할 때 처음 읽는 문서다.
> 프로젝트 루트에 `CLAUDE.md`와 `AGENTS.md` 두 이름으로 복사해 두면 두 CLI 모두 자동으로 읽는다.

## 프로젝트 한 줄

FEAK-TC: 한국어 글쓰기 반복 수정을 **transition 단위로 평가·제어**하는 writing agent.
채점기(kanana-8B+LoRA, 학습 완료)가 진단하고, LLM이 수정 후보를 만들며,
RV가 목표 달성·내용 보존·편집 적절성을 검증해 FEAK 신호와 함께 controller에 제공하는 구조다.
**현재 신규 학습 대상은 RV이며, RV 학습과 반복 controller 통합은 아직 미완료다.**
기존 TVM 개발은 종료하고 결과·코드는 이전 실험 기록으로 보존한다.

## 문서 지도 (자세한 내용은 docs/)

| 문서 | 내용 | 언제 읽나 |
|---|---|---|
| `docs/FEAK_TC_RV_METHOD_2026-09-08.md` | RV 중심 현재 방법론·판정 원칙·평가 설계 | 항상 먼저, 과거 TVM 설계보다 우선 |
| `중간정리/RV_DATA_STATUS_2026-09-08.md` | 데이터 규모·실제 산출물·남은 검증·다음 작업 | 항상 함께 |
| `docs/PROJECT_CONTEXT.md` | 과거 TVM 설계의 연구 배경·결정 기록 | 역사적 맥락 확인 시 |
| `docs/IMPLEMENTATION_MVP.md` | 지금 구현할 MVP 루프 상세 지시 | MVP 작업 시 |
| `docs/SPEC_CORRUPTION_TVM.md` | 기존 corruption + TVM 학습 스펙 | 기존 실험 재현·출처 확인 시 |
| `중간정리/MVP_FINAL_REPORT_2026-07-23.md` | 완료된 MVP 구현·검증 최종 보고서 | MVP 결과 확인 시 |

## 현재 단계와 우선순위

1. [완료] MVP 루프 (`docs/IMPLEMENTATION_MVP.md`) + validity/heuristic 수리
2. [완료] 연속 점수(soft score) 전환과 threshold 검토
   (`중간정리/SCORER_NOISE_M_SWEEP.md`, `중간정리/SOFT_THRESHOLD_SWEEP_20.md`)
3. [완료] span 고정 patcher, validity guard, BGE-M3 기반 의미 보존 검증
   (`중간정리/STAGE_A_BGE_100_RESULTS_2026-07-20.md`)
4. [완료] proposer targeting 개선 — DELETE_OR_FOCUS 핵심어 정의 문장 보호 가드,
   in/out-of-sample 검증 (`중간정리/DOF_DEFINITION_GUARD_2026-07-22.md`)
5. [완료] corruption rule v3 — STEP1~3, GBM sanity 통과
   (`중간정리/CORRUPTION_RULEV3_STEP1_3_RESULTS_2026-07-29.md`)
6. [완료] G2 블라인드 검증 — 2-LLM + Claude, 라벨 결함 2건 발견
   (`중간정리/G2_TWO_LLM_BLIND_REVIEW_2026-08-12.md`)
7. [완료] corruption rule v4/v5 — 템플릿 아티팩트 제거, SHUFFLE_FLOW 강화,
   검색형 OFFTOPIC 전환, 최종 1,000쌍 품질 게이트 통과
   (`중간정리/CORRUPTION_RULEV5_1000_RESULTS_2026-08-20.md`)
8. [완료] feature-only GBM·BGE-M3 text pairwise 학습곡선 — 1,000쌍에서 포화,
   추가 생성 중단 (`중간정리/CORRUPTION_LEARNING_CURVES_2026-08-20.md`)
9. [완료·개발 종료] TVM 학습 및 실제 수정 20글/99후보 평가 — 합성 성능의 실제 전이는 충분히 입증되지 않음
10. [완료] RV v1/v2 데이터 파일럿 — 50글/300후보 중 227개 LLM 합의 subset 보존
    (`중간정리/RV_DATA_PILOT_V2_2026-09-02.md`)
11. **지금: 구체적인 수정 요구 + RV 역할에 맞는 소규모 비교 세트 + 사람 검수 양식 준비**
12. 그 다음: 라벨·분할·학습 설정 고정 → RV 최소 학습·실제 수정 평가 → controller/Trajectory Guard 통합

227개는 사람 검수 완료 데이터가 아니다. 기존 corruption 1,000쌍과 파일럿을 재사용하고,
유형을 맞추기 위한 반복 생성이나 같은 후보의 이유 없는 전수 LLM 재판정을 재개하지 않는다.
상세 종료 기준과 미확정 설계는 위 2026-09-08 문서를 따른다.

## 절대 규칙 (No Feature Creep)

- MVP 단계에서 **하지 않는 것**: TVM 학습, corruption 생성, drift/rollback, MongoDB, multi-step trajectory, FastAPI/프론트엔드, 채점기 재학습
- transition feature는 문서에 정의된 것만. 새 feature 추가는 사용자 승인 + ablation 근거 필요
- 원문 텍스트는 절대 파괴하지 않는다 (reversible patch)
- 모든 LLM 출력은 스키마 검증된 JSON
- 기존 채점기/자질 계산 코드는 수정하지 않는다 (호출만)

## 구현 관례

- Python, 작은 모듈 단위, 각 모듈에 스모크 테스트
- 실제 채점기/LLM 연결 전에 **스텁(stub)으로 루프 전체가 도는 것 먼저**
- 설정값(임계·가중치)은 configs/*.yaml 상수로, 코드에 하드코딩 금지
- 함수 시그니처가 불명확하면 추측하지 말고 사용자에게 확인

## Git/GitHub 작업 흐름

- 사용자가 명시적으로 요청하지 않는 한 `main`에 직접 push하지 않는다.
- 작업마다 topic branch를 만들고, 검증 후 해당 브랜치만 원격에 push한다.
- `main` 반영은 GitHub Pull Request를 통해 진행한다. 검증이 끝난 뒤 PR 생성과 병합까지
  에이전트가 `gh` CLI로 직접 수행한다 (GitHub 웹 불필요).
- PR 병합이 확인된 뒤에만 로컬 `main`을 동기화하고 작업 브랜치를 정리한다.

## 사용자 환경

- VSCode, 연구실 GPU (A6000 x6)
- 채점기: kanana-8B + LoRA (학습 완료, 함수 호출 가능) — 글 → 8 rubric 점수
- 자질(feature): 룰베이스 계산, 채점기와 독립 — 글 → 29개 언어학 feature
- 후보 생성 LLM: GPT-4o-mini 계열 API (키는 .env)
