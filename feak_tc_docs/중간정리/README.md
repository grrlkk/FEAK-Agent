# 실험 기록 안내

이 폴더는 지난 실험의 설정·수치·판단 근거를 남긴 보고서 모음이다.
결과를 확인하고 재현 조건을 찾는 용도이며, **현재 실행 지시가 아니다.**

각 보고서의 '다음 작업', '남은 과제' 항목은 작성 시점의 계획이다.
현재 방향은 [작업 지침](../CODEX.md)과 로컬 `paper_docs/`의 최종 문서를 따른다.
보유 데이터의 현재 위치는 [RV 데이터 진행 현황](RV_DATA_STATUS_2026-09-08.md)에 있다.

## 기록 목록

### RV — Revision Verifier

| 문서 | 내용 |
|---|---|
| [RV 데이터 진행 현황](RV_DATA_STATUS_2026-09-08.md) | 2026-09-08 점검. 보유 산출물과 과거 완료 여부 |
| [RV 파일럿 v2 결과](RV_DATA_PILOT_V2_2026-09-02.md) | 재생성·재라벨링 후 227개 후보 |
| [RV 3-LLM 블라인드 검증](RV_THREE_LLM_BLIND_REVIEW_2026-09-01.md) | 파일럿 전수 교차 판정 |
| [RV 파일럿 결과](RV_DATA_PILOT_2026-08-31.md) | 최초 파일럿 |

### Corruption 데이터

| 문서 | 내용 |
|---|---|
| [rule v5 최종 1,000쌍](CORRUPTION_RULEV5_1000_RESULTS_2026-08-20.md) | 학습 풀 확정 결과 |
| [학습곡선](CORRUPTION_LEARNING_CURVES_2026-08-20.md) | feature/text 학습곡선과 생성 중단 판단 |
| [rule v5 검색형 OFFTOPIC](CORRUPTION_RULEV5_RETRIEVAL_RESULTS_2026-08-18.md) | 검색 기반 OFFTOPIC 생성 |
| [G2 2-LLM 블라인드 평가](G2_TWO_LLM_BLIND_REVIEW_2026-08-12.md) | rule v3 선호 평가 |
| [SHUFFLE_FLOW 민감도](SHUFFLE_SENSITIVITY_2026-08-12.md) | 문장 순서 훼손 반응과 존폐 결정 |
| [rule v3 STEP1~3 결과](CORRUPTION_RULEV3_STEP1_3_RESULTS_2026-07-29.md) | 초기 규칙 구현·검증 |
| [생성 운영 결정](CORRUPTION_GEN_DECISIONS_2026-07-22.md) | 스펙이 비워둔 파라미터 확정 |
| [정의 문장 보호 가드](DOF_DEFINITION_GUARD_2026-07-22.md) | DELETE_OR_FOCUS 가드와 검증 |

### MVP와 채점기

| 문서 | 내용 |
|---|---|
| [MVP 최종 보고서](MVP_FINAL_REPORT_2026-07-23.md) | 한 단계 휴리스틱 구현 결과 |
| [Target-gain ablation과 2인 평가](TARGET_GAIN_ABLATION_AND_HUMAN_REVIEW_2026-07-29.md) | ablation·사람 평가 진행 |
| [Stage A BGE-M3 100건](STAGE_A_BGE_100_RESULTS_2026-07-20.md) | 3-run 비교 |
| [임베딩 모델 비교](EMBEDDING_MODEL_EVAL_2026-07-19.md) | BGE-M3 채택 근거 |
| [채점기 노이즈 m-sweep](SCORER_NOISE_M_SWEEP.md) | Kanana 반복 채점 분산 |
| [soft threshold sweep](SOFT_THRESHOLD_SWEEP_20.md) | 연속 점수 threshold 검토 |

## 본문이 언급하는 제거된 파일

2026-09-08 정리로 아래 파일을 저장소에서 제거했다. 보고서 본문은 작성 시점 기록이므로
그대로 두었고, 이름이 나오면 다음 대체 문서나 Git 이력을 보면 된다.
제거 목록과 복구 방법은 [정리 기록](../../docs/CLEANUP_2026-09-08.md)에 있다.

| 본문에 나오는 이름 | 현재 참고할 곳 |
|---|---|
| `docs/PROJECT_CONTEXT.md` | 로컬 `paper_docs/FEAK_TC_METHOD_FINAL.md` |
| `docs/FEAK-TC_평가구조_RevisionVerifier_정리.md` | 로컬 `paper_docs/FEAK_TC_METHOD_FINAL.md` |
| `docs/FEAK_TC_RV_METHOD_2026-09-08.md` | 로컬 `paper_docs/FEAK_TC_METHOD_FINAL.md` |
| `docs/CODEX_01_RV_데이터파일럿_작업지시.md` | 로컬 `paper_docs/FEAK_TC_RV_DATA_GENERATION.md` |
| `docs/TASK_데이터생성_단계.md` | 로컬 `paper_docs/FEAK_TC_RV_DATA_GENERATION.md` |
| `HANDOFF_CODEX.md` | 없음. TVM 추가 개발은 종료했다 |
| `scripts/validate_definition_guard.py` | 없음. 결과는 위 가드 보고서에 있다 |
