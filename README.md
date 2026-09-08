# FEAK-Agent

한국어 글쓰기의 수정 성공과 내용 보존을 검증하고, 반복 수정의 채택·재계획·복구·종료를 연구하는 저장소다.

## 현재 연구 기준

최종 문서는 로컬의 [paper_docs](feak_tc_docs/paper_docs/)에 있다.

| 문서 | 내용 |
|---|---|
| [서론·관련연구](feak_tc_docs/paper_docs/FEAK_TC_INTRO_RELATED_2026-09-09.md) | 연구 배경, 문제와 선행연구 |
| [최종 방법론](feak_tc_docs/paper_docs/FEAK_TC_METHOD_FINAL.md) | FEAK·RAG·Planner·RV·경로 가드·Controller |
| [RV 데이터 생성 및 학습 지침](feak_tc_docs/paper_docs/FEAK_TC_RV_DATA_GENERATION.md) | 수정 요구, 후보 구성, 두 축 라벨, 분할과 학습 |

새 학습 대상은 RV 하나이며, 출력은 목표 달성 `target_fulfillment`과 내용 보존 `preservation`의
두 축이다. RAG는 계획을 위한 우수 사례를 제공하고, 경로 가드는 기존 LLM으로 누적 훼손을 확인한다.

논문 설계와 구현 상태는 구분한다. 현재 실행 가능한 제어 코드는 한 단계 휴리스틱 MVP이며,
`feak_tc/rv/`와 RV 실행 스크립트는 기존 4축 파일럿의 생성·검수·재사용 도구다.
최종 2축 schema와 전문가 피드백 기반 생성, RV 학습, RAG 및 반복 경로 제어의 통합은 후속 구현 대상이다.

## 폴더 안내

```text
feak_tc_docs/paper_docs/  최종 논문 문서 — 로컬 보관
feak_tc_docs/CODEX.md     작업 지침과 문서 우선순위
feak_tc_docs/docs/        기존 구현 스펙과 참고문헌 확인 기록
feak_tc_docs/중간정리/    과거 실험 근거와 데이터 현황
feak_tc/diagnose/         FEAK·Kanana·stub 진단 연결
feak_tc/mvp/              한 단계 수정·patch·품질 평가·휴리스틱
feak_tc/rv/               기존 RV 파일럿 데이터 도구
feak_tc/corruption/       기존 corruption 생성·검사·분석 라이브러리
feak_tc/data/             AI-Hub JSON 정규화
feak_tc/schemas/          원본 데이터 schema
src/apps/                기존 한국어 분석·채점 코드
scripts/                 현재 실행 및 데이터 재사용 진입점
configs/                 실행 설정
tests/                   회귀 테스트
data/                    원본 데이터 — 로컬 보관
experiments/results/     생성 데이터·검수 자료·실험 결과 — 로컬 보관
```

[기존 데이터 현황](feak_tc_docs/중간정리/RV_DATA_STATUS_2026-09-08.md)은 보유 산출물을 찾는 데 사용한다.
[실험 기록 안내](feak_tc_docs/중간정리/README.md)의 과거 계획은 현재 실행 지시가 아니다.

## 실행과 검증

환경별 의존성은 기존 환경에 맞게 선택한다.

- `requirements.txt`: core/dev
- `requirements-kanana.txt`: Kanana 실행
- `requirements-legacy.txt`: 기존 UKTA/KoBERT 실행

테스트:

```bash
python -m pytest -q
```

Stub을 사용한 한 단계 MVP 확인:

```bash
HF_HUB_OFFLINE=1 python scripts/run_mvp.py \
  --diagnoser stub \
  --text "인권은 인간이 가지는 기본적인 권리이다. 우리는 서로의 권리를 존중해야 한다." \
  --proposer-mode deterministic \
  --patcher-mode deterministic \
  --surface-normalizer off
```

이 실행은 학습된 RV나 최종 반복 Controller의 실행이 아니다.
실제 채점기 연결은 [Diagnoser Integration](docs/DIAGNOSER_INTEGRATION.md)을 참고한다.

## 보존한 데이터 도구

| 용도 | 대표 진입점 |
|---|---|
| 기존 corruption 출처·품질 점검 | `audit_corruption_source_selection.py`, `audit_corruption_generation.py` |
| 원문 선별·chain 생성·측정 | `select_corruption_sources.py`, `run_corruption_chains.py`, `measure_corruption_chains.py` |
| 기존 학습 풀 재조립 | `build_corruption_rulev5_dataset.py`, `build_corruption_training_pool.py` |
| 기존 RV 후보 생성·재구성 | `build_rv_data_pilot.py`, `rebuild_rv_data_pilot_v2.py` |
| 기존 RV 검수·재라벨링 | `evaluate_rv_pilot_llm_judges.py`, `relabel_rv_data_pilot_v2.py` |
| 원본 진단·실행 결과 분석 | `smoke_diagnose.py`, `analyze_mvp_logs.py`, `compare_mvp_logs.py` |

표의 파일은 모두 `scripts/` 아래에 있다. 도구를 보존했다는 사실이 데이터 재생성이나
같은 후보의 반복 재판정을 재개한다는 뜻은 아니다. 새 데이터는 최종 생성 지침에 맞춰 준비한다.

## 로컬 자료와 정리 이력

데이터, 결과, 모델, 비밀키와 논문 원고는 Git에 포함하지 않는다.
`paper_docs/`와 SAC 원고는 로컬 보관 자료이므로 새 clone에는 자동으로 내려오지 않는다.
API 키는 기존 `.env` 또는 `secrets/`를 사용한다.

구버전 지시서와 일회성 실행 파일의 정리 내역·복구 방법은
[정리 기록](docs/CLEANUP_2026-09-08.md)에 있다.
