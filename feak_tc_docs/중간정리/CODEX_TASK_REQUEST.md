# CODEX 작업 요청 — MVP 로그 신뢰성 마무리 (TVM 전 마지막 단계)

> 작성: 2026-07-10 (Claude Code). 이 문서를 CODEX에게 그대로 전달한다.

## 먼저 읽을 것 (순서대로)

1. `feak_tc_docs/중간정리/CLAUDE_REVIEW_RESPONSE.md` — 리뷰 요청에 대한 답변 + 적용된 수정
2. `feak_tc_docs/중간정리/STAGE_A_BEFORE_AFTER_20.md` — 같은 20건 재실행 비교 결과
3. `feak_tc_docs/중간정리/ACTION_GENERATION_REDESIGN.md` — action 생성 재설계 방향 (Stage B는 아직 착수 금지)

요약: validity filter / target_gain_min 1.0 / rubric 정규화 / elite-gap 재구현이
적용되어 gain≤0 선택률이 42%→0%가 됐다. 남은 병목은 채점기 노이즈다
(temperature 0.7 샘플링 앙상블, 설계 기본 m=20인데 지금 m=3으로 사용 중,
m=3 노이즈 추정 ±1.2 → gain=1 accept는 노이즈와 구분 불가).

## 작업 0 — 커밋

현재 working tree의 변경(validity.py, heuristic/transition/loop/batch 수정,
configs/elite_features.yaml, scripts 3개 신규, 중간정리 문서들)을 의미 단위로
나눠 커밋하라. 기능 변경과 문서를 섞지 말 것.

## 작업 1 — m-sweep 노이즈 측정 (m 결정의 근거 만들기)

`scripts/measure_scorer_noise.py`를 `--kanana-m 3 / 10 / 20`으로 각각 실행하고
(`--n-essays 5`로 올려서), m별 rerun/synonym max|diff| 표를
`experiments/results/scorer_noise_sweep.json` + 중간정리 md로 정리하라.
GPU는 `nvidia-smi`로 빈 것 확인 후 사용 (주로 device 3).

- 판단 기준: rerun noise가 ±0.5 이하로 떨어지는 최소 m을 보고하라.
- m 최종 결정은 사용자가 한다. 결정 전까지 batch 기본값을 바꾸지 말 것.

## 작업 2 — goal_preservation을 SBERT 임베딩으로 교체

- `feak_tc/mvp/transition.py`의 goal_preservation(현재 토큰 보존율)을
  한국어 SBERT 계열 고정 임베딩 코사인 유사도로 교체.
  모델 후보: `jhgan/ko-sbert-sts` 또는 `snunlp/KR-SBERT-V40K-klueNLI-augSTS`.
  로드는 lazy, 임베딩 모델 미설치 시 기존 토큰 방식 fallback + 로그에 방식 기록.
- emb_sim도 같은 임베딩으로 계산 (drift 추적과 공용 — PROJECT_CONTEXT 확정 결정).
- `configs/heuristic.yaml`의 goal_preservation_min은 임베딩 기준 재보정이
  필요하므로 일단 0.7 유지, 20건 로그에서 임베딩 값 분포를 뽑아 보고만 하라.
- 기존 토큰 방식 함수는 삭제하지 말 것 (transition feature 재계산 스크립트에서 필요).

## 작업 3 — targeting.py 주제 마커 일반화

`feak_tc/mvp/targeting.py`의 `_TOPIC_MARKERS`/`_EXAMPLE_MARKERS`/`_STYLE_RED_FLAGS`에
특정 에세이(인권 주제)에 과적합된 하드코딩이 있다 ("5.18", "맛있는 밥" 등).

- 주제 의존 마커를 제거하고, 주제 무관 신호로 대체하라:
  질문(question) 텍스트와의 어휘 겹침, 문장 길이, 연결어 유무, 어미 반복 등
  이미 있는 29개 feature/stub 유틸을 활용.
- 남길 상수는 `configs/targeting.yaml`로 빼라 (코드 하드코딩 금지 관례).
- `tests/test_mvp_loop.py`의 targeting 관련 테스트가 있으면 갱신, 없으면 추가.

## 작업 4 — reject_all 7건 검토 자료 추출

`mvp_stage_a_20_v2.jsonl`에서 decision=reject_all인 7건의
(에세이 앞부분 300자, before rubrics, 각 후보의 action/gain/reject_reasons)를
사람이 훑기 좋은 md 표로 추출해 중간정리 폴더에 저장하라.
판단(gain_min이 과보수인지)은 사용자가 한다.

## 금지 사항 (변경하려면 사용자 승인)

- action taxonomy, transition feature 정의, PROJECT_CONTEXT.md 확정 결정
- Stage B(action prior 학습), corruption/TVM 착수
- `non_target_drop_max` / `target_gain_min` 값 변경 (m 결정 후에 조정)
- 채점기(essay_scoring_llm) 코드 수정

## 완료 기준

- pytest 전체 통과 (현재 26개 + 신규)
- 작업 1/4의 산출물이 중간정리 폴더에 md로 존재
- 작업 2/3 적용 후 20건 스모크(stub, deterministic)가 에러 없이 완주
