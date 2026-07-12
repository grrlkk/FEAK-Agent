# 중간정리 — CODEX_TASK_REQUEST 작업 진행 기록

> 작성: 2026-07-11. `CODEX_TASK_REQUEST.md`의 작업 1~4 진행 결과.
> 작업 0(커밋)은 요청대로 건너뛰었다.

## 준수한 금지 사항

- `target_gain_min` / `non_target_drop_max` 값은 변경하지 않았다.
- action taxonomy는 변경하지 않았다.
- corruption / TVM / Stage B는 착수하지 않았다.
- `essay_scoring_llm` 채점기 코드는 수정하지 않았다.
- transition feature 필드는 추가하지 않았다.

## 작업 1 — m-sweep 노이즈 측정

실행:

```bash
python scripts/measure_scorer_noise.py --device-id 3 --n-essays 5 --kanana-m 3  --output experiments/results/scorer_noise_m3.json
python scripts/measure_scorer_noise.py --device-id 3 --n-essays 5 --kanana-m 10 --output experiments/results/scorer_noise_m10.json
python scripts/measure_scorer_noise.py --device-id 3 --n-essays 5 --kanana-m 20 --output experiments/results/scorer_noise_m20.json
```

산출물:

```text
experiments/results/scorer_noise_m3.json
experiments/results/scorer_noise_m10.json
experiments/results/scorer_noise_m20.json
experiments/results/scorer_noise_sweep.json
feak_tc_docs/중간정리/SCORER_NOISE_M_SWEEP.md
```

요약:

| m | rerun max\|diff\| | whitespace max\|diff\| | synonym max\|diff\| |
|---:|---:|---:|---:|
| 3 | 1.0 | 1.0 | 1.0 |
| 10 | 1.0 | 1.0 | 1.0 |
| 20 | 1.0 | 1.0 | 1.0 |

판정:

- 요청 기준인 `rerun max|diff| <= 0.5`를 만족하는 m은 없었다.
- 최종 rubric이 정수라서 m을 올려도 1점 단위 변동은 남을 수 있다.
- 따라서 m 결정 전 threshold를 건드리지 않는 현재 보류 방침이 맞다.

## 작업 2 — goal_preservation / emb_sim SBERT 교체

수정 파일:

```text
feak_tc/mvp/transition.py
```

변경:

- `goal_preservation`과 `emb_sim`을 같은 semantic similarity 값으로 계산하도록 변경했다.
- 한국어 SBERT 후보:
  - `jhgan/ko-sbert-sts`
  - `snunlp/KR-SBERT-V40K-klueNLI-augSTS`
- 로드는 lazy로 수행한다.
- `FEAK_EMBEDDING_MODEL` 환경변수로 모델명을 override할 수 있다.
- `sentence-transformers` 또는 모델 캐시가 없으면 기존 token 기반 fallback을 사용한다.
- 사용 방식은 transition feature를 늘리지 않고 `candidate.metadata["similarity"]`에 기록한다.

이번 환경 결과:

```text
method: token_fallback
reason: SBERT 모델이 로컬 캐시에 없고 Hugging Face 접속 불가
```

stub/deterministic 20건 smoke에서 fallback 분포:

```text
count: 100
min: 0.7049180327868853
max: 1.0
mean: 0.9759317097705957
```

주의:

- 실제 SBERT 분포는 모델이 로컬에 설치/캐시된 뒤 다시 측정해야 한다.
- `configs/heuristic.yaml`의 `goal_preservation_min=0.7`은 요청대로 유지했다.

## 작업 3 — targeting.py 주제 마커 일반화

수정/추가 파일:

```text
feak_tc/mvp/targeting.py
feak_tc/mvp/propose.py
feak_tc/diagnose/kanana.py
configs/targeting.yaml
tests/test_mvp_loop.py
```

변경:

- `_TOPIC_MARKERS` 하드코딩을 제거했다.
- `5.18`, `헌법`, `인권`, `권리`, `맛있는 밥` 같은 주제/에세이 특화 마커를 targeting 코드에서 제거했다.
- connector/example/style red flag/stopword 상수는 `configs/targeting.yaml`로 이동했다.
- targeter는 다음 신호로 문장을 점수화한다.
  - question 텍스트와의 어휘 겹침
  - question이 없으면 essay 내부 반복/상위 내용어 fallback
  - 문장 길이
  - 예시 표현 유무
  - 연결어 유무
  - 어미/토큰 반복
- `propose.py`가 `Diagnosis.metadata["question"]`을 targeter에 전달하도록 연결했다.
- `KananaDiagnoser`가 `metadata["question"]`을 남기도록 했다.
- 핵심 키워드는 여전히 diagnoser/targeter 입력으로 넣지 않는다.

## 작업 4 — reject_all 7건 검토 자료 추출

산출물:

```text
feak_tc_docs/중간정리/REJECT_ALL_7_REVIEW.md
```

대상:

```text
experiments/results/mvp_stage_a_20_v2.jsonl
decision=reject_all 7건
```

포함 내용:

- essay 앞부분 300자
- before rubrics
- 각 후보의 action / target_rubric / target_gain / non_target_drop / heuristic_score / reject_reasons

검토 대상 record:

```text
train_46
train_79
train_81
train_85
train_91
train_119
train_120
```

## 검증

pytest:

```text
28 passed
```

stub/deterministic 20건 smoke:

```bash
python scripts/run_mvp_batch.py \
  --input data/data_jsonl/train.jsonl \
  --output experiments/results/mvp_stage_a_stub_deterministic_post_codex.jsonl \
  --overwrite --limit 20 --min-chars 250 \
  --diagnoser stub --proposer-mode deterministic --patcher-mode deterministic \
  --n-per-action 1
```

결과:

```text
total=20
ok=20
error=0
```

## 요청에서 유의할 점

이상한 요구는 아니지만, 작업 2의 “SBERT로 교체”는 현재 환경에서 모델 캐시가 없으면
실제 SBERT 값이 아니라 fallback 값으로만 검증된다. 따라서 SBERT 기반
`goal_preservation_min` 재보정은 모델을 로컬에 준비한 뒤 다시 해야 한다.

또한 작업 1 결과상 m=20에서도 최종 정수 rubric의 max|diff|가 1.0이므로,
`target_gain=1`을 노이즈와 분리하려면 정수 rubric이 아니라 `soft_mean` 또는
`rf_corrected_score`를 별도로 분석하는 추가 판단이 필요하다.

