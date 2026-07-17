# Stage A soft-score baseline 100건 결과

작성일: 2026-07-16

## 실행 조건

```text
input: data/data_jsonl/train.jsonl
rows: 100 (min_chars=250)
diagnoser: Kanana
kanana_m: 10
proposer/patcher: gpt-4o-mini
n_per_action: 1
config: configs/heuristic_stage_a_soft.yaml
target_gain_min: 0.3
non_target_drop_max: 1.0
GPU: 1, 2, 3
```

최종 병합 로그:

```text
experiments/results/mvp_stage_a_100_soft_m10.jsonl
```

검증 결과:

```text
rows: 100
unique record_id: 100
status ok: 100
error: 0
score_basis: rf_corrected (500/500 candidates)
similarity: token_fallback (500/500 candidates)
```

## Decision 결과

| decision | count |
|---|---:|
| accept | 52 |
| reject_all | 48 |

선택 action:

| action | count |
|---|---:|
| ADD_DETAIL | 24 |
| DELETE_OR_FOCUS | 12 |
| STYLE_REFINE | 6 |
| COMPRESS | 5 |
| RESTRUCTURE | 5 |

선택 후보의 `target_gain`은 최소 0.301, 평균 0.661, 최대 2.465였다.
평균 `non_target_drop`은 0.136이었고, 52개 중 9개는
`goal_preservation < 0.9`였다.

## 해석

`accept 52/100`을 실제 수정 성공률로 해석하면 안 된다.
이 결과는 patch consistency, repetition, factuality guard를 강화하기 전의
pre-guard baseline이다.

기존 20건 질적 검토에서는 다음 문제가 이미 확인됐다.

- target span과 실제 수정 위치 불일치
- 삭제 후 도입부 또는 문맥 붕괴
- 문장 중복
- 원문에 없는 수치와 통계 추가
- action label과 실제 수정 유형 불일치

따라서 다음 비교는 같은 100건 로그의 후보를 대상으로 guard를 적용한 뒤
accept 수뿐 아니라 위 실패 유형이 얼마나 줄었는지를 봐야 한다.

현재 로그에서도 선택 후보 52개 중 9개가 낮은 의미 보존 경고에 해당한다.
특히 `DELETE_OR_FOCUS` 선택 12개는 우선 질적 검토 대상이다.

## 실행 중 발견한 안정성 문제

FEAK 자질 subprocess가 로컬 모델 캐시가 있어도 Hugging Face에 HEAD 요청을 보내면서
DNS 재시도, 300초 timeout, 일부 native segmentation fault가 발생했다.

다음 실행 안정성 조치를 적용했다.

```text
HF_HUB_OFFLINE=1
TRANSFORMERS_OFFLINE=1
OMP_NUM_THREADS=1
MKL_NUM_THREADS=1
MPLCONFIGDIR=/tmp/feak_matplotlib
```

GPU 1, 2, 3 shard가 동시에 자질 추출기를 초기화하지 않도록 process 간 파일 잠금도
추가했다. 수정 후 남은 46건은 46/46 성공했고 오류는 없었다.

## 다음 작업

1. 선택된 52개를 action별로 표본 추출해 질적 판정한다.
2. DELETE_OR_FOCUS 12개와 `goal_preservation < 0.9`인 9개를 전수 검토한다.
3. patch/target consistency와 repetition guard를 구현한다.
4. ADD_DETAIL의 숫자, 연도, 기관, 조사 결과 추가를 검출한다.
5. 같은 100건 후보에 guard를 재적용해 pre/post-guard 결과를 비교한다.
6. 한국어 SBERT를 로컬에 준비한 뒤 token fallback 결과와 비교한다.
