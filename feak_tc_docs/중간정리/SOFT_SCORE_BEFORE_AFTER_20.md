# 중간정리 — soft score transition 전후 20건 비교

> 실행일: 2026-07-12  
> 비교: `experiments/results/mvp_stage_a_20_v2.jsonl` → `experiments/results/mvp_stage_a_20_v3.jsonl`

## 실행 조건

```text
input: data/data_jsonl/train.jsonl
limit: 20
min_chars: 250
diagnoser: kanana
kanana_m: 10
transition gain/drop basis: rf_corrected_score
threshold 변경: 없음
```

`mvp_stage_a_20_v3.jsonl`은 첫 5건을 순차 실행한 뒤, feature extraction segfault를 피하기 위해
남은 15건을 5건씩 3개 shard로 나누어 실행하고 원래 record 순서로 합쳤다. shard 실행 시
`FEAK_FEATURE_CUDA_VISIBLE_DEVICES=`로 FEAK feature extraction은 CPU에 고정했고, Kanana scoring은
GPU 1/2/3을 사용했다.

## compare_mvp_logs 요약

| 항목 | v2 | v3 |
|---|---:|---:|
| ok rows | 20 | 20 |
| accept | 13 | 0 |
| reject_all | 7 | 20 |
| chosen ADD_DETAIL | 10 | 0 |
| chosen DELETE_OR_FOCUS | 1 | 0 |
| chosen COMPRESS | 2 | 0 |
| reject reason: target_gain | 76 | 100 |
| reject reason: non_target_drop | 19 | 4 |
| re-diagnoses spent | 96 | 97 |

## target_gain 분포

| 기준 | n | min | p25 | median | mean | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 all candidates | 100 | -3.000 | 0.000 | 0.000 | 0.020 | 0.000 | 3.000 |
| v2 accepted | 13 | 1.000 | 1.000 | 1.000 | 1.538 | 2.000 | 3.000 |
| v3 all candidates | 100 | -1.650 | -0.155 | 0.000 | -0.023 | 0.112 | 0.780 |

v3에서는 모든 candidate metadata에 `score_basis: rf_corrected`가 기록됐다. 정수 rubric에서는
`target_gain=1/2/3`처럼 보이던 accept 후보들이, 연속 점수 기준에서는 최대 gain도 0.780에
그쳤다. 따라서 기존 `target_gain_min=1.0`을 유지하면 모든 후보가 `target_gain`으로 reject된다.

## gain < 0.5 accept

v3 accepted candidate가 0건이므로, gain < 0.5인데 accept된 후보도 0건이다.

다만 이 숫자는 “문제가 없다”는 뜻이 아니라, 현재 threshold를 그대로 두면 soft score 전환 후
accept가 완전히 닫힌다는 뜻이다. m과 threshold 최종 결정 전까지는 이 결과를 측정값으로만
봐야 한다.

## non_target_drop 분포

| 기준 | n | min | p25 | median | mean | p75 | max |
|---|---:|---:|---:|---:|---:|---:|---:|
| v2 all candidates | 100 | 0.000 | 0.000 | 1.000 | 0.840 | 1.000 | 3.000 |
| v2 accepted | 13 | 0.000 | 0.000 | 0.000 | 0.308 | 1.000 | 1.000 |
| v3 all candidates | 100 | 0.000 | 0.122 | 0.217 | 0.315 | 0.456 | 2.541 |

`non_target_drop`도 정수 0/1 덩어리에서 연속 분포로 바뀌었다. reject reason 기준으로는
`non_target_drop`이 19건에서 4건으로 줄었고, 대부분의 reject는 `target_gain` 쪽으로 이동했다.

## 해석

- soft score 전환 자체는 적용됐다. v3 후보 100개 모두 `score_basis=rf_corrected`다.
- target_gain이 정수 1.0 덩어리에서 소수 분포로 퍼졌다.
- threshold를 유지한 결과, v3 20건은 모두 reject_all이다.
- 이 결과는 threshold를 지금 바꾸자는 결론이 아니라, 기존 `target_gain_min=1.0`이
  연속 점수 단위에서는 너무 높은 기준일 가능성이 크다는 측정 결과다.
- SBERT 모델은 로컬 캐시에 없어 이번 실행에서도 similarity는 `token_fallback`으로 계산됐다.

