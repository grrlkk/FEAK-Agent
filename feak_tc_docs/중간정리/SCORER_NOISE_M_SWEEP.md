# 중간정리 — Kanana m-sweep 노이즈 측정

> 작업 1 결과. `scripts/measure_scorer_noise.py`를 `--kanana-m 3/10/20`,
> `--n-essays 5`, `--device-id 3`으로 실행했다.

## 실행 조건

```text
input: data/data_jsonl/train.jsonl
n_essays: 5
min_chars: 300
device_id: 3
variants:
  - rerun: 동일 텍스트 2회 채점
  - whitespace: 공백 collapse 후 채점
  - synonym: 의미 보존 치환 1개 후 채점
```

결과 파일:

```text
experiments/results/scorer_noise_m3.json
experiments/results/scorer_noise_m10.json
experiments/results/scorer_noise_m20.json
experiments/results/scorer_noise_sweep.json
```

## 요약 표

| kanana m | n | rerun max\|diff\| | whitespace max\|diff\| | synonym max\|diff\| | 판정 |
|---:|---:|---:|---:|---:|---|
| 3 | 5 | 1.0 | 1.0 | 1.0 | ±0.5 이하 미달 |
| 10 | 5 | 1.0 | 1.0 | 1.0 | ±0.5 이하 미달 |
| 20 | 5 | 1.0 | 1.0 | 1.0 | ±0.5 이하 미달 |

## 사용된 record

세 실행 모두 같은 5개 record가 사용됐다.

```text
train_129
train_132
train_134
train_136
train_140
```

## 관찰

- m을 3에서 20까지 올려도, 정수 rubric 기준 최대 변동은 1점으로 남았다.
- 측정 기준인 `rerun max|diff| <= 0.5`를 만족한 m은 없다.
- 따라서 현재 정수화된 최종 rubric으로 보면 `target_gain=1`과 `non_target_drop=1`은
  여전히 채점기 노이즈와 분리하기 어렵다.
- m=20에서도 `whitespace`와 `synonym` 변형에서 1점 변동이 남았다.

## 해석

이번 측정만으로는 `m=20이면 노이즈가 ±0.5 이하로 내려간다`고 말할 수 없다.
다만 최종 출력이 정수 rubric이므로, 내부 soft mean의 분산이 줄어도 최종 정수 점수에서는
1점 단위 변동이 남을 수 있다.

후속 판단 시 선택지는 다음과 같다.

1. 최종 정수 score 대신 `soft_mean`/`rf_corrected_score` 기반 transition을 별도 분석한다.
2. `target_gain=1` accept는 별도 재채점 검증 대상으로 둔다.
3. m 증가는 비용 대비 정수 점수 안정화 효과가 제한적일 수 있으므로, 최종 m 결정은
   속도와 soft score 안정성까지 함께 보고 정한다.

## 결론

문서의 판단 기준인 `rerun noise ±0.5 이하`만 적용하면, 이번 sweep에서는
**m=3/10/20 중 어떤 값도 기준을 만족하지 못했다.**

따라서 `target_gain_min` 또는 `non_target_drop_max`를 지금 조정하면 안 되고,
사용자 m 결정 전까지 현재 threshold를 유지하는 것이 맞다.

