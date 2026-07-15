# Stage A baseline 확장 계획

작성일: 2026-07-14

## 목적

이번 단계의 목적은 최종 TVM 프레임워크를 완성하는 것이 아니라, TVM 없이
`Kanana + Action Generator + Patcher + heuristic selector`만 사용했을 때의 baseline을
100건 규모로 확보하는 것이다.

이 baseline은 이후 `global drift guard`와 `TVM v0`를 붙였을 때 비교 기준으로 사용한다.

## 현재 Stage A 구조

```text
원문
→ Kanana 원문 진단
→ Action 후보 생성
→ Patcher가 후보별 수정문 생성
→ Kanana가 후보별 수정문 재채점
→ rf_corrected_score 기반 transition 계산
→ heuristic + validity rule로 선택/reject
```

바른 API surface normalizer는 구현되어 있지만, Stage A baseline에서는 기본적으로 끈다.
이유는 prior 20건 로그와 비교 가능성을 유지하기 위해서다.

## m 결정

Stage A baseline에서는 `kanana-m = 10`을 사용한다.

근거:

| m | rf_corrected rerun max\|diff\| | rf_corrected rerun mean\|diff\| |
|---:|---:|---:|
| 3 | 0.685 | 0.158 |
| 10 | 0.225 | 0.080 |
| 20 | 0.281 | 0.081 |

`m=20`이 `m=10`보다 비용은 크지만 rerun 안정성이 일관되게 좋아지지 않았다.
따라서 비용 대비 기준으로 `m=10`을 Stage A baseline 값으로 둔다.

## threshold 결정

Stage A baseline 전용 config:

```text
configs/heuristic_stage_a_soft.yaml
```

사용값:

```text
target_gain_min = 0.3
non_target_drop_max = 1.0
```

근거:

- soft score 기준 기존 `target_gain_min=1.0`은 너무 높아 20건이 모두 reject됐다.
- `m=10`의 관측 rerun max noise가 0.225였고, 0.3은 그보다 약간 높은 보수적 기준이다.
- 20건 virtual sweep에서 `target_gain_min=0.3`은 7/20건 accept였다.
- `non_target_drop_max`는 0.3~1.0으로 바꿔도 accept 수 변화가 상대적으로 작았기 때문에 Stage A에서는 1.0을 유지한다.

주의:

이 threshold는 최종값이 아니다. 직접 질적 검토에서 잘못된 patch와 근거 없는 detail 추가가
통과한 사례가 있었다. 따라서 이 값은 baseline 생성을 위한 임시 조건이다.

## 100건 실행 명령

```bash
python scripts/run_mvp_batch.py \
  --input data/data_jsonl/train.jsonl \
  --output experiments/results/mvp_stage_a_100_soft_m10.jsonl \
  --config configs/heuristic_stage_a_soft.yaml \
  --diagnoser kanana \
  --kanana-m 10 \
  --device-id 3 \
  --proposer-mode llm \
  --patcher-mode llm \
  --n-per-action 1 \
  --limit 100 \
  --min-chars 250 \
  --overwrite
```

GPU를 병렬로 사용할 경우 GPU 1, 2, 3에 shard를 나눠 실행한 뒤 병합한다. GPU 0은 사용하지 않는다.

## 100건에서 볼 것

단순 accept 수보다 아래 실패 유형을 수집하는 것이 중요하다.

- 근거 없는 수치, 연도, 기관명 추가
- target_span과 실제 patch 위치 불일치
- 삭제 후 도입부나 문맥 붕괴
- 문장 중복 또는 긴 n-gram 반복 증가
- 낮은 target_gain이지만 명백히 좋은 STYLE_REFINE
- 높은 target_gain이지만 사람이 보면 나쁜 ADD_DETAIL

## 다음 단계

100건 baseline 이후에는 같은 로그를 기준으로 다음을 구현하고 비교한다.

1. global drift guard
2. TVM v0
3. Stage A baseline vs guard/TVM 적용 결과 비교
