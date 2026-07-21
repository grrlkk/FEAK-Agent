# 중간정리 — soft score threshold sweep 20건

> 작성일: 2026-07-13
> 입력 로그: `experiments/results/mvp_stage_a_20_v3.jsonl`
> 목적: Kanana/LLM 재실행 없이, 기존 soft score transition 로그에서 threshold만 바꿔 가상 재판정

## 1. 왜 이 작업을 했나

soft score 전환 후 기존 threshold인 `target_gain_min = 1.0`을 그대로 적용하니,
20건 모두 `reject_all`이 됐다.

이 결과는 soft score 구현이 실패했다는 뜻이 아니라, 기존 정수 점수 기준의 `+1` 개선이
연속 점수 기준에서는 대부분 `+1.0` 미만이었다는 뜻이다.

따라서 다음 실험 전에 먼저 확인해야 할 것은 다음이다.

```text
연속 점수 기준에서 어느 정도 target_gain을 실제 개선으로 볼 것인가?
```

## 2. 재판정 방식

- 새로 Kanana를 돌리지 않았다.
- 새로 LLM을 호출하지 않았다.
- 기존 v3 로그의 후보 100개 transition 값을 그대로 사용했다.
- validity reject는 그대로 유지했다.
- `target_gain_min`, `non_target_drop_max`만 가상으로 바꿔 다시 accept/reject를 계산했다.

기본 설정:

```text
input: experiments/results/mvp_stage_a_20_v3.jsonl
rows: 20
candidates: 100
score_basis: rf_corrected_score
edit_ratio_max: 0.5
goal_preservation_min: 0.7
evidence_match_min: 0.3
accept_threshold: 0.0
```

## 3. target_gain_min sweep

`non_target_drop_max = 1.0`은 그대로 두고 `target_gain_min`만 바꿨다.

| target_gain_min | accept | reject_all | chosen ADD_DETAIL | DELETE_OR_FOCUS | COMPRESS | RESTRUCTURE | STYLE_REFINE |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.0 | 19 | 1 | 9 | 5 | 2 | 2 | 1 |
| 0.1 | 16 | 4 | 9 | 3 | 0 | 2 | 2 |
| 0.2 | 12 | 8 | 8 | 1 | 0 | 2 | 1 |
| 0.3 | 7 | 13 | 4 | 1 | 0 | 2 | 0 |
| 0.4 | 6 | 14 | 3 | 1 | 0 | 2 | 0 |
| 0.5 | 5 | 15 | 3 | 1 | 0 | 1 | 0 |
| 0.6 | 4 | 16 | 3 | 1 | 0 | 0 | 0 |
| 0.7 | 2 | 18 | 1 | 1 | 0 | 0 | 0 |
| 0.8 | 0 | 20 | 0 | 0 | 0 | 0 | 0 |

## 4. non_target_drop_max와의 2D sweep

accept 수만 표시했다.

| target_gain_min \\ non_target_drop_max | 0.3 | 0.5 | 0.7 | 1.0 |
|---:|---:|---:|---:|---:|
| 0.1 | 15 | 16 | 16 | 16 |
| 0.2 | 11 | 11 | 12 | 12 |
| 0.3 | 6 | 6 | 7 | 7 |
| 0.4 | 5 | 5 | 6 | 6 |
| 0.5 | 5 | 5 | 5 | 5 |

현재 20건에서는 `non_target_drop_max`를 0.3까지 낮춰도 accept 수 변화가 크지 않다.
즉 지금은 `non_target_drop`보다 `target_gain_min`이 훨씬 강하게 decision을 좌우한다.

## 5. target_gain_min = 0.3일 때 accept 후보

`target_gain_min=0.3`, `non_target_drop_max=1.0` 기준이다.

| record | action | target rubric | gain | drop | edit | preserve | instruction |
|---|---|---|---:|---:|---:|---:|---|
| train_45 | ADD_DETAIL | content_2 | 0.760 | 0.183 | 0.229 | 1.000 | 투표의 중요성을 강조하는 구체적인 사례나 설명을 추가 |
| train_59 | RESTRUCTURE | organization_1 | 0.438 | 0.575 | 0.132 | 0.937 | 앞 문장과 연결해 흐름 개선 |
| train_70 | ADD_DETAIL | content_2 | 0.390 | 0.000 | 0.189 | 1.000 | 차별 문제의 구체적인 사례나 통계 추가 |
| train_85 | DELETE_OR_FOCUS | task_1 | 0.780 | 0.000 | 0.118 | 0.882 | 주제와 관련 약한 부분 삭제/초점화 |
| train_94 | ADD_DETAIL | content_2 | 0.676 | 0.131 | 0.145 | 1.000 | 유럽 침략의 구체적 사례나 결과 추가 |
| train_97 | ADD_DETAIL | content_2 | 0.625 | 0.000 | 0.169 | 1.000 | 외국인 노동자 차별 사례 추가 |
| train_114 | RESTRUCTURE | organization_1 | 0.534 | 0.000 | 0.029 | 0.971 | 문장 순서나 연결 표현 조정 |

## 6. 0.3 근처 near-miss 후보

`target_gain_min=0.3`에서 아깝게 reject된 후보들이다.

| record | action | target rubric | gain | drop | 비고 |
|---|---|---|---:|---:|---|
| train_90 | ADD_DETAIL | content_2 | 0.297 | 0.200 | 0.3 바로 아래 |
| train_117 | ADD_DETAIL | content_2 | 0.290 | 0.237 | 0.3 바로 아래 |
| train_91 | STYLE_REFINE | expression_1 | 0.244 | 0.203 | 약한 개선 |
| train_99 | ADD_DETAIL | content_3 | 0.234 | 0.159 | 약한 개선 |
| train_40 | ADD_DETAIL | content_2 | 0.214 | 0.000 | 약한 개선 |

이 near-miss 때문에 `0.3`은 아주 단단한 경계라기보다, 다음 실험을 위한 실용적 시작점으로 보는 것이 맞다.

## 7. 해석

### target_gain_min = 0.1

accept가 16/20으로 많다.
하지만 m=10 기준 rerun mean noise가 약 0.080이므로, 0.1은 noise와 너무 가까워 보인다.

### target_gain_min = 0.2

accept가 12/20이다.
후보를 넉넉하게 살리지만, 약한 개선도 많이 통과할 가능성이 있다.

### target_gain_min = 0.3

accept가 7/20이다.
m=10 rerun max noise가 약 0.225였으므로, 0.3은 noise 상한보다 약간 높은 보수적 시작점이다.

### target_gain_min = 0.5

accept가 5/20이다.
조금 더 보수적이며, 명확한 후보만 남긴다. 다만 좋은 약한 수정까지 버릴 위험이 있다.

## 8. 현재 결론

숫자만 보면 다음 값이 다음 실험용 시작점으로 가장 합리적이다.

```text
다음 실험용 provisional threshold:
target_gain_min = 0.3
non_target_drop_max = 1.0 유지
```

단, 이것은 최종 threshold가 아니다. 직접 질적 검토 결과, `target_gain_min=0.3`에서도
잘못된 patch와 근거 없는 detail 추가가 통과했다. 따라서 threshold 확정보다 먼저
patch validity와 factuality guard를 강화해야 한다.

관련 문서:

```text
feak_tc_docs/중간정리/SOFT_THRESHOLD_QUAL_REVIEW_20.md
```

## 9. 다음 해야 할 일

1. `target_gain_min=0.3`에서 accept된 7개 후보를 사람이 직접 읽고 질적으로 판단한다.
2. 특히 ADD_DETAIL이 불필요한 세부사항을 추가하는지 확인한다.
3. `train_90`, `train_117`처럼 0.3 근처 후보를 함께 보고 0.25/0.3/0.35 중 어디가 나은지 판단한다.
4. 임시 threshold를 정하면 50건 로그로 확장한다.
5. 이후 TVM 학습용 로그 설계로 넘어간다.
