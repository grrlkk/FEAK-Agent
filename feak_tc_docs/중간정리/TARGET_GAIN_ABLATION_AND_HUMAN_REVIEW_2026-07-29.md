# Target-gain ablation 및 2인 사람 평가 진행 현황

작성일: 2026-07-29
브랜치: `corruption_data`

## 1. `target_gain` 제외 LightGBM

기존 STEP3와 같은 42개 인접 transition, 같은 5-fold essay-group split, 같은
`LGBMRanker` 설정을 사용하고 입력에서 `target_gain`만 완전히 제거했다.

| 실험 | 정답 수 | 정확도 | Wilson 95% 하한 | p-value |
|---|---:|---:|---:|---:|
| 전체 feature | 42/42 | 100.0% | 0.916 | `2.27e-13` |
| `target_gain` 제외 | 39/42 | 92.9% | 0.810 | `2.82e-9` |

operator별 결과:

| Operator | 정답 수 | 정확도 |
|---|---:|---:|
| `DELETE_SPECIFICS` | 11/12 | 91.7% |
| `SHUFFLE_FLOW` | 2/3 | 66.7% |
| `INSERT_OFFTOPIC` | 5/5 | 100.0% |
| `INJECT_LEX_REPEAT` | 21/22 | 95.5% |

주로 사용된 feature importance는 다음 순서였다.

1. `non_target_drop`: 108
2. `target_gap_reduction`: 102
3. `evidence_match`: 90

### 해석

`target_gain` 하나를 제거해도 랜덤 수준으로 떨어지지 않았다. 따라서 기존 100%가
오직 `target_gain` 한 값만 읽은 결과는 아니다.

다만 이 실험은 “채점기 정보를 모두 제거한 실험”은 아니다. `non_target_drop`은
다른 rubric 점수 변화이고, `target_gap_reduction`과 `evidence_match`도 transition
방향을 구분할 수 있는 신호다. 현재 결론은 **단일 target 점수 의존은 아니지만,
42쌍 소표본에서 남은 세 feature에 강하게 의존한다**는 것이다.

결과 파일:

- `experiments/results/corruption_g2_gpt5mini_rulev3_lightgbm_30_no_target_gain_report.json`
- `experiments/results/corruption_g2_gpt5mini_rulev3_lightgbm_30_no_target_gain_predictions.jsonl`

## 2. 2인 사람 블라인드 평가

동일한 블라인드 50쌍으로 두 평가자용 파일을 만들었다. A/B 배치는 기존 표본과
동일하고, 문항 순서는 평가자마다 별도로 섞었다.

- 평가자 1: `experiments/results/corruption_g2_gpt5mini_rulev3_human_rater1_50.jsonl`
- 평가자 2: `experiments/results/corruption_g2_gpt5mini_rulev3_human_rater2_50.jsonl`

두 파일에는 다음 정보가 없다.

- 정답 A/B
- corruption operator
- target rubric 및 점수 하락
- cleaner/corrupted stage

각 평가자는 다른 평가자와 상의하지 않고 모든 행의 `preference`에 `A`, `B`,
`TIE` 중 하나를 기록한다. `notes`는 선택 사항이다. 평가자에게 정답 key,
chain, audit 파일을 전달하면 안 된다.

두 파일이 모두 작성되면 다음 집계기를 사용한다.

```bash
python scripts/evaluate_two_human_reviews.py \
  --rater-one experiments/results/corruption_g2_gpt5mini_rulev3_human_rater1_50.jsonl \
  --rater-two experiments/results/corruption_g2_gpt5mini_rulev3_human_rater2_50.jsonl \
  --key experiments/results/corruption_g2_gpt5mini_rulev3_human_key_50.jsonl \
  --report-out experiments/results/corruption_g2_gpt5mini_rulev3_human_2rater_report.json \
  --disagreements-out experiments/results/corruption_g2_gpt5mini_rulev3_human_disagreements.jsonl
```

집계기는 두 평가자의 개별 일치율, 평가자 간 일치율, 불일치 목록을 계산한다.
불일치는 두 사람의 합의 또는 제3자가 `adjudicated_preference`를 작성한 뒤 다시
집계한다. 50쌍의 최종 판정이 모두 정해지고 정답 key와의 일치율이 70% 이상일
때만 G2 human gate를 `passed`로 처리한다.

현재는 실제 사람 라벨이 입력되지 않았으므로 상태는 `pending_raters`다.

## 현재 판단

- `target_gain` 단일 feature 의존 가설: 기각할 근거가 있음
- corruption 품질의 사람 검증: 아직 미완료
- STEP4 대량 생성: 사람 2인 평가가 끝날 때까지 보류
- 커밋·push: 하지 않음
