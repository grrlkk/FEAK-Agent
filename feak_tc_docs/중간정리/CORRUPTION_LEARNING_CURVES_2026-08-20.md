# Corruption 1,000쌍 feature/text 학습곡선

> 실행일: 2026-08-20
> 데이터: rule v5 최종 1,000쌍, 683개 essay
> 결론: 같은 corruption을 더 생성하지 않고 현재 1,000쌍으로 TVM Stage-1을 시작한다.

## 판단 질문과 사전 기준

최종 1,000쌍이 부족한지 판단하기 위해 feature-only 모델과 텍스트 모델의 학습곡선을
같은 split에서 비교했다. 주 판단 모델은 수치 transition feature를 전혀 보지 않는
BGE-M3 텍스트 모델이다.

실행 전에 다음 규칙을 `configs/corruption_learning_curve.yaml`에 고정했다.

- 마지막 구간 정확도 증가가 1.5%p를 넘으면 추가 생성
- 증가가 1.5%p 이하이고 최종 정확도가 75% 이상이면 생성 중단
- 곡선이 평탄하지만 75% 미만이면 데이터를 더 만들기 전에 모델/construct를 진단

## 실험 설계

- split unit: `essay_id`
- 5-fold `StratifiedGroupKFold`, operator 층화
- 같은 essay의 transition은 train/test에 동시에 들어가지 않음
- 각 곡선 점에서 held-out 1,000쌍을 fold별로 정확히 한 번씩 평가
- fold 안의 학습 essay 순서를 고정하고 100 → 200 → 400 → 600 → full로 중첩 확대
- full은 fold별 train 789~812쌍, 평균 800쌍
- seed: 20260820 본 실행, 20260821·20260822 민감도 재실행

비교 모델은 다음과 같다.

1. 9개 transition feature를 모두 쓰는 LightGBM ranker
2. 순환성이 가장 큰 `target_gain`을 뺀 LightGBM ranker
3. BGE-M3 frozen state embedding 차이에 단일 선형 head를 둔 진단 모델
4. BGE-M3 frozen state embedding 차이에 `reverse_action`별 선형 head를 둔 주 텍스트 모델

텍스트 state prompt에는 과제, 개선 action, 목표 rubric, 글만 들어간다. 점수, feature,
`target_drop`, accepted 여부는 입력하지 않는다. 네 번째 모델은 최종 TVM이 action token에
조건화되는 구조를 싼 linear probe로 근사한 것이며, Kanana TVM 자체는 아니다.

## 본 실행 학습곡선

| 평균 train 쌍 | feature GBM 전체 | GBM `target_gain` 제외 | BGE 단일 축 | BGE action 조건부 |
|---:|---:|---:|---:|---:|
| 100 | 100.0% | 94.5% | 65.0% | 88.9% |
| 200 | 100.0% | 95.3% | 66.7% | 90.6% |
| 400 | 100.0% | 95.8% | 68.9% | 92.6% |
| 600 | 100.0% | 96.6% | 70.3% | 93.3% |
| 800(full) | 100.0% | 96.7% | 71.4% | 93.1% |

9-feature GBM의 100%는 `target_gain`이 라벨 구성에 직접 들어간 결과이므로 데이터 규모
근거로 사용하지 않는다. `target_gain` 제외 결과도 100쌍에서 이미 94.5%이고 마지막
구간 증가는 0.1%p뿐이다.

단일 BGE 축에서는 OFFTOPIC/LEX와 DELETE의 길이 방향이 충돌해 DELETE가 26.5%까지
역전됐다. `reverse_action`별 방향을 분리하자 DELETE 95.6%, 전체 93.1%로 회복됐다.
이는 첫 71.4%가 데이터 부족이 아니라 모델 construct 문제였음을 보여준다.

## seed 민감도

| seed | BGE action 조건부 600 | full | 마지막 증가 | 결정 |
|---:|---:|---:|---:|---|
| 20260820 | 93.3% | 93.1% | -0.2%p | 생성 중단 |
| 20260821 | 93.2% | 93.6% | +0.4%p | 생성 중단 |
| 20260822 | 93.9% | 94.2% | +0.3%p | 생성 중단 |

세 seed의 full 평균은 93.6%, 마지막 증가 평균은 +0.17%p다. 모든 실행에서 사전 고정한
1.5%p 기준보다 작아 결론이 split seed에 의존하지 않았다.

## operator별 해석

세 seed full 평균:

| operator | 정확도 |
|---|---:|
| `DELETE_SPECIFICS` | 96.5% |
| `INJECT_LEX_REPEAT` | 99.6% |
| `INSERT_OFFTOPIC` | 99.9% |
| `SHUFFLE_FLOW` | 67.3% |

SHUFFLE는 다른 operator보다 어렵지만 평균 곡선이 400~800 구간에서 약 66~67%로
평탄하다. BGE-M3 mean-pooled state embedding이 문장 순서에 둔감한 한계와도 일치한다.
따라서 같은 SHUFFLE corruption을 더 만드는 것보다 순서에 민감한 Kanana TVM에서
operator별 held-out 성능을 재확인하는 것이 낫다. 현재 SHUFFLE 170쌍은 유지하며,
본 TVM에서도 낮고 학습곡선이 상승 중일 때만 해당 operator를 표적 추가 생성한다.

## 결정과 한계

현재 1,000쌍은 명백한 악화 방향을 가르치는 Stage-1 시작 데이터로 충분하다. 추가
corruption 생성은 중단한다. 다음 단계는 스펙대로 Kanana 새 adapter/head를 1 epoch로
학습하고 essay-held-out 및 operator별 성능을 측정하는 것이다.

이 결과는 synthetic corruption 내부 일반화 결과다. 실제 LLM 수정이나 사람 선호에 대한
일반화, hard negative 판별, 좋은 글을 더 좋게 만드는 능력을 증명하지 않는다. 그 부분은
실제 수정쌍과 사람 held-out 평가를 섞는 Stage-2에서 검증해야 한다.

## 검증

- `pytest -q`: 140 passed, 2 warnings
- 본 실행과 검증된 cache 재사용 실행의 JSON 보고서가 byte-identical
- 세 seed 모두 1,000개 transition을 held-out에서 정확히 한 번씩 평가
- cache 재사용 시 pair ID, model snapshot, state prompt SHA-256을 모두 검증

## 산출물

- 본 보고서 JSON: `experiments/results/corruption_g1_rulev5_1000_learning_curves.json`
  - SHA-256: `127dbb45828a4fdb24d67b6903eb2fd8b783a4f86f2c015159fd2b25a95620ab`
- 본 예측 JSONL: `experiments/results/corruption_g1_rulev5_1000_learning_curve_predictions.jsonl`
- BGE state cache: `experiments/results/corruption_g1_rulev5_1000_bge_m3_states_verified.npz`
  - pair ID뿐 아니라 2,000개 state prompt의 SHA-256을 검증한 뒤 재사용
  - prompt SHA-256: `812fe2a9bd981ea7ba7378e0d6d7d6ce01735bad6272e32edd341b09309ab341`
  - file SHA-256: `20a32220e79d8308a873005b9bbc87bedbad6f008aae62cfb0cf5d49fe33e6a3`
- seed 민감도 보고서:
  - `experiments/results/corruption_g1_rulev5_1000_learning_curves_seed20260821.json`
  - `experiments/results/corruption_g1_rulev5_1000_learning_curves_seed20260822.json`

`experiments/results/`는 Git 제외 대상이며, 코드·설정·이 결과 문서만 Git에 기록한다.
