# Corruption Rule v3 — STEP1~3 결과 보고

작성일: 2026-07-29
브랜치: `corruption_data`

## 결론

`TASK_데이터생성_단계.md`의 STEP1~3을 구현하고 검증했다.

- 메인 corruption chain에서 `INJECT_GRAMMAR_ERR`를 제외했다.
- 문법 오류 데이터는 맞춤법 교정 단계 검증용 별도 데이터로 분리했다.
- `DELETE_SPECIFICS`와 `SHUFFLE_FLOW`의 보존 조건을 강화했다.
- 기존과 같은 30개 에세이로 체인을 다시 생성하고 재측정했다.
- margin `0.3`을 적용해 90개 전이 중 42개를 채택했다.
- 채택된 인접 전이로 essay-group split LightGBM sanity test를 수행했고 42/42를 맞혔다.

따라서 STEP3 sanity gate는 통과했다. 다만 이 결과는 feature 기반 최소 학습 가능성 확인이며, 사람 선호나 TVM의 최종 타당성을 증명한 결과는 아니다.

## 변경된 데이터 생성 구조

메인 체인의 operator는 다음 네 개다.

| Operator | Target rubric | 핵심 보존 조건 |
|---|---|---|
| `DELETE_SPECIFICS` | `content_2` | 선택한 구체 정보 span만 삭제하고 그 밖의 문자는 바꾸지 않음 |
| `SHUFFLE_FLOW` | `organization_1` | 문장 집합은 보존하되 실제 담화 의존 문장의 선행 관계를 끊음 |
| `INSERT_OFFTOPIC` | `organization_2` | 원문을 보존하고 주제 이탈 문장만 삽입 |
| `INJECT_LEX_REPEAT` | `expression_1` | 원문 의미를 유지하며 어휘 반복만 주입 |

`INJECT_GRAMMAR_ERR`는 메인 체인에서 빠졌으며, 다음 용도에만 사용한다.

```text
D_raw → 맞춤법 검사기 API → D0 → 메인 corruption chain
```

별도 surface-validation 샘플 10개를 생성했고, 각 샘플에는 조사 교체·띄어쓰기·철자 오류가 포함된다.

## 30개 에세이 재생성 결과

- 에세이: 30편
- 상태: 30편 모두 정상 완료
- 총 상태: 120개
- 총 인접 전이: 90개
- 문법 오류 전이: 0개
- 보존 조건 검사: 전 전이 통과

margin `0.3` 적용 결과:

| Operator | 생성 | 채택 | 채택률 |
|---|---:|---:|---:|
| `DELETE_SPECIFICS` | 25 | 12 | 48.0% |
| `SHUFFLE_FLOW` | 21 | 3 | 14.3% |
| `INSERT_OFFTOPIC` | 18 | 5 | 27.8% |
| `INJECT_LEX_REPEAT` | 26 | 22 | 84.6% |
| 합계 | 90 | 42 | 46.7% |

이전 rule v2에서 문법 오류를 제외한 채택률은 27/65, 41.5%였다. rule v3는 42/90, 46.7%다. `SHUFFLE_FLOW` 채택률이 낮아진 것은 임의 문장 이동을 막고 실제 담화 의존성 단절만 허용하도록 조건을 강화한 영향이다.

## STEP3 LightGBM sanity test

- 입력: margin `0.3`을 통과한 인접 전이 42개
- 모델: `LGBMRanker`, objective `lambdarank`
- 분할: essay ID 기준 5-fold group split
- 정확도: 42/42, 100%
- Wilson 95% 하한: 0.916
- 양측 이항검정 p-value: `2.27e-13`
- operator별 정확도: 네 operator 모두 100%
- stage gap: 전부 1

중요한 한계가 있다. feature importance에서 `target_gain`이 압도적으로 컸다. 따라서 현재 결과는 “채택 전이의 방향이 feature 공간에서 구분된다”는 sanity check로 해석해야 한다. 풍부한 일반화 성능, 사람 판단과의 일치, 최종 TVM 성능을 입증한 것은 아니다.

## 사람 선호 평가 준비 상태

블라인드 비교용 50쌍과 정답 key를 만들었다.

- 유효 후보: 58쌍
- 추출: 50쌍
- 고유 에세이: 21편
- stage gap: 1단계 35쌍, 2단계 11쌍, 3단계 4쌍
- 상태: `pending_human_review`
- 계획: 2인 평가, 선호 일치율 기준 0.7

사람 평가를 아직 수행하지 않았으므로 human gate를 통과했다고 볼 수는 없다.

## 주요 결과 파일

- 메인 체인: `experiments/results/corruption_g1_gpt5mini_rulev3_30_chains.jsonl`
- margin 0.3 채택 데이터: `experiments/results/corruption_g1_gpt5mini_rulev3_30_accepted_m03.jsonl`
- 생성·채택 요약: `experiments/results/corruption_g1_gpt5mini_rulev3_30_summary_m03.json`
- surface-validation 문법 오류: `experiments/results/corruption_surface_rulev3_10.jsonl`
- LightGBM 보고서: `experiments/results/corruption_g2_gpt5mini_rulev3_lightgbm_30_report.json`
- LightGBM 예측: `experiments/results/corruption_g2_gpt5mini_rulev3_lightgbm_30_predictions.jsonl`
- 사람 평가 블라인드 50쌍: `experiments/results/corruption_g2_gpt5mini_rulev3_human_review_50.jsonl`
- 사람 평가 정답 key: `experiments/results/corruption_g2_gpt5mini_rulev3_human_key_50.jsonl`

## 다음 게이트

STEP3 결과로 STEP4의 100/200/300편 learning curve 생성은 기술적으로 진행 가능하다. 다만 최대 300편 기준으로 약 1,200개 상태의 생성·채점·자질 추출이 필요해 API 비용과 GPU 실행 시간이 큰 작업이다.

다음 실행 단위는 다음과 같다.

1. 100편 생성·재측정·margin 필터·LightGBM 평가
2. 동일 조건으로 200편과 300편까지 확장
3. 데이터 규모별 성능과 operator별 안정성 비교
4. 별도로 준비한 50쌍에 대해 2인 블라인드 평가

아직 커밋이나 원격 push는 하지 않았다.
