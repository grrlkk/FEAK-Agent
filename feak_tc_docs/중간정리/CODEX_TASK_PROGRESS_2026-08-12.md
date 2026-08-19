# Corruption rule v4 진행 기록

- 실행일: 2026-08-17
- 브랜치: `corruption_data`
- 결론: v4 30편 파일럿의 자동 게이트는 통과했다. 대량 생성은 실제 사람 2인의
  블라인드 50쌍 검수 완료 후 시작한다.

## 독립 판단

`CODEX_TASK_CORRUPTION_V4.md`의 핵심 진단인 고정 LEX/OFFTOPIC 템플릿 제거와
SHUFFLE 민감도 선행 측정에는 동의했다. 다음 항목은 그대로 적용하지 않았다.

- 자연어 10-gram 2%만으로는 7-token 가변 템플릿을 놓치므로, 생성 edit span만
  대상으로 exact canonical signature, word 5~8-gram, char 20-gram을 검사한다.
- 30편에서 2%는 1건이므로 우연한 공통 표현까지 실패시킨다. 위반 기준은 서로 다른
  원문 `max(3편, ceil(2%))`으로 정했다.
- 전역 `target_drop > 0.5`는 LEX 쏠림을 악화시켰다. 전체 625개 threshold 조합을
  비교해 m=10 재측정 노이즈 상한을 엄격히 넘는 기본 0.225와 LEX 0.4를 선택했다.
- 고정 문구를 만들 수 없는 DELETE/SHUFFLE까지 LLM 비중을 높일 이유가 없다.
  두 연산자는 원문 span만 삭제·이동하는 rule-only로 두고, 새 문장을 만드는
  LEX/OFFTOPIC만 LLM-only로 제한했다.
- 두 LLM의 선호 판정은 공식 사람 검수를 대체하지 않는다. 최종 생산 게이트는 서로
  독립적인 사람 2인과 불일치 adjudication으로 유지한다.

## 구현

- LEX/OFFTOPIC 고정 rule 문구 및 rule fallback 제거
- LLM anchor 오복사는 생성 문장을 바꾸지 않고 가장 가까운 원문 문장으로 배치점만 복구
- 모든 후속 연산자의 target/anchor가 원문 유래인지 검증
- DELETE의 반복 5-gram, 잔존 본문 8-gram 중복 및 측정 후 교차축 개선 가드
- artifact와 operator balance 감사를 필터 파이프라인에 자동 연결
- SHUFFLE 상한 진단, 측정 재사용, shard 병합/재분배 도구 추가
- 전역 whole-text LLM 정규화를 identity pass로 변경. v4는 국소 편집 외 텍스트가
  그대로인지 별도 보존 검사로 확인한다.

## SHUFFLE 민감도

원문 10편을 seed 3개로 완전 셔플한 30쌍에서 RF-corrected `organization_1` 하락은
평균 0.647, 중앙값 0.685였고 19/30이 0.5를 넘었다. 채점기는 문장 순서에
반응하므로 SHUFFLE_FLOW를 유지하되 한 문장 이동에서 원문 문장 2개 이동으로
강화했다. 세부 보고는 `SHUFFLE_SENSITIVITY_2026-08-12.md`에 있다.

## v4 30편 결과

### 생성·채택

| operator | 생성 | 채택 | 채택률 | target drop 최소 | 중앙값 | 최대 |
|---|---:|---:|---:|---:|---:|---:|
| DELETE_SPECIFICS | 25 | 7 | 28.0% | 0.274 | 0.469 | 1.224 |
| INJECT_LEX_REPEAT | 26 | 16 | 61.5% | 0.407 | 1.008 | 1.684 |
| INSERT_OFFTOPIC | 18 | 11 | 61.1% | 0.314 | 0.655 | 0.913 |
| SHUFFLE_FLOW | 21 | 7 | 33.3% | 0.229 | 0.265 | 0.757 |
| 합계 | 90 | 41 | 45.6% |  |  |  |

- chain: 30/30 `ok`, 상태 120개 측정 완료, 누락/중복 0
- accepted essay: 24편
- fallback: 0
- preservation failure: 0
- 채택 분포: DELETE 17.1%, LEX 39.0%, OFFTOPIC 26.8%, SHUFFLE 17.1%
- 40% operator 상한: 통과
- artifact audit: 통과, 위반 0
- DELETE measured confound: 2건 폐기 (`train_28624`, `train_5652`)
- LightGBM feature-only sanity: 41/41, grouped pairwise accuracy 1.0. 가장 큰 중요도가
  `target_gain`이므로 이 수치는 파이프라인 sanity이며 텍스트 TVM 성능 증거로 해석하지 않는다.

주요 결과:

- `experiments/results/corruption_g1_gpt5mini_rulev4_30_chains.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_audit.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_accepted.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_summary.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_quality.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_g2_lightgbm_report.json`

## 사람 블라인드 게이트

유효 후보 52쌍 중 50쌍을 추출했다. stage gap은 1/2/3이 각각 40/8/2이고,
24개 원문을 포함한다. A/B key는 26/24로 균형이다.

- rater 1: `experiments/results/corruption_g1_gpt5mini_rulev4_30_human_rater1.jsonl`
- rater 2: `experiments/results/corruption_g1_gpt5mini_rulev4_30_human_rater2.jsonl`
- 숨긴 key: `experiments/results/corruption_g1_gpt5mini_rulev4_30_human_review_key.jsonl`

두 rater가 각각 `preference`를 `A`, `B`, `TIE` 중 하나로 채운 뒤
`scripts/evaluate_two_human_reviews.py`로 평가한다. 두 파일이 모두 완성되고,
불일치가 adjudication되며, 최종 key agreement가 70% 이상일 때만 STEP4 대량
생성을 시작한다.

## 2-LLM API 블라인드 프록시 게이트 (2026-08-18)

사용자 승인에 따라 실제 사람 2인 게이트 대신 먼저 OpenAI API 기반 독립 모델 2개로
같은 50쌍을 검수했다. Codex/CLI 토큰이나 sub-agent는 사용하지 않았다. 각 pair는
별도 API context에서 평가했고, 숨긴 key와 상대 모델 결과는 모든 model input에서
제외했다.

- reviewer 1: `gpt-5-mini-2025-08-07`, 46/50 = 92%
- reviewer 2: `gpt-4.1-2025-04-14`, 46/50 = 92%
- inter-rater exact agreement: 46/50 = 92%
- 4개 불일치는 두 모델 중 하나의 새 blind call을 번갈아 사용해 재검수
- 최종 key agreement: 45/50 = 90%
- cleaner와 반대인 A/B를 선택한 오답: 두 reviewer 모두 0건
- 각 reviewer의 미일치 4건은 전부 `TIE`
- 판정: `passed` (70% 기준)

operator 포함 기준 식별률은 reviewer 1/reviewer 2 각각
`DELETE_SPECIFICS` 8/9, 8/9, `INJECT_LEX_REPEAT` 24/24, 23/24,
`INSERT_OFFTOPIC` 17/17, 17/17, `SHUFFLE_FLOW` 9/12, 10/12였다.
`SHUFFLE_FLOW`가 상대적으로 약하므로 100편 생산 후 operator별 acceptance와 blind
식별률을 다시 점검한다.

결과 파일:

- `experiments/results/corruption_g1_gpt5mini_rulev4_30_two_llm_api_report.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_two_llm_api_adjudication.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_30_two_llm_api_disagreements.jsonl`

이 결과는 `2-LLM API proxy gate`이며 실제 사람 검수로 표기하지 않는다.

## Rule v4 100편 1차 생산 결과 (2026-08-18)

`data/corruption/main_1000.jsonl`에서 v4 파일럿과 Stage A와 겹치지 않는
소스를 사용했다. 첫 100편은 `ok/partial/failed = 77/19/4`였고, 보존
가드를 완화하지 않고 다음 40편을 oversampling해 `ok` 34편을 더 확보했다.
소스 순서를 복원한 뒤 첫 `ok` 100개만 최종 풀로 고정했다.

생성 게이트:

- 완전 체인 100개, 상태 400개, 인접 전이 300개
- 보존 실패 0, fallback 0, normalization 0
- 생성 operator: `DELETE_SPECIFICS` 81, `SHUFFLE_FLOW` 77,
  `INJECT_LEX_REPEAT` 72, `INSERT_OFFTOPIC` 70
- 생성 edit artifact 위반 0
- 최대 operator 비중 27% (`DELETE_SPECIFICS`), 40% cap 통과

Kanana m=10 재측정은 GPU 4개에서 400/400개를 완료했다. 측정 key
`(record_id, state_index)`는 400개 모두 unique하며 missing/extra/duplicate는 0이다.

파일럿에 사전 고정한 임계 `global > 0.225`, `LEX_REPEAT > 0.4`를
그대로 적용한 결과:

- 채택 129/300 = 43.0%, 80개 essay
- `DELETE_SPECIFICS` 25, `INJECT_LEX_REPEAT` 40,
  `INSERT_OFFTOPIC` 38, `SHUFFLE_FLOW` 26
- 최대 채택 operator 비중 31.0%, balance 통과
- 채택 edit artifact 위반 0
- DELETE cross-axis improvement confound 5건 제거
- acceptance gate 재계산 검증 통과

현재 배치에서 모든 operator에 `> 0.225`를 쓰면 146개가 채택되고 최대
비중은 39.0%로 cap을 통과하지만, 100편 결과를 본 후의 post-hoc 임계
변경이 되므로 공식 채택 풀에는 반영하지 않았다.

essay-grouped 5-fold LightGBM sanity는 129/129를 맞혔다. `target_gain`을
제외한 사후 ablation은 120/129 = 93.0%였다. 다만 둘 다 text를 사용하지
않은 feature sanity이므로 최종 TVM 일반화 성능을 증명하지는 않는다.

주요 결과 파일:

- `experiments/results/corruption_g1_gpt5mini_rulev4_100_chains.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_generated_quality.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_audit.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_accepted.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_summary.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_quality.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_g2_lightgbm_report.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_100_g2_lightgbm_no_target_gain_report.json`

## 다음 단계

1. 동일한 pre-registered threshold로 200편까지 순차 확대
2. 200편 artifact/balance/acceptance와 `SHUFFLE_FLOW` 안정성 재평가
3. 이상이 없을 때 300편으로 확대
4. feature-only GBM과 텍스트 pairwise 모델의 학습곡선을 모두 측정해 더 늦은 포화점을
   최종 데이터 규모 기준으로 사용

기존 v3 채택 42쌍은 오염 분석 기록으로만 보존하고 TVM 학습에는 사용하지 않는다.

## Rule v4 누적 200편 결과 (2026-08-18)

### 소스 확대와 선택 편향 감사

기존 생성 시도에서 채택하지 않았던 `ok` 11편을 버리지 않고 재사용하고, 원문 offset
140 이후 120편을 추가 생성했다. 추가 구간은 `ok/partial/failed = 103/13/4`였으며,
누적 260회 시도에서 원문 순서상 첫 `ok` 200편을 고정했다. 기존 100편과 겹치지 않는
증분 100편만 새로 측정했다.

생성 완료 여부에 따른 소스 선택 편향을 별도 자동 감사했다.

- 누적 시도: `ok/partial/failed = 214/37/9`
- 선택 200편 vs 미선택 60편의 표준화 평균차(SMD): 글자 수 0.450, 문장 수 0.344,
  사람 평균점수 -0.084, 문항 길이 -0.372
- 생성 `ok` 214편 vs 미완료 46편: 글자 수 0.534, 문장 수 0.424,
  사람 평균점수 -0.073, 문항 길이 -0.411
- 결론: 점수 편향은 작지만 짧은 글에서 완전 체인이 덜 만들어지는 길이 편향은
  무시할 수 없다. 현재 200편 checkpoint는 진행하되 최종 규모 전에 partial chain
  활용 또는 길이 층화 방안을 검토한다.

질문 문구는 260편에서 163종으로 희소해 exact-question 분포 비교의 TV distance가
과대해질 수 있으므로 길이 대리 지표와 함께 기록했다. 감사 결과는
`experiments/results/corruption_g1_gpt5mini_rulev4_200_selection_bias.json`에 있다.

### 생성·측정

- 완전 체인 200개, 상태 800개, 인접 전이 600개
- 보존 실패 0, fallback 0, normalization 0, duplicate record ID 0
- 생성 operator: `DELETE_SPECIFICS` 151, `INJECT_LEX_REPEAT` 147,
  `INSERT_OFFTOPIC` 147, `SHUFFLE_FLOW` 155
- 생성 최대 operator 비중 25.8%, 40% cap 통과
- 생성 edit artifact 위반 0
- 기존 측정 400개를 재사용하고 증분 400개만 GPU 4개에서 m=10으로 측정
- 누적 측정 key `(record_id, state_index)` 800개 모두 unique,
  missing/extra/duplicate 0, 모든 row에 RF-corrected 8개와 feature 29개 존재

### 채택과 corpus artifact 격리

파일럿에 사전 고정한 `global > 0.225`, `LEX_REPEAT > 0.4`를 그대로 적용했다.
최초 점수·교차축 필터에서는 259개가 통과했지만, 채택 부분집합 artifact 감사가
다음 공통 OFFTOPIC 5-gram을 서로 다른 3편에서 발견했다.

`어제 본 영화의 결말이 아직도`

생성 전체 200편에서는 3/200이라 `max(3편, ceil(2%)) = 4편` 기준 아래였지만,
채택된 생성 edit의 서로 다른 원문 127편에서는 3/127로 기준을 넘었다. 관련 전이
3개(`train_15553:stage2`, `train_32431:stage1`, `train_55345:stage3`)를 임의로 하나만
남기지 않고 전부 `corpus_artifact`로 격리했다. 필터가 violation cluster 전체를
자동 격리하고 재감사하도록 변경했으며 최종 artifact 위반은 0이다.

최종 결과:

- 채택 256/600 = 42.7%, 161개 essay
- essay당 채택 수: 0개 39편, 1개 79편, 2개 69편, 3개 13편
- `DELETE_SPECIFICS` 48/151 = 31.8%
- `INJECT_LEX_REPEAT` 78/147 = 53.1%
- `INSERT_OFFTOPIC` 73/147 = 49.7%
- `SHUFFLE_FLOW` 57/155 = 36.8%
- 최대 채택 operator 비중 30.5%, balance 통과
- DELETE cross-axis improvement confound 11건 제거
- acceptance gate 재계산 검증 통과
- 자연 행 key `(chain_id, stage_k)` 256개 모두 unique, 완전 중복 row 0

현재 데이터에서 모든 operator에 `> 0.225`를 적용하면 artifact 격리 후 282개가
남고 최대 비중은 36.9%지만, 누적 결과를 본 뒤의 post-hoc 임계 변경이므로 공식
채택 풀에는 반영하지 않았다.

### G2 feature sanity

essay-grouped 5-fold LightGBM은 256/256을 맞혔다. `target_gain`이 중요도 203으로
압도적이므로 파이프라인 sanity 이상의 의미를 부여하지 않는다. `target_gain` 제외
사후 ablation은 244/256 = 95.3%였고 operator별 정확도는 DELETE 100%, LEX 94.9%,
OFFTOPIC 94.5%, SHUFFLE 93.0%였다. 이 ablation도 scorer-derived feature를 사용하므로
텍스트 TVM 일반화 증거는 아니다.

주요 결과 파일:

- `experiments/results/corruption_g1_gpt5mini_rulev4_200_chains.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_generated_quality.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_selection_bias.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_audit.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_accepted.jsonl`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_summary.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_quality.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_g2_lightgbm_report.json`
- `experiments/results/corruption_g1_gpt5mini_rulev4_200_g2_lightgbm_no_target_gain_report.json`

## 200편 checkpoint 판단

100→200편에서 채택률은 43.0%→42.7%, 최대 operator 비중은 31.0%→30.5%로
안정적이었고 SHUFFLE 채택률도 33.8%→36.8%로 유지됐다. 따라서 rule v4의 생성·측정·
필터 규칙은 200편까지 재현됐다. 256개 비교쌍은 예비 TVM 실험에 사용할 수 있다.

300편으로 확대하기 전에는 다음을 먼저 결정한다.

1. 현재 partial chain의 이미 유효한 앞단 전이를 포함해 짧은 글 편향을 줄일지,
   또는 원문 길이 층화 quota를 둘지 비교한다.
2. 동일한 사전 고정 임계와 corpus artifact 자동 격리를 유지한다.
3. 300편까지 이상이 없으면 feature-only GBM과 텍스트 pairwise 모델의 100/200/300
   학습곡선을 비교하고, 더 늦게 포화하는 쪽으로 최종 규모를 정한다.
