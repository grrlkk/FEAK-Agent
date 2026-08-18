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

## 다음 단계

1. 2-LLM API 프록시 게이트 통과에 따라 100편 생성
2. artifact/balance/acceptance와 `SHUFFLE_FLOW` 식별력 재평가
3. 이상이 없을 때 200편, 300편으로 순차 확대
4. feature-only GBM과 텍스트 pairwise 모델의 학습곡선을 모두 측정해 더 늦은 포화점을
   최종 데이터 규모 기준으로 사용

기존 v3 채택 42쌍은 오염 분석 기록으로만 보존하고 TVM 학습에는 사용하지 않는다.
