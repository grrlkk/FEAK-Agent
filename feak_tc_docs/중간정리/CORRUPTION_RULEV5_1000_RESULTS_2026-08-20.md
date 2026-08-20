# Corruption rule v5 최종 1,000쌍 학습 풀 결과

> 실행일: 2026-08-20
> 설정: `configs/corruption.yaml`
> 점수: Kanana m=10, RF-corrected
> 결론: 최종 1,000쌍 학습 풀이 모든 자동 품질 게이트를 통과했다.

## 입력 생성과 측정 무결성

기존 rule v5 clean-prefix 193쌍을 고정 base로 유지하고, 아직 사용하지 않은 source에서
추가 chain을 생성했다. 짧은 글을 일괄 제외해 생기는 길이 편향을 줄이기 위해 검증된
partial prefix도 포함했다.

- 추가 chain: 781개 (`ok` 698, `partial` 83)
- 추가 corruption transition: 2,221개
- 측정 상태: 3,002개
  - 원문 상태 781개
  - corruption 이후 상태 2,221개
- GPU 4개 측정 결과의 `(record_id, state_index)`와 chain이 요구하는 key가
  3,002/3,002로 정확히 일치
- 중복·누락·추가 측정 key: 0
- 생성 시 preservation 실패, fallback, normalization: 모두 0
- 생성 corpus exact/n-gram, BGE 의미 군집, provenance, relevance 감사 통과

추가 생성 operator는 `DELETE_SPECIFICS` 579, `INJECT_LEX_REPEAT` 530,
`INSERT_OFFTOPIC` 561, `SHUFFLE_FLOW` 551개였다. 최대 비중은 26.1%로 40% 상한을
통과했다.

## G1 필터 결과

사전 고정한 임계 `global > 0.225`, `INJECT_LEX_REPEAT > 0.4`와 DELETE 교차축
개선 가드를 그대로 적용했다.

- 채택: 1,040/2,221 = 46.8%
- 채택된 essay: 670개
- acceptance gate 재계산: 전 행 일치
- operator:
  - `DELETE_SPECIFICS`: 193/579 = 33.3%
  - `INJECT_LEX_REPEAT`: 324/530 = 61.1%
  - `INSERT_OFFTOPIC`: 328/561 = 58.5%
  - `SHUFFLE_FLOW`: 195/551 = 35.4%
- 채택 subset 최대 operator 비중: `INSERT_OFFTOPIC` 31.5%
- 반복 구조 cluster에 걸린 12개 transition은 corpus artifact로 전체 격리
- 격리 후 exact/n-gram violation 0
- BGE 의미 cluster violation, 동일 edit, provenance 누락, relevance 초과: 모두 0

## 최종 1,000쌍 조립

기존 clean base 193쌍은 전부 보존했다. 신규 채택 1,040쌍은 `target_drop` 내림차순으로
검토하면서 자연 key, transition ID, text pair, 생성 edit 중복과 기존 base edit의
downstream 재등장을 제거했다. 그중 807쌍을 추가해 목표 크기 1,000을 채웠다.

- 최종 transition: 1,000개
- essay: 683개
- 고유 transition ID: 1,000개
- 고유 자연 key `(chain_id, stage_k)`: 1,000개
- 고유 `(text_before, text)` pair: 1,000개
- 생성 edit: 1,298개, 모두 고유
- base edit가 신규 context에 재등장해 제외: 114개
- 생성 edit 중복으로 제외: 23개
- operator 40% 상한: 통과
- exact/n-gram artifact: 통과
- BGE 의미 artifact·provenance·relevance·cue 분포: 통과

| operator | n | 비율 | target drop 최소 | 중앙값 | 평균 | 최대 |
|---|---:|---:|---:|---:|---:|---:|
| `DELETE_SPECIFICS` | 181 | 18.1% | 0.231 | 0.598 | 0.773 | 2.493 |
| `INJECT_LEX_REPEAT` | 340 | 34.0% | 0.406 | 0.777 | 0.911 | 2.986 |
| `INSERT_OFFTOPIC` | 309 | 30.9% | 0.227 | 0.603 | 0.704 | 2.320 |
| `SHUFFLE_FLOW` | 170 | 17.0% | 0.244 | 0.429 | 0.519 | 1.648 |

## 산출물과 체크섬

`experiments/results/`는 `.gitignore` 대상이므로 데이터와 JSON 보고서는 현재 workspace에
보존하고, Git에는 이 결과 문서만 기록한다.

- 최종 학습 풀: `experiments/results/corruption_g1_rulev5_1000_training.jsonl`
  - SHA-256: `993ac439c82d04292b1f3ba7eac20eb34907829932c6aa28bd6d8fbfceca1094`
- 최종 풀 보고서: `experiments/results/corruption_g1_rulev5_1000_training_report.json`
  - SHA-256: `a285ba13095c1eb3fc8bf5444eef09e11dae0ef36ce22db3da9ff0fa7ad95b51`
- 추가 G1 audit: `experiments/results/corruption_g1_rulev5_1000_remaining_v2_audit.jsonl`
- 추가 accepted: `experiments/results/corruption_g1_rulev5_1000_remaining_v2_accepted.jsonl`
- 추가 G1 요약: `experiments/results/corruption_g1_rulev5_1000_remaining_v2_summary.json`
- 추가 품질 보고서: `experiments/results/corruption_g1_rulev5_1000_remaining_v2_quality.json`

## 검증

- `pytest -q`: 134 passed, 2 warnings
- 최종 JSONL line count: 1,000
- 독립 `jq` 재검산: transition ID, 자연 key, text pair 각각 1,000개 고유
- accepted가 아닌 행: 0

## 다음 단계

Corruption Stage-1 데이터 생성은 완료했다. 다음 작업은 같은 essay-group split으로
feature-only GBM과 text pairwise 모델의 학습곡선을 비교한 뒤, 더 늦게 포화하는 모델을
기준으로 TVM 학습 규모와 구성을 확정하는 것이다.
