# SHUFFLE_FLOW 채점 민감도 진단

- 실행일: 2026-08-17
- 목적: 조직 채점기가 문장 순서 훼손에 반응하는지 확인하고 SHUFFLE_FLOW 존폐를 결정
- 입력: v3 파일럿 원문 10편
- 처리: corruption operator를 거치지 않고 각 글의 전체 문장 순서를 seed가 다른 3회 무작위화
- 측정: Kanana `m=10`, RF-corrected `organization_1`, 원문 점수 - 셔플 점수

## 결과

| 항목 | 결과 |
|---|---:|
| paired 측정 수 | 30 |
| 원문 수 | 10 |
| 평균 하락 | 0.647 |
| 중앙값 하락 | 0.685 |
| 최소 / 최대 | -0.041 / 1.519 |
| 양의 하락 비율 | 29/30 (96.7%) |
| 하락 > 0.3 | 22/30 (73.3%) |
| 하락 > 0.5 | 19/30 (63.3%) |
| target specificity 중앙값 | -0.442 |

원자료 및 전체 pair 결과는
`experiments/results/corruption_shuffle_sensitivity_v4_10x3_full_report.json`에 있다.

## 판단

조직 채점기가 문장 순서에 둔감하다는 가설은 기각한다. 전체 셔플에서
`organization_1`이 안정적으로 하락하므로 SHUFFLE_FLOW를 메인 체인에서 제거할
근거가 없다. v3의 낮은 채택률은 채점기보다 한 문장만 이동하던 operator 강도 문제로
보는 편이 타당하다.

다만 target specificity 중앙값이 음수이므로 전체 셔플 자체를 학습 데이터 생성
operator로 쓰지는 않는다. 전체 셔플은 상한 진단으로만 사용하고, 실제 v4 operator는
문장 집합을 보존하면서 서로 다른 원문 문장 2개를 멀리 이동해 기존 인접 관계 2개를
끊도록 제한한다. 최종 채택은 `organization_1` 하락 0.5 초과와 사람 블라인드 검수를
모두 통과한 전이만 대상으로 한다.

## 결정

- SHUFFLE_FLOW 유지
- `edits_per_step`: 1 → 2
- DELETE_SPECIFICS와 함께 구조 연산자는 rule-only로 실행
- 모든 moved/anchor span은 원문 유래인지 검증
- 문장 multiset 불변 및 각 moved 문장의 기존 predecessor 단절을 검증

