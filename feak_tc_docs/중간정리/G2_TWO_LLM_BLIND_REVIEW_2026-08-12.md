# G2 Rule v3 — 2-LLM 블라인드 선호 평가

작성일: 2026-08-12
브랜치: `corruption_data`

## 결론

서로 독립된 두 LLM 평가자가 rule v3 블라인드 50쌍을 평가했다.

- LLM 1: 46/50, 92.0%
- LLM 2: 48/50, 96.0%
- 최초 평가자 간 완전 일치: 47/50, 94.0%
- 불일치 재검토 후 합의: 48/50쌍
- 합의된 48쌍 중 key 일치: 47/48, 97.9%
- 미합의 2쌍을 모두 오답 처리한 보수적 일치율: 47/50, 94.0%

따라서 **2-LLM proxy gate는 70% 기준을 통과**했다. 다만 평가자가 실제 사람이
아니므로 이 결과를 사람 선호 일치율로 쓰지 않는다. 공식 human gate 상태는 계속
`pending_human_review`다.

## 블라인드 절차

1. 동일한 50쌍을 평가자마다 서로 다른 순서로 제공했다.
2. 두 평가자는 서로의 입력·판정과 정답 key를 보지 않고 최초 판정을 완료했다.
3. 최초 결과 파일의 SHA-256을 고정한 뒤 세 불일치만 상대 근거와 함께 재검토했다.
4. 재검토가 끝난 후 처음으로 정답 key를 공개해 결과를 계산했다.

평가 입력에는 다음 정보가 없었다.

- 정답 A/B
- corruption operator
- cleaner/corrupted stage
- target rubric 및 점수 하락

초기 판정 파일 해시:

- LLM 1: `364b72f9848815b0ca0a00154f21cd63c16a443fa4c1ffb1a198ea31205c1957`
- LLM 2: `6b53f0cae291f923a9faaeb667f26aa30843f5ff735dea2ca0018b99bc852701`
- 정답 key: `c8992a7ebae43ac5062c822bbe7e67e440f8c42648de8b66063aaa83e17c723a`

## 최초 독립 평가

| 지표 | LLM 1 | LLM 2 |
|---|---:|---:|
| 정답 | 46/50 | 48/50 |
| 정확도 | 92.0% | 96.0% |
| Wilson 95% 구간 | 81.2%~96.8% | 86.5%~98.9% |
| TIE | 2 | 0 |

두 평가자의 최초 판정이 다른 문항은 `G2H-007`, `G2H-020`, `G2H-029`였다.

## 불일치 재검토

두 평가자에게 상대 평가의 선택과 근거만 공개하고 원문 A/B를 다시 읽게 했다.
정답 key는 이때도 공개하지 않았다.

- `G2H-029`: 두 평가자가 A로 합의했다. `DELETE_SPECIFICS`가 삭제한 문장이 뒤의
  지시어와 인과관계의 선행 내용이어서 A가 더 응집적이라는 판단이다.
- `G2H-007`: B 대 TIE로 미합의했다. 수학 답과 설명은 같고 일반 원칙 문장의 위치만
  달라, 순서 차이를 실질적 품질 차이로 볼지가 갈렸다.
- `G2H-020`: B 대 TIE로 미합의했다. 역사 기록의 주관성에 관한 결론 문장 위치만
  달라 두 배치의 자연스러움 판단이 갈렸다.

두 미합의 문항은 모두 `SHUFFLE_FLOW`, stage gap 1이다. 이는 이 operator가 만든
일부 순서 변화가 독자에게 명백한 훼손으로 체감되지 않는다는 기존 위험을 다시
확인한다.

## 합의 라벨과 다른 사례

두 평가자가 함께 A를 선택했지만 key는 B인 문항이 한 건 있었다.

### `G2H-044` — `DELETE_SPECIFICS`

- 과제: 국제기구가 필요한 이유
- key: B
- 두 LLM: A
- target drop: 0.7771 (`content_2`)

B는 NATO와 인터폴 사례를 포함해 더 구체적이지만, 세계은행 예시 문장이 여러 번
연속 반복된다. A는 해당 세부 내용과 함께 심한 반복도 제거한다. 두 평가자는 A가
세부성은 낮아도 전체 가독성과 조직성이 더 좋다고 독립적으로 판단했다.

이는 단순 평가 오류보다 **선호 라벨 역전 가능성**이 큰 사례다. target rubric은
의도대로 하락했어도 전체 글 선호는 상승할 수 있으므로, noise margin만으로는 이
문제를 걸러내지 못한다. `DELETE_SPECIFICS`에 다음 검사가 추가로 필요하다.

- 삭제 span에 기존 반복·어법·주제 이탈 결함이 포함됐는지 검사
- 포함됐다면 학습쌍과 사람/LLM 선호 평가 표본에서 제외
- target rubric 하락과 전체 선호를 별도 축으로 기록

## 판정

| Gate | 상태 |
|---|---|
| 2-LLM proxy, 기준 70% | **passed** |
| 실제 사람 2인 평가 | `pending_human_review` |

이번 결과는 corruption 방향이 대체로 사람이 읽는 품질 기준과 유사하다는 강한
보조 증거다. 동시에 `SHUFFLE_FLOW`의 체감 차이 부족과 `DELETE_SPECIFICS`의 라벨
역전이라는 남은 결함도 확인했다. 대량 생성 전에 두 유형을 자동 감사 대상으로
표시하는 것이 안전하다.

## 결과 파일

- LLM 1 최초 판정: `experiments/results/corruption_g2_gpt5mini_rulev3_llm_rater1_choices_50.jsonl`
- LLM 2 최초 판정: `experiments/results/corruption_g2_gpt5mini_rulev3_llm_rater2_choices_50.jsonl`
- 불일치 재검토: `experiments/results/corruption_g2_gpt5mini_rulev3_llm_reconsiderations.jsonl`
- 집계 보고서: `experiments/results/corruption_g2_gpt5mini_rulev3_llm_2rater_report.json`
