# RV 데이터 진행 현황 및 다음 작업 — 2026-09-08

> 점검일: 2026-09-08. 마지막 확인된 데이터 생성·재라벨링 결과는 2026-09-02이다.
> 코드 기준: `rv-data-pilot`의 `82aca27` 및 해당 작업 디렉터리의 결과 파일.
> 현재 방법론은 [RV 중심 방법론](../docs/FEAK_TC_RV_METHOD_2026-09-08.md)을 따른다.
> 이번 작업은 문서 정리다. 데이터 재생성, LLM 재판정, RV 학습은 실행하지 않았다.

## 1. 현재 위치

**기존 corruption 생성과 RV 데이터 파일럿은 완료했고, RV 학습과 실제 수정에서의 효과 검증은
아직 수행하지 않았다.** 현 파일럿을 재사용하면서 RV의 역할에 맞는 수정 요구·비교 예제를 보강할
단계다. 227개 후보의 존재와 LLM 합의가 최종 학습 데이터의 충분성이나 사람 기준 타당성을
증명하지는 않는다.

| 항목 | 상태 | 확인된 결과 |
|---|---|---|
| FEAK·한 단계 휴리스틱 MVP | 완료 | 진단, 후보 생성, reversible patch, validity, transition 계산 |
| Corruption rule v5 풀 | 완료 | 683개 essay, 1,000개 transition |
| Corruption 규모 검토 | 완료 | 학습곡선 결과에 따라 같은 방식의 추가 생성 중단 |
| 기존 TVM 실험 | 완료·현재 개발 종료 | 학습·합성 평가·실제 수정 파일럿 기록 보존 |
| Corruption → RV source 연결 | 완료 | 1,000/1,000 transition과 raw 이력 연결 |
| RV v1 파일럿 | 완료·고정 라벨 방식 폐기 | 50개 essay/state, 300개 후보 |
| RV v2 후보 재생성·재라벨링 | 완료 | 유형 gate 254개, 최종 선택 227개 |
| RV 사람 검수 | 결과 미확인 | 사용자가 검수 의사를 밝혔으나 제출된 판정 결과 없음 |
| 구체적 수정 요구·새 비교 세트 | 설계 정리, 미구현 | 원문과 다른 성공, 통제된 실패·편집량 반례 등 |
| RV 모델 학습·real calibration | 미실행 | backbone·학습 설정·최종 split 미확정 |
| RV Controller·Trajectory Guard | 미통합 | 현재 실행 경로는 한 단계 휴리스틱 MVP |

## 2. 기존 corruption은 어떻게 만들어졌는가

### 출발 글과 조작

출발 글은 AI-Hub 데이터의 사람 채점자 2인 평균 점수로 질문별 상위 약 25%를 선정했다.
선별 코드에는 300~2,500자·6문장 이상 조건, 과제별 층화, 기존 Stage-A 글 제외와 별도
held-out 풀 분리가 있다. 출발 글의 높은 점수가 이후 모든 수정의 정답을 보장하지는 않는다.

설정상 한 chain에 서로 다른 operator를 최대 3단계 적용하고, 단계당 두 편집을 기록한다.
각 상태와 편집 문자열, operator, target rubric, 대응 action, 생성 출처를 보관한다.

| 조작 | 생성 방식 | 대응 action | 최종 transition 수 |
|---|---|---|---:|
| `DELETE_SPECIFICS` | 규칙으로 사례·근거 문장 삭제 | `ADD_DETAIL` | 181 |
| `SHUFFLE_FLOW` | 규칙으로 문장 위치 이동 | `RESTRUCTURE` | 170 |
| `INSERT_OFFTOPIC` | 다른 과제의 문장 검색·삽입 | `DELETE_OR_FOCUS` | 309 |
| `INJECT_LEX_REPEAT` | GPT-5 mini 반복 문장 생성 | `STYLE_REFINE` | 340 |
| 합계 | | | 1,000 |

`COMPRESS` 전용 corruption은 현재 이 풀에 없다. 맞춤법 오류 주입은 별도 surface 검증용으로,
위 주 chain의 RV 학습 범위에 포함되지 않는다.

### 생성 후 선별

- 텍스트 편집·보존 조건, 중복·반복 템플릿, 주제 무관성·의미 군집 등 자동 검사.
- Kanana m=10의 RF-corrected 목표 rubric 하락이 `0.225`를 초과해야 통과.
- 어휘 반복은 별도로 `0.4` 초과 하락을 요구.
- 삭제가 비대상 품질을 개선하는 혼동 사례를 추가 검사.
- 기존 base 193쌍과 신규 807쌍을 합쳐 최종 1,000쌍 구성.

따라서 이 풀은 채점기 기반 선별을 거친 합성 데이터다. 채점기와 완전히 독립적인 사람 선호
데이터로 해석하지 않는다. 실제 수정에서의 RV 타당성은 별도 평가해야 한다.

근거: [원문 선별 코드](../../scripts/select_corruption_sources.py),
[생성 설정](../../configs/corruption.yaml),
[최종 1,000쌍 보고서](CORRUPTION_RULEV5_1000_RESULTS_2026-08-20.md).

## 3. RV 파일럿에 재사용한 범위

기존 풀의 flat transition을 raw chain과 연결해 원문·전체 상태·직전/다음 상태·편집 이력을 복원했다.

- transition 본문과 step metadata가 맞는 raw 출처 확인: 1,000/1,000.
- 다음 상태까지 남아 있는 transition: 653개, 535개 essay.
- 그중 50개 essay에서 state 하나씩 선택: stage 1 25개, stage 2 25개.
- 조작 분포: DELETE 13, LEX_REPEAT 13, OFFTOPIC 12, SHUFFLE 12.

같은 현재 글에 여섯 후보를 구성해 총 300행을 만들었다.

| 후보 유형 | 현재 파일럿의 생성 방법 |
|---|---|
| `correct_repair` | 직전 trajectory state 사용 |
| `partial_repair` | 직전 state에 두 corruption edit 중 하나만 재적용 |
| `wrong_target` | GPT-5 mini로 목표 결함을 남기면서 다른 부분 수정 |
| `over_edit` | GPT-5 mini로 목표를 다루면서 비대상 내용도 과도하게 변경 |
| `further_corruption` | 다음 trajectory state 사용 |
| `no_edit` | 현재 state 그대로 사용 |

직전 상태는 완벽한 정답문이 아니다. 현재 `x2`에 대한 `correct_repair=x1`에는 이전 단계의
결함이 남을 수 있다. `further_corruption`은 다음 단계의 다른 결함을 넣은 글일 수 있으므로
네 축의 라벨을 유형 이름만으로 고정할 수 없다.

근거: [RV v1 audit·생성 보고서](RV_DATA_PILOT_2026-08-31.md).

## 4. v1 검증에서 드러난 문제와 v2 조치

### v1: 후보 유형별 고정 라벨은 사용 불가

v1은 `correct_repair → 모든 축 pass` 같은 고정 mapping을 사용했다.
생성 후보 100개에 대한 3-LLM 검증에서 실제 수정과 mapping의 불일치가 확인됐다.

- 비생성 두 모델이 모두 intended type에 동의한 `wrong_target`: 2/50.
- 같은 기준의 `over_edit`: 37/50.
- 유형이 같아도 편집 적절성·내용 보존은 후보마다 달랐다.
- ‘학습에 쓸 만한가’라는 단일 usable 질문도 판정 기준으로 부적합했다.

이 고정 mapping은 최신 라벨의 근거로 사용하지 않는다. 과거 라벨은 `legacy_labels_v1`에
보관되어 있으며, 최신 네 축 판정과 구분해야 한다.

### v2: 선택 재생성 + instance별 라벨

- `wrong_target` 50개 전수와 ADD_DETAIL의 `over_edit` 13개를 재생성.
- 나머지 trajectory/replay 후보와 over-edit 37개는 보존.
- 실패 후보에 대한 강화 재생성도 수행했으나 wrong-target 통과가 14→16개에 그쳐 반복 중단.
- 생성 모델 GPT-5 mini를 제외한 GPT-5·GPT-4.1이 유형 gate를 판정.
- GPT-5·GPT-4.1·o3가 300개 후보의 네 축을 직접 판정하고 축별 2/3 합의를 집계.

세 라벨 판정 모델은 같은 provider의 모델이다. 별도 모델의 합의는 사람 판정이나 완전히
독립적인 검증으로 간주하지 않는다.

근거: [v1 블라인드 검증](RV_THREE_LLM_BLIND_REVIEW_2026-09-01.md),
[v2 결과](RV_DATA_PILOT_V2_2026-09-02.md).

## 5. 현재 227개 후보의 의미

| 후보 유형 | 전체 | 유형/출처 gate 통과 | 최종 선택 |
|---|---:|---:|---:|
| `correct_repair` | 50 | 50 | 50 |
| `partial_repair` | 50 | 50 | 47 |
| `wrong_target` | 50 | 16 | 12 |
| `over_edit` | 50 | 38 | 22 |
| `further_corruption` | 50 | 50 | 46 |
| `no_edit` | 50 | 50 | 50 |
| 합계 | 300 | 254 | 227 |

254개는 유형/출처 gate 통과 수이고, 227개는 그중 네 축 모두 합의가 있는 후보 수다.
`training_eligible=true`는 **현재 v2 규칙을 통과했다**는 뜻이다. 사람 검수가 끝났거나,
RV 학습의 충분성이 검증됐거나, 새로운 방법론의 요건을 모두 만족한다는 뜻은 아니다.

최종 집합에는 50개 state가 모두 남아 있고, 후보 수는 state당 2~6개다.
보고서 기준 sample ID 중복, state 내부 동일 후보, null label은 모두 0이며 schema 검사를 통과했다.

| 축 | pass | partial | fail | 300개 기준 Fleiss' κ |
|---|---:|---:|---:|---:|
| `target_fulfillment` | 73 | 41 | 113 | 0.679 |
| `preservation` | 167 | 21 | 39 | 0.422 |
| `edit_appropriateness` | 57 | 47 | 123 | 0.517 |
| `action_consistency` | 58 | 54 | 115 | 0.557 |

라벨 분포는 최종 227개, κ는 전수 300개에 대한 판정자 간 일치도다. κ는 모델 정확도나
사람 정답과의 일치율이 아니다. preservation의 불확실성이 상대적으로 크게 남아 있다.

현재 구현은 생성 후보의 intended type에 두 모델이 동의해야 학습 대상이 된다.
새 방법론에서 논의한 ‘유형이 달라도 실제 축별 판정이 가능하면 보존’하는 방향은 아직 코드에
반영되지 않았다. 탈락한 73개도 전수 파일에 보관되어 있으며 새 기준으로 자동 복귀시키지 않았다.

## 6. 보유한 실제 수정 데이터와 기존 TVM 결과

별도로 20개 글에서 5개 action의 GPT-5 mini 후보 100개를 생성했고, validity 검사 후
99개 후보가 남았다. 당시 GPT-5 mini가 두 번 판정했다. 사람 평가도, 최신 RV 네 축 라벨도 아니다.

| 과거 TVM 지표 | Qwen | Kanana |
|---|---:|---:|
| 합성 test pairwise accuracy | 83% | 95% |
| 두 차례 판정 모두 개선인 top-1 | 4/20 (20%) | 7/20 (35%) |
| top-1을 한 차례라도 악화로 판정 | 8/20 (40%) | 9/20 (45%) |

합성 성능이 실제 수정 선택으로 충분히 이어졌다는 근거를 얻지 못했다.
이 결과는 TVM을 계속 튜닝하기보다 RV의 수정 검증 역할을 명확히 하는 전환 배경이다.
RV가 더 낫다는 실험 결과는 아직 없다.

99개 후보는 새로운 schema 점검·사람 라벨링·개발 분석에 재사용할 수 있다.
이미 분석한 데이터이므로 최종 미공개 평가셋으로 다시 취급하지 않는다.
생성 모델과 판정 모델이 같다는 한계도 기록한다.

근거: [실제 수정 20건 보고서](../../experiments/results/TVM_REAL_GPT5_MINI_20_REPORT_2026-08-24.md).
이 보고서는 Git ignore 경로에 있는 로컬 산출물이며, 위 핵심 결과를 본 문서에도 남겼다.

## 7. 파일 위치와 용도

경로는 저장소 루트 기준이다. 데이터·raw 응답은 `experiments/results/` 아래에 있으며 Git ignore 대상이다.
따라서 코드만 clone하면 자동으로 따라오지 않는다. 다음 파일의 존재와 주요 행 수·해시는
2026-09-08 작업 디렉터리에서 확인했다.

| 용도 | 파일 | 크기 단위 |
|---|---|---|
| 기존 corruption 풀 | [corruption_g1_rulev5_1000_training.jsonl](../../experiments/results/corruption_g1_rulev5_1000_training.jsonl) | 1,000 transition |
| RV v1 원본 | [rv_data_pilot_50.jsonl](../../experiments/results/rv_data_pilot_50.jsonl) | 300 후보, 고정 라벨은 과거 기록 |
| v2 후보 생성본 | [rv_data_pilot_50_v2_candidates.jsonl](../../experiments/results/rv_data_pilot_50_v2_candidates.jsonl) | 재라벨링 입력 300 후보 |
| 최신 전수 판정 | [rv_data_pilot_50_v2_relabel_all.jsonl](../../experiments/results/rv_data_pilot_50_v2_relabel_all.jsonl) | 300 후보, 탈락·불일치 포함 |
| 최신 선택 subset | [rv_data_pilot_50_v2_relabel_train.jsonl](../../experiments/results/rv_data_pilot_50_v2_relabel_train.jsonl) | 227 후보 |
| 공개 검수 packet | [rv_data_pilot_50_v2_relabel_public.jsonl](../../experiments/results/rv_data_pilot_50_v2_relabel_public.jsonl) | 50행, 각 행 C1~C6 |
| 코드↔샘플 대응표 | [rv_data_pilot_50_v2_relabel_hidden_key.jsonl](../../experiments/results/rv_data_pilot_50_v2_relabel_hidden_key.jsonl) | 독립 판정 후 연결용 |
| 최신 집계 | [rv_data_pilot_50_v2_relabel_report.json](../../experiments/results/rv_data_pilot_50_v2_relabel_report.json) | 분포·합의·제외 통계 |
| 실제 후보 feature 파일 | [tvm_real_pairs_gpt5_mini_20_features.jsonl](../../experiments/results/tvm_real_pairs_gpt5_mini_20_features.jsonl) | 99 후보, RV 라벨 미생성 |

raw chain은 [RV 입력 설정](../../configs/rv_pilot.yaml)의 `input.raw_chains`에 출처 우선순위가
명시되어 있다. 주요 파일은 `experiments/results/migrated_3080/`에 있다.
LLM 원응답은 최신 전수 판정과 같은 prefix의 `_gpt5.jsonl`, `_gpt41.jsonl`, `_o3.jsonl`이다.

확인한 SHA-256:

```text
corruption_g1_rulev5_1000_training.jsonl
993ac439c82d04292b1f3ba7eac20eb34907829932c6aa28bd6d8fbfceca1094

rv_data_pilot_50_v2_candidates.jsonl
a18b705b348de83f4de1567bf597566b8ff614b9474f8025e71b682f2546c1a9

rv_data_pilot_50_v2_relabel_all.jsonl
e6ea16c454b9d30fc152d95045850e3b6e07ce2e0c1b0cf4b012ad01386e5cf2

rv_data_pilot_50_v2_relabel_train.jsonl
945d8b767b5582e415bab141dac3034f71bc230b1e10770d1c997dfda63ffdf8
```

## 8. 사람 검수의 현재 상태와 필요한 준비

현재 `public.jsonl`은 모델 판정에 사용한 공개 packet이며, 사람이 쉽게 입력하도록 만든 완성된
검수 화면이나 스프레드시트가 아니다. 이번 문서 작업에서도 별도 RV 입력 양식은 만들지 않았다.

다음 작업에서 필요한 양식:

- 과제, 현재 글, 구체적인 수정 요구, 수정 후보를 읽기 편하게 제시.
- `review_id`와 `candidate_code`를 유지해 hidden key로 원본에 연결.
- 네 축의 빈 입력란, 판정 근거, 판단 불가·정의 모호성 기록란 제공.
- 기존 LLM 라벨·후보 유형을 보지 않고 먼저 판정하고, 이후 비교.
- 원본 데이터에 덮어쓰지 않고 별도 사람 판정 파일로 저장.

검수 목적을 구분해야 한다.

| 목적 | 제공 정보 | 해석 |
|---|---|---|
| 기존 합성 라벨 audit | 현재 글·후보·요구 + 복원 참고문·편집 이력 | 생성 과정과 축별 라벨의 타당성 확인 |
| 실제 RV 평가 | 운영 가능한 현재 글·후보·구체적인 요구 | 숨은 원문 없이 수정 성공을 판단하는 능력 확인 |

현재 공개 packet에는 `reference_repair`와 `known_corruption_edits`가 포함된다.
따라서 이것을 그대로 읽은 판정은 첫 번째 조건이다. 라벨·유형만 가린 검수와 복원 참고문 없는
실사용 평가를 혼동하지 않는다.

소규모 예제는 정의를 다듬는 데 사용할 수 있지만 ‘30개면 검증이 충분하다’는 기준은 확정하지 않았다.
새 비교 세트 검수와 기존 라벨 audit에는 탈락·불일치 사례도 포함해 쉬운 사례만 남는 편향을 확인한다.
최종 사람 평가의 표본 수·판정자 수·합의 절차는 실험 계획에서 별도로 고정해야 한다.

## 9. 현재 데이터로 아직 확인하지 못한 것

1. 227개 라벨이 사람 판단과 얼마나 일치하는가.
2. 원문과 다르게 타당하게 고친 수정도 RV가 통과시킬 수 있는가.
3. 목표 해결과 내용 훼손을 분리해서 배울 수 있는 비교 예제가 충분한가.
4. 편집량·문체·생성 유형 같은 표면 단서만으로 풀리는 과제가 아닌가.
5. 현재 `intent`보다 구체적인 수정 요구를 실제 Planner가 제공할 수 있는가.
6. 원문 복원 정보 없이 실제 수정 후보를 검증할 수 있는가.
7. RV를 추가하면 유효한 수정 채택을 유지하면서 유해한 채택이 줄어드는가.

반복 지연의 주된 원인은 생성 후보를 여섯 유형에 맞추고 LLM 판정을 반복하는 동안,
이 질문들을 직접 평가하는 기준과 실제 사람 평가가 남아 있었다는 점이다.

## 10. 다음 작업의 범위와 종료 조건

| 순서 | 작업 | 완료 산출물 |
|---|---|---|
| 1 | 기존 state 일부와 실제 후보를 이용해 구체적인 수정 요구 정리 | 과제·문제 위치·목표·보존 조건이 있는 샘플 |
| 2 | 원문과 다른 성공과 통제된 실패를 포함한 비교 세트 구성 | 목표/보존 불일치·편집량 반례가 있는 후보 묶음 |
| 3 | 사람이 입력할 양식과 판정 기준 제공 | 라벨·유형을 가린 입력물, 독립 판정 결과 |
| 4 | 불일치 원인을 반영해 schema·학습/평가 분할·최소 학습 설정 고정 | 실행 전 고정한 설정과 중단 기준 |
| 5 | RV 최소 학습과 실제 수정 평가 | 축별 지표, FEAK에 추가했을 때의 효과 |
| 6 | 효과 확인 후 반복 Controller와 guard 통합 | 채택·거절·복구·종료 평가 |

종료·보존 원칙:

- 기존 1,000쌍은 유지하며 전체 corruption을 다시 만들지 않는다.
- 227개는 v2 시점의 subset으로 보존한다. 새 기준 적용 결과는 새 버전으로 남긴다.
- 300개 전수 LLM 재판정이나 wrong-target 반복 생성은 구체적인 변경·검증 목적이 있을 때만 한다.
- 유형을 맞추기 위한 생성 성공률을 RV 학습의 주 성과로 삼지 않는다.
- 기존 TVM의 학습률·backbone 탐색을 다시 시작하지 않는다.
- 새로운 데이터의 수량 확대는 정의·비교 사례·실제 수정 평가의 부족이 확인된 부분에 한정한다.

## 11. Git 상태와 문서 인계

2026-09-08 확인 시 RV 작업은 `rv-data-pilot`에 있고,
[PR #20](https://github.com/grrlkk/FEAK-Agent/pull/20)은 `main` 대상으로 열려 있었다.
TVM 구현은 별도 `tvm-stage1-cross-backbone` 브랜치에 있다. 현재 RV 브랜치에 TVM 소스가
없다는 사실만으로 TVM 학습이 미실행이라고 판단하지 않는다.

작업 시작 문서의 과거 ‘지금: TVM Stage-1’ 표기는 최신 방향과 맞지 않아 새 기준 문서로 갱신한다.
`HANDOFF_CODEX.md`는 이전 TVM 학습률 확장 작업의 인계 기록이며 최신 실행 지시로 사용하지 않는다.

## 12. 관련 코드·보고서

- [Corruption 학습곡선과 생성 중단 판단](CORRUPTION_LEARNING_CURVES_2026-08-20.md)
- [RV source 연결·후보 구성](../../feak_tc/rv/pilot.py)
- [v2 재생성](../../feak_tc/rv/rebuild.py)
- [v2 유형 gate·축별 재라벨링](../../feak_tc/rv/relabel.py)
- [v2 재라벨링 설정](../../configs/rv_relabel_v2.yaml)
- [한 단계 휴리스틱 실행 경로](../../feak_tc/mvp/loop.py)

원문 파일 확인과 문서 대조만 수행했으며, 과거 보고서의 테스트·LLM 검증을 이번에 재실행한
것으로 기록하지 않는다.
