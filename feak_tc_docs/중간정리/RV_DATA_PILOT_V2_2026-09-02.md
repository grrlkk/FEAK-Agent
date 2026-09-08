# Revision Verifier 데이터 파일럿 v2 결과

날짜: 2026-09-02

기준 문서:

- `feak_tc_docs/docs/CODEX_01_RV_데이터파일럿_작업지시.md`
- `feak_tc_docs/docs/FEAK-TC_평가구조_RevisionVerifier_정리.md`

범위: synthetic RV 데이터 생성 가능성 재검증. **RV 모델 학습은 수행하지 않았다.**

## 결론

v1 300행 전체를 candidate type 고정 label로 학습하는 기존 no-go 판단은 유지한다. 다만
실패 후보를 선택 재생성하고 모든 transition을 instance별로 다시 판정한 결과,
**227행 filtered subset은 synthetic weak-supervision 파일럿으로 사용할 수 있다.**

판정은 다음과 같이 구분한다.

- 300행 전체: 학습용으로 사용 불가
- 후보 유형 품질 gate 통과: 254행
- 네 축 모두 3개 비생성 모델의 2/3 합의 확보: 227행, 75.7%
- 최종 판단: 파일럿 synthetic pretraining 실험에 한한 conditional go
- production 규모 확대 또는 실제 RV 학습 데이터 확정: 아직 보류

세 판정자 모두 OpenAI 모델이므로 완전 독립이 아니고, 사람 annotation도 아니다. 따라서 이
결과는 문서 §13의 Stage 1 synthetic pilot을 지지하지만 Stage 2 real revision calibration을
대체하지 않는다.

## v2에서 바꾼 것

### 선택 재생성

기존 trajectory/replay 후보는 보존했다.

- 유지: `correct_repair`, `partial_repair`, `further_corruption`, `no_edit`
- 전수 재생성: `wrong_target` 50개
- 조건부 재생성: `DELETE_SPECIFICS` / `ADD_DETAIL`의 `over_edit` 13개
- 유지: 나머지 `over_edit` 37개

GPT-5 mini 생성 결과는 다음 구조 조건을 먼저 통과시켰다.

- `wrong_target`: target corruption을 복원하지 않고 비대상 문장을 실질적으로 변경
- ADD_DETAIL `over_edit`: 삭제된 target 문장을 복원한 뒤 비대상 주장도 삭제/왜곡/추가
- trajectory state나 기존 후보의 exact copy 금지
- 최소 편집률 및 길이 범위 검사

첫 블라인드 평가에서 탈락한 42개는 강화 규칙으로 한 번 더 생성했다. 41개가 구조 검사를
통과했고, `train_50939:g1:stage2:wrong_target` 1개는 반복 재시도에도 5% 최소 편집률을 넘지
못해 강제로 합격시키지 않았다.

### Candidate type 품질 gate

생성 모델인 GPT-5 mini는 판정에서 제외했다. GPT-5와 GPT-4.1이 후보 A/B를 블라인드로
판정하고 **두 모델 모두 provenance type과 일치한 경우만** 생성 후보 gate를 통과시켰다.

| 생성 후보 | strict 통과 | 통과율 |
|---|---:|---:|
| `wrong_target` | 16/50 | 32% |
| `over_edit` | 38/50 | 76% |
| 합계 | 54/100 | 54% |

v1의 비생성 모델 동시 일치는 `wrong_target` 2/50, `over_edit` 37/50이었다. 선택 재생성으로
`wrong_target`은 16개까지 늘었지만 50개 전체를 살릴 수준은 아니었다. 강화 재생성 전후도
14개에서 16개로만 변해 반복 생성을 중단했다. 즉 **wrong-target 생성 문제는 완전히 해결되지
않았고 strict subset만 보존**했다. ADD_DETAIL over-edit도 여전히 어려워 생성 반복보다
사후 gate가 필요하다.

### Instance-level 재라벨링

v1의 candidate-type 고정 mapping과 `usable` 질문을 폐기했다. 각 state의 6개 후보를 무작위
`C1`~`C6`로 가리고 다음 세 비생성 모델이 후보마다 네 축을 직접 판정했다.

- `gpt-5-2025-08-07`
- `gpt-4.1-2025-04-14`
- `o3-2025-04-16`

축 정의도 다시 썼다.

- `target_fulfillment`: target 결함의 실제 해결 정도
- `preservation`: target 밖의 기존 주장/근거/의미 보존
- `edit_appropriateness`: intended action에 필요하고 비례적인 편집인지
- `action_consistency`: 관찰된 수정이 intended action/intent와 일치하는지

각 축은 `pass`, `partial`, `fail`이고 2/3 다수결로 확정했다. 세 모델이 모두 다른 경우 해당
축은 unresolved로 남겼다. LLM 생성 후보는 네 축 합의 외에 위 strict type gate도 통과해야
`training_eligible=true`가 된다.

## 최종 데이터

| candidate type | 전체 | quality gate 통과 | 최종 train |
|---|---:|---:|---:|
| `correct_repair` | 50 | 50 | 50 |
| `partial_repair` | 50 | 50 | 47 |
| `wrong_target` | 50 | 16 | 12 |
| `over_edit` | 50 | 38 | 22 |
| `further_corruption` | 50 | 50 | 46 |
| `no_edit` | 50 | 50 | 50 |
| 합계 | 300 | 254 | 227 |

최종 227행은 50개 state를 모두 포함한다. state당 후보는 최소 2개, 중앙값 5개, 최대 6개다.
중복 sample ID, 같은 state 안의 중복 후보 text, null label은 모두 0이다. 227행 전부 기존
`validate_rv_sample` schema 검사를 통과했다.

### 최종 label 분포

| label | pass | partial | fail |
|---|---:|---:|---:|
| target fulfillment | 73 | 41 | 113 |
| preservation | 167 | 21 | 39 |
| edit appropriateness | 57 | 47 | 123 |
| action consistency | 58 | 54 | 115 |

유형별 label도 고정 mapping이 아니라 후보별로 달라졌다. 중요한 sanity check는 다음과 같다.

- `correct_repair` 50개: 네 축 모두 pass
- `wrong_target` 12개: target/edit/action 모두 fail, preservation은 pass 9 / partial 3
- `over_edit` 22개: target pass 20, preservation pass 4 / partial 8 / fail 10,
  edit pass 2 / partial 9 / fail 11
- `no_edit` 50개: preservation pass 50, target/edit/action은 fail 49 / partial 1
- `further_corruption` 46개: target/edit/action fail 45 / partial 1

이 분포는 candidate type을 정답 label로 복사하지 않으면서도 RV가 구분해야 할 transition
신호를 보존한다.

## 판정자 일치도

| 항목 | Fleiss' kappa | 3자 완전 일치 |
|---|---:|---:|
| target fulfillment | 0.679 | 70.7% |
| preservation | 0.422 | 61.0% |
| edit appropriateness | 0.517 | 59.3% |
| action consistency | 0.557 | 60.3% |
| 관찰 candidate type | 0.525 | 45.7% |

v1의 kappa는 target 0.637, preservation 0.148, edit appropriateness 0.224, action consistency
0.434였다. 판정자 구성이 달라 직접적인 paired 비교는 아니지만, 운영 정의를 다시 쓴 뒤 약했던
세 축의 일치도가 모두 상승했다. 특히 preservation과 edit appropriateness가 낮은 일치도
구간에서 벗어났다.

합의가 없었던 후보 수는 `over_edit` 20, `wrong_target` 11, `further_corruption` 4,
`partial_repair` 3이다. 축별 unresolved는 target 3, preservation 30, edit 16, action 9로,
preservation이 여전히 가장 어려운 축이다.

## 해석

### 확인된 것

1. 기존 progressive corruption trajectory는 RV source state와 구조 후보 생성에 재사용할 수 있다.
2. 네 label을 candidate type에서 고정 복사하지 않고 instance별로 만들 수 있다.
3. 3개 비생성 LLM의 명시적 운영 정의와 2/3 합의로 파일럿 weak label을 만들 수 있다.
4. 후보 생성 품질과 label 합의를 분리한 gate가 작동한다.
5. cache는 공개 packet digest에 묶여 후보가 바뀐 state만 재평가한다.

### 아직 확인되지 않은 것

1. 227행이 사람 판정과 일치하는지는 측정하지 않았다.
2. 세 판정자는 같은 provider라 오류 상관이 있을 수 있다.
3. `wrong_target` 생성 recall은 여전히 32%라 전체 규모 생성 효율이 낮다.
4. 실제 Planner/LLM 수정 후보 분포에 대한 calibration은 수행하지 않았다.
5. 이 데이터로 RV를 학습했을 때 실제 후보 선택 성능이 오르는지는 아직 모른다.

따라서 다음 단계는 227행 전체를 바로 확장하는 것이 아니다. 먼저 candidate type과 축 label을
층화해 사람이 일부를 이중 annotation하고, 그 결과로 LLM 합의 label의 precision과 calibration을
측정해야 한다. 이후 실제 Planner가 만든 LLM revision 후보를 같은 schema로 사람 평가해
synthetic-to-real 차이를 보정한다.

## 산출물 및 재현

`experiments/results/`는 Git ignore 대상이므로 데이터와 model raw response는 workspace에
보존하고, 코드/설정/테스트/결과 문서만 Git에 기록한다.

- v2 staging 300행: `experiments/results/rv_data_pilot_50_v2_candidates.jsonl`
  - SHA-256: `a18b705b348de83f4de1567bf597566b8ff614b9474f8025e71b682f2546c1a9`
- 전수 relabel all 300행: `experiments/results/rv_data_pilot_50_v2_relabel_all.jsonl`
  - SHA-256: `e6ea16c454b9d30fc152d95045850e3b6e07ce2e0c1b0cf4b012ad01386e5cf2`
- 최종 train 227행: `experiments/results/rv_data_pilot_50_v2_relabel_train.jsonl`
  - SHA-256: `945d8b767b5582e415bab141dac3034f71bc230b1e10770d1c997dfda63ffdf8`
- 집계: `experiments/results/rv_data_pilot_50_v2_relabel_report.json`
- 모델별 raw response: 같은 prefix의 `_gpt5.jsonl`, `_gpt41.jsonl`, `_o3.jsonl`
- 공개 packet / hidden key: 같은 prefix의 `_public.jsonl`, `_hidden_key.jsonl`

재현 명령:

```bash
python scripts/rebuild_rv_data_pilot_v2.py
python scripts/evaluate_rv_pilot_llm_judges.py --config configs/rv_llm_judge_v2.yaml
python scripts/retry_rv_data_pilot_v2_failures.py
python scripts/relabel_rv_data_pilot_v2.py
```
