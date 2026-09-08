# Revision Verifier 데이터 파일럿 결과

> 실행일: 2026-08-31
> 기준 문서: `docs/FEAK-TC_평가구조_RevisionVerifier_정리.md`
> 범위: 데이터 생성 가능성 검증만 수행. RV 모델은 학습하지 않음.

## 결론

기존 progressive corruption 1,000개 transition은 RV 데이터의 source state로 전부
재사용할 수 있다. corruption 자체를 다시 생성할 필요는 없다. 다만 flat training pool에
없는 `x0`, 전체 state 순서, 이전/다음 state ID, canonical changed span은 raw chain과의
metadata join으로 보강해야 한다.

50개 essay에서 6개 후보씩 총 300행 파일럿을 생성했다. `correct_repair`,
`further_corruption`, `no_edit`는 trajectory를 직접 재사용했고, `partial_repair`는 한
corruption step에 기록된 2개 edit 중 1개만 재적용해 정확히 만들었다. 따라서 LLM은
trajectory에서 얻을 수 없는 `wrong_target`, `over_edit`에만 사용했다.

이 파일럿은 RV용 synthetic pretraining 데이터 생성이 **가능함**을 확인한 결과이지,
300행 전체가 human ground truth라는 뜻은 아니다. 특히 GPT 생성 후보의 의미적 유형과
candidate-type 기반 label은 실제 수정쌍으로 calibration하기 전에 사람 검수 표본이 필요하다.

## Corruption 데이터 Audit

입력은 `corruption_g1_rulev5_1000_training.jsonl` 1,000행, 683개 essay다. 설정에
명시한 provenance 우선순위에 따라 raw chain 1,358개를 검사했다.

- training transition과 raw `(states[k-1], states[k])`가 정확히 일치: 1,000/1,000
- raw step의 operator/rubric/action/intent/edits까지 일치하는 source 선택: 1,000/1,000
- 다음 state가 있어 `further_corruption`으로 쓸 수 있는 transition: 653개, 535 essay
- 여러 raw version에서 같은 transition이 확인된 행: 122개
- version별 다음 state가 달랐던 행: 37개
- 최종 provenance 선택: remaining v2 807, retrieval prefix 129, rule v4 64

동일 transition의 raw version이 여러 개인 경우 설정 파일의 최신 provenance 순서를
사용한다. 선택된 source는 transition 본문과 step metadata가 모두 일치한다.

| 필드 | 판단 | 근거 |
|---|---|---|
| `essay_id` | 그대로 사용 가능 | 1,000/1,000 존재 |
| `corruption_type` | 그대로 사용 가능 | `corruption_op` 1,000/1,000 |
| `target_rubric` | 그대로 사용 가능 | measured row와 raw step 일치 |
| `before_text`, `after_text` | 그대로 사용 가능 | raw 인접 state와 exact match |
| `x0` | metadata 보강 필요 | pool에는 없고 raw `states[0]`에서 1,000/1,000 복원 |
| `x1, x2, ...` | metadata 보강 필요 | pool은 flat하며 raw state 배열 join 필요 |
| `changed_span` | metadata 보강 필요 | 모든 행에 textual edit 2개가 있으나 canonical 필드/offset 없음 |
| 이전/다음 state link | metadata 보강 필요 | raw index로 ID 생성, next는 653행에서만 존재 |
| `question` | metadata 보강 필요 | pool 878행, raw join 후 1,000행 |
| corruption source | 재생성 불필요 | 1,000행 모두 exact resolution 성공 |

상세 기계 판독 audit는 `experiments/results/rv_data_pilot_50_audit.json`에 있다.

## Candidate 구성

| candidate type | 생성 방법 | source |
|---|---|---|
| `correct_repair` | 현재 corruption 직전 state | `trajectory_previous` |
| `partial_repair` | 2개 corruption edit 중 첫 edit만 이전 state에 재적용 | `corruption_edit_replay` |
| `wrong_target` | target defect를 유지하고 명시한 non-target rubric만 수정 | GPT-5 mini |
| `over_edit` | target을 복구하면서 non-target 내용을 과도하게 변경 | GPT-5 mini |
| `further_corruption` | raw trajectory의 다음 state | `trajectory_next` |
| `no_edit` | 현재 corrupted state 그대로 | `trajectory_current` |

LLM 모델은 `gpt-5-mini-2025-08-07`이다. strict JSON Schema를 사용하고, trajectory
state 또는 다른 후보를 그대로 복사한 출력은 거절한다. 실패 시 전체 triplet이 아니라
문제가 있는 후보 필드만 다시 생성하며, 성공 결과는 resume cache에 기록한다.

## Dataset Schema

필수 학습 필드는 다음과 같다.

- 식별/연결: `essay_id`, `chain_id`, `state_id`, `stage_k`, `previous_state_id`,
  `next_state_id`, `source_transition_id`, `sample_id`
- transition 입력: `question`, `before_text`, `after_text`, `target_rubric`,
  `intended_action`, `intent`, `corruption_type`, `changed_spans`
- 후보: `candidate_type`, `candidate_source`, `provenance`
- 핵심 label: `target_fulfillment`, `preservation`, `edit_appropriateness`
- 보조 label: `action_consistency`
- label 성격: `weak_supervision=true`, `label_source`

Label vocabulary는 `pass`, `partial`, `fail`이다. `edit_appropriateness`는 현재 RV
문서의 edit minimality/appropriateness를 데이터 필드 하나로 표현한다. 전체 JSON Schema는
`experiments/results/rv_data_pilot_50_schema.json`에 생성된다.

## Pilot 분포

- essay/state: 50/50, essay당 anchor transition 1개
- candidate row: 300, candidate type별 50개
- stage: stage 1 = 25, stage 2 = 25
- corruption/rubric:
  - `DELETE_SPECIFICS` / `content_2`: 13
  - `INJECT_LEX_REPEAT` / `expression_1`: 13
  - `INSERT_OFFTOPIC` / `organization_2`: 12
  - `SHUFFLE_FLOW` / `organization_1`: 12
- candidate source: trajectory 150, exact edit replay 50, GPT-5 mini 100

| label | pass | partial | fail |
|---|---:|---:|---:|
| target fulfillment | 100 | 50 | 150 |
| preservation | 200 | 50 | 50 |
| edit appropriateness | 100 | 50 | 150 |
| action consistency | 100 | 50 | 150 |

## 구조 검증

- 50개 state 모두 6개 candidate type 보유
- 300개 `sample_id` 모두 고유
- state 내부 candidate text 중복 0
- 모든 label에 `weak_supervision=true`
- partial edit replay가 exact repair보다 적은 문자 변경: 50/50
- over-edit가 partial보다 큰 문자 변경: 50/50
- over-edit가 exact repair보다 큰 문자 변경: 49/50
- wrong-target이 NO-EDIT가 아님: 50/50
- `origin/main` 기준 전체 repository test: 145 passed, 1 NVML warning

문자 편집률은 구조 검증 신호일 뿐 의미적 label의 정답 판정은 아니다. 특히 shuffle의
exact repair는 문장 이동 때문에 문자 alignment 변화가 커서 over-edit보다 편집률이 조금
클 수 있다.

## 산출물

`experiments/results/`는 저장소 정책상 Git ignore 대상이므로 데이터 파일은 workspace에
보존하고, 생성 코드·설정·테스트와 이 결과 문서만 Git에 기록한다.

- dataset: `experiments/results/rv_data_pilot_50.jsonl`
  - 300행, 1.6MB
  - SHA-256: `1593c9757558ef64459b8844dc0bb080667223bc7c3cee623a63c4e33dcd40d1`
- audit: `experiments/results/rv_data_pilot_50_audit.json`
  - SHA-256: `02d3edf0e6a0e5c7a976cec7860987e98103ad2bac17bb12d60d6947aeb3e36d`
- schema: `experiments/results/rv_data_pilot_50_schema.json`
  - SHA-256: `0a85aa52c6cc75d1440740f9b29927060469f56b16bfe91599fe34cd53baf575`
- distribution/QC report: `experiments/results/rv_data_pilot_50_report.json`
- resumable LLM cache: `experiments/results/rv_data_pilot_50_llm_cache.jsonl`

재현 명령:

```bash
python scripts/build_rv_data_pilot.py --workers 4
```

## 다음 검증 경계

전체 규모 생성이나 RV 학습으로 바로 넘어가면 안 된다. 다음 단계는 candidate type별
층화 표본을 사람이 확인해 `wrong_target`과 `over_edit`의 의미적 적합률, 그리고 네 weak
label의 calibration 오차를 측정하는 것이다. 이 검증 전까지 본 파일럿은 synthetic
pretraining 후보 데이터로만 취급한다.
