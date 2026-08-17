# FEAK-TC MVP 최종 보고서

작성일: 2026-07-23

상태: **MVP 완료 — corruption 데이터 생성 단계 진입 조건 충족**

## 1. 보고서 범위

이 문서는 FEAK-TC의 **one-step revision MVP**와 그 후속 신뢰성 개선을 하나로
정리한 최종 보고서다. 다음 범위를 완료된 MVP로 본다.

- 실제 Kanana 채점기와 독립 FEAK 자질 계산기를 연결한 진단
- action별 국소 수정 후보 생성
- 원문을 보존하는 reversible patch
- 수정 전후 재진단과 transition feature 계산
- 구조·환각·중복·의미 훼손 후보의 사전 차단
- 학습되지 않은 휴리스틱에 의한 accept/reject/stop
- 단건 및 배치 실행, 후보별 감사 가능한 JSONL 로그

TVM 학습, corruption 데이터 생성, 다단계 제어, drift/rollback은 MVP 이후
단계이므로 이 보고서의 완료 판정에서 제외한다. 저장소에 이미 존재하는 corruption
파일럿 구현과 결과는 별도 문서
[`CORRUPTION_GEN_DECISIONS_2026-07-22.md`](CORRUPTION_GEN_DECISIONS_2026-07-22.md)에서
다룬다.

## 2. 최종 결론

FEAK-TC MVP는 한국어 논술 한 편에 대해 아래 한 바퀴를 실제로 수행한다.

```text
원문
  → Kanana 8-rubric + 독립 FEAK 29-feature 진단
  → 약점 rubric 선정
  → 5개 수정 action별 후보 생성
  → target span 고정 국소 patch
  → 구조·환각·중복 validity 검사
  → 수정본 재진단
  → transition feature 계산
  → hard constraint + 휴리스틱 선택
  → accept / reject_all / stop + 전체 로그
```

최종 100편 실험에서는 100편 모두 오류 없이 처리했고, 500개 후보를
BGE-M3로 평가했다. 56편에서 수정이 채택됐으며 채택 결과의 자동 스캔
플래그는 baseline 21건에서 **0건**으로 감소했다. 이후 발견된
DELETE_OR_FOCUS의 핵심어 정의 문장 삭제 문제도 별도 보호 가드로 막고,
미사용 328편·5,812문장에 대한 out-of-sample 검증을 완료했다.

따라서 MVP는 “최종 사용자에게 무조건 안전한 자동 교정기”가 완성됐다는 뜻이
아니라, **TVM 학습용 transition과 corruption 데이터를 만들기 위한 신뢰 가능한
one-step 파이프라인이 확보됐다**는 의미에서 완료됐다.

## 3. 목표와 완료 기준

### 3.1 연구 목표

선행 FEAK의 일회성 진단·피드백을 반복 수정 제어 agent로 확장하기 전에,
학습 없이 다음 전제를 먼저 검증하는 것이 MVP 목표였다.

1. 글 한 편이 진단부터 후보 선택까지 실제로 한 바퀴 도는가?
2. 각 수정의 전후 상태와 선택 근거를 transition 단위로 기록할 수 있는가?
3. 채점기 점수만 오른 결함 수정을 규칙·의미 보존 신호로 걸러낼 수 있는가?
4. 생성된 transition을 다음 단계 학습 데이터로 사용할 만큼 감사할 수 있는가?

### 3.2 완료 기준 판정

| 완료 기준 | 결과 | 판정 |
|---|---|---|
| 실제 채점기·자질 계산 연결 | Kanana 8-rubric, FEAK 29-feature adapter 구현 | 완료 |
| action별 후보 생성 | 5개 수정 action을 매번 명시적으로 펼쳐 생성 | 완료 |
| 원문 보존 | 전체 rewrite 대신 span 기반 별도 수정본·patch 기록 | 완료 |
| 수정 전후 평가 | 연속 rubric 점수와 9개 transition field 계산 | 완료 |
| 결함 후보 차단 | 구조, 신규 수치·인용, 중복, 의미 보존 hard guard | 완료 |
| one-step 의사결정 | accept/reject_all/stop과 근거 기록 | 완료 |
| 실데이터 안정 실행 | 100/100 성공, 500/500 후보 평가 | 완료 |
| targeting 잔여 결함 보완 | 정의 문장 보호 가드 in/out-of-sample 검증 | 완료 |
| 테스트 | `tests/test_mvp_loop.py` 44개 통과 | 완료 |

## 4. 구현 결과

### 4.1 진단 계층

공통 인터페이스는 다음 한 줄로 고정했다.

```python
diagnoser.diagnose(text) -> Diagnosis
```

구현된 adapter는 다음과 같다.

| 경로 | 역할 |
|---|---|
| `feak_tc/diagnose/base.py` | `Diagnosis`와 `Diagnoser` protocol |
| `feak_tc/diagnose/kanana.py` | Kanana-8B+LoRA lazy-load adapter |
| `feak_tc/diagnose/feak_kobert.py` | 기존 FEAK/KoBERT wrapper |
| `feak_tc/diagnose/stub.py` | 외부 모델 없이 재현 가능한 테스트 diagnoser |

Kanana는 아래 8개 rubric을 1~9점 척도로 출력한다.

```text
task_1
content_1, content_2, content_3
organization_1, organization_2
expression_1, expression_2
```

FEAK 29개 언어학 자질은 채점기 입력에 넣지 않고 독립적으로 계산한다.
Kanana 모델은 재사용하고, 자질 추출은 격리 subprocess에서 먼저 수행해
불필요한 GPU 점유가 남지 않게 했다.

### 4.2 후보 생성과 targeting

수정 후보는 자유 생성 하나에 의존하지 않고 action별로 분리한다.

```text
ADD_DETAIL
DELETE_OR_FOCUS
COMPRESS
RESTRUCTURE
STYLE_REFINE
STOP
```

실제 수정 후보는 앞의 5개 action별로 생성하고, STOP은 더 나은 수정이 없을 때의
제어 결정으로 사용한다. LLM 모드에서는 스키마 검증된 JSON만 허용하며,
`auto` 모드에서 API 또는 파싱 오류가 나면 deterministic proposer로 대체한다.

`targeting.py`는 질문, 본문 주제어, 반복, action별 특성을 이용해 문장별 target
후보를 순위화한다. 최종 보완으로 DELETE_OR_FOCUS가 핵심어 정의 문장을 고르는
경우를 다음 세 층에서 막았다.

1. 핵심어 정의 문장 탐지
2. 휴리스틱 targeting 점수 감점
3. LLM이 해당 span을 반환해도 폐기 후 deterministic 후보로 대체

### 4.3 span 고정 reversible patch

초기 LLM patcher는 proposer가 지정한 target span과 다른 위치를 수정할 수 있었다.
이를 다음 구조로 변경했다.

- 코드는 target span을 문자 offset으로 고정한다.
- LLM은 지정 위치에 들어갈 국소 텍스트만 생성한다.
- DELETE_OR_FOCUS는 원칙적으로 기계적 삭제를 사용한다.
- 삭제 후 다음 문장의 접속이 깨질 때만 제한적으로 수선한다.
- `before`, `after`, `operation`, `reason`을 `Patch`에 저장한다.
- 원문은 덮어쓰지 않고 수정본을 별도 생성한다.

이 변경으로 “수정 계획은 맞지만 엉뚱한 문장을 삭제하는 문제”가 구조적으로
불가능해졌다. 재실행의 삭제 후보 100개에서 위치 오류는 0건이었다.

### 4.4 validity guard

`feak_tc/mvp/validity.py`는 재채점 전에 저비용 구조 검사를 수행한다. 위반 후보는
추가 Kanana 호출 없이 바로 거부한다.

현재 검사 항목은 다음과 같다.

- 변화가 없는 patch
- 글 전체 붕괴 또는 span 과도 축소
- 완결되지 않은 문장
- 원문에 없던 숫자·연도·비율 삽입
- “조사에 따르면”, “통계에 따르면” 등 신규 근거 표지 삽입
- 수정으로 새로 생긴 근사 중복 문장

이 가드는 의미 품질을 대신 판단하지 않는다. 명백한 구조 파손과 환각 패턴만
차단하고, 의미 판단은 transition feature와 이후 TVM의 역할로 남긴다.

### 4.5 transition feature

후보마다 다음 9개 field를 기록한다.

| Feature | 의미 |
|---|---|
| `action_type` | 수행한 수정 종류 |
| `target_rubric` | 개선 목표 rubric |
| `target_gain` | 목표 rubric 연속 점수 변화 |
| `non_target_drop` | 비목표 rubric의 최대 하락 |
| `target_gap_reduction` | 관련 FEAK 자질이 elite band에 가까워진 정도 |
| `evidence_match` | 수정 위치와 진단 약점의 일치도 |
| `edit_ratio` | 전체 대비 변경 토큰 비율 |
| `goal_preservation` | 수정 전후 의미 보존도 |
| `emb_sim` | 수정 전후 임베딩 유사도 |

`target_gain`과 `non_target_drop`은 가능한 경우 정수 rubric이 아니라
`rf_corrected_score`를 사용한다. `goal_preservation`과 `emb_sim`은
BGE-M3 cosine similarity를 사용하며, 모델을 사용할 수 없는 환경에서는
token retention fallback과 fallback 사실을 로그에 기록한다.

### 4.6 휴리스틱 선택과 로그

MVP selector는 학습된 reward model이 아니다. hard constraint를 통과한 후보 중
명시적 가중합 점수가 가장 높은 후보를 채택한다.

최종 Stage A 설정의 주요 hard constraint는 다음과 같다.

```text
target_gain_min       = 0.3
non_target_drop_max   = 1.0
edit_ratio_max        = 0.5
goal_preservation_min = 0.95  # BGE-M3 기준
evidence_match_min    = 0.3
```

후보마다 patch, 수정본, 전후 진단, transition feature, 휴리스틱 점수,
거부 여부와 거부 사유를 JSON으로 직렬화한다. 배치 실행은 레코드별 성공·실패를
JSONL로 남겨 후속 threshold sweep과 질적 검토를 모델 재실행 없이 수행할 수 있다.

## 5. 신뢰성 개선 과정

### 5.1 채점기 노이즈 측정과 연속 점수 전환

Kanana의 정수 rubric은 `m=3/10/20` 모두 동일 텍스트 재채점에서 최대 1점 차이가
남았다. 정수 `+1`을 곧바로 실제 개선으로 해석하기 어려웠다.

RF 보정 연속 점수의 동일 텍스트 재채점 결과는 다음과 같다.

| Kanana m | max \|diff\| | mean \|diff\| |
|---:|---:|---:|
| 3 | 0.685 | 0.158 |
| 10 | **0.225** | **0.080** |
| 20 | 0.281 | 0.081 |

이에 transition 계산을 `rf_corrected_score` 우선으로 전환했다. 기존 20편·100후보
로그의 threshold만 재판정한 결과, `target_gain_min=0.3`은 m=10에서 관측한
재실행 노이즈 상한 0.225보다 조금 높은 보수적 시작점이었다.

이 값은 7/20편을 채택했지만, 질적 검토에서 점수만으로는 잘못된 patch와 근거 없는
세부 추가를 걸러내지 못한다는 사실도 확인했다. 따라서 threshold만 조이는 대신
patch 구조와 validity guard를 먼저 개선했다.

### 5.2 baseline 결함 분석

2026-07-16 baseline 100편은 실행 오류 없이 완료됐고 52편에서 수정이 채택됐다.
채택본을 검토한 결과 최소 15건에서 다음 결함이 확인됐다.

- 삭제 위치 오류: 12건 중 11건
- 원문에 없는 통계·수치 삽입: 3건
- 압축문 삽입 후 원문이 남아 생긴 중복: 1건

특히 환각 3건은 기존 의미 보존 점수 1.0으로 통과했다. 이 결과는 단일 유사도나
rubric 향상만으로 품질을 보증할 수 없고, span 고정과 별도 validity guard가
필요하다는 근거가 됐다.

### 5.3 의미 보존 모델 비교

질적 라벨이 확정된 good 47건과 bad 12건으로 의미 유사도 모델을 비교했다.

| 방법 | 최대 입력 | AUC | 최적 임계 정확도 |
|---|---:|---:|---:|
| token retention fallback | - | 0.762 | 0.814 |
| `jhgan/ko-sbert-sts` | 128 | 0.647 | 0.797 |
| **`BAAI/bge-m3`** | 8192 | **0.819** | **0.881** |
| `nlpai-lab/KURE-v1` | 8192 | 0.796 | 0.864 |

고전 한국어 SBERT는 입력 길이 128 제한 때문에 에세이 후반 수정에 취약했다.
BGE-M3는 모든 good 47건에서 0.9637 이상이었고 bad의 절반은 0.95 미만이었다.
이에 BGE-M3를 기본 모델로 채택하고 의미 보존 임계값을 0.95로 보정했다.

단, 통계 환각 3건은 BGE-M3에서도 높은 유사도를 보였다. 임베딩은 환각 검출기가
아니므로 신규 숫자·근거 표지 guard를 계속 유지해야 한다.

### 5.4 최종 100편 재실행

동일한 100편에 대해 baseline, span 고정+guard, BGE-M3 적용 결과를 비교했다.

| 구분 | baseline 07-16 | spanfix+guard 07-19 | BGE-M3 07-20 |
|---|---:|---:|---:|
| 처리 성공 | 100/100 | 100/100 | **100/100** |
| 채택 수 | 52 | 48 | **56** |
| 채택 중 자동 스캔 플래그 | 21 | 0 | **0** |
| 채택 의미 보존 최솟값 | 0.733 | 0.733 | **0.960** |
| 의미 유사도 | token fallback | token fallback | **BGE-M3** |

BGE-M3 run에서는 500/500 후보가 fallback 없이 평가됐다. 선택 action 분포는
다음과 같다.

| Action | 채택 수 |
|---|---:|
| ADD_DETAIL | 26 |
| COMPRESS | 12 |
| DELETE_OR_FOCUS | 8 |
| STYLE_REFINE | 6 |
| RESTRUCTURE | 4 |

validity guard는 후보 단계에서 숫자 6건, 신규 인용 1건, 중복 6건 등 총 13건을
차단했다. BGE-M3 적용 후 정당한 COMPRESS 채택은 7건에서 12건으로 증가했다.
표면 토큰 겹침이 낮다는 이유만으로 좋은 압축을 거부하던 문제가 줄어든 결과다.

### 5.5 핵심어 정의 문장 보호

최종 100편 run에서도 구조적으로 깨끗하지만 내용상 중요한 정의 문장을
DELETE_OR_FOCUS가 선택한 사례 1건이 발견됐다. 이를 막는 가드를 구현한 뒤
두 단계로 검증했다.

**In-sample 소급 검증**

- DOF 후보 100건 중 핵심어 정의 문장 6건 차단
- 기존 채택 DOF 8건 중 문제 사례 1건만 차단
- 질적으로 깨끗했던 나머지 7건은 유지

**Out-of-sample 검증**

- 가드 설계에 사용하지 않은 328편
- 164개 질문에 걸친 층화 표집
- 총 5,812문장 검사
- question 있음 조건에서 보호 문장 38건 전수 육안 검토
- 명백한 오탐 0건, 경계 사례 1~2건
- DOF top-1 변경은 11/328편

가드의 개입 범위는 작으면서 정의 문장에 대한 정밀도는 높았다. 다만 이 규칙은
TVM 학습 데이터 오염을 막기 위한 임시 비계다. 장기적으로는 삭제 span의 주제
중심성을 TVM이 학습하도록 하고, 하드 가드와 학습된 판단을 ablation해야 한다.

## 6. 검증 상태

### 6.1 자동 테스트

2026-07-23 현재 다음 명령을 다시 실행했다.

```bash
pytest -q tests/test_mvp_loop.py
```

결과:

```text
44 passed, 2 warnings
```

경고 2건은 SWIG 타입의 `DeprecationWarning`이며 테스트 실패나 MVP 동작 오류는
아니다.

### 6.2 실행 경로

단건 stub smoke:

```bash
python scripts/run_mvp.py \
  --text-file sample_essay.txt
```

실제 Kanana 단건:

```bash
python scripts/run_mvp.py \
  --diagnoser kanana \
  --device-id 3 \
  --question "인권의 뜻과 특징에 대해 서술하세요" \
  --keywords "인간(사람), 당연, 권리, 존중(침해)" \
  --text-file sample_essay.txt
```

Stage A 배치 실행기는 `scripts/run_mvp_batch.py`, 채점기 노이즈 측정기는
`scripts/measure_scorer_noise.py`, 정의 문장 가드 검증기는
`scripts/validate_definition_guard.py`다.

## 7. 남은 한계

1. **one-step 한정**

   여러 번 수정했을 때의 누적 오류, 수렴, 반복 종료 조건은 아직 검증하지 않았다.

2. **휴리스틱은 TVM이 아님**

   현재 selector는 명시적 baseline이다. 미묘한 의미 손실이나 action 간 장기 가치는
   학습하지 못한다.

3. **drift/rollback 미구현**

   원문 의도에서 점차 멀어지는 전역 drift와 이미 채택된 수정의 rollback은 다음
   단계다.

4. **threshold는 잠정값**

   `target_gain_min=0.3`과 `goal_preservation_min=0.95`는 현재 표본에 맞춘
   Stage A 기준이다. 특히 의미 모델 비교의 bad 표본은 12건으로 작다.

5. **자동 스캔 0건은 완전 무결점 보장이 아님**

   알려진 구조·환각·중복 패턴이 없었다는 뜻이다. 모든 의미 오류를 사람이 전수
   판정했다는 뜻은 아니다.

6. **임베딩은 환각을 검출하지 못함**

   의미 유사도가 높아도 원문에 없던 통계나 근거가 추가될 수 있어 validity guard가
   별도로 필요하다.

7. **정의 문장 가드는 임시 비계**

   영구적인 의미 규칙으로 확장하면 hand-crafted heuristic이 TVM 역할을 침범한다.

## 8. 최종 판정과 다음 단계

MVP는 다음 상태로 종료한다.

- one-step loop의 모든 구성요소가 연결됐다.
- 실제 Kanana와 FEAK feature를 사용한 배치 실행이 가능하다.
- 후보와 결정 근거가 transition 단위로 재현·감사 가능하다.
- baseline에서 확인된 주요 구조 결함을 원인별로 차단했다.
- 100편 배치와 별도 out-of-sample targeting 검증을 통과했다.
- MVP 테스트 44개가 통과했다.

따라서 다음 작업은 MVP 기능을 더 늘리는 것이 아니라 아래 순서로 진행한다.

1. FEAK-guided corruption 데이터 생성
2. corruption 선호쌍과 사람 선호의 정합성 검증
3. GBM baseline과 Kanana adapter B 기반 TVM 학습
4. 휴리스틱 selector와 TVM selector 비교
5. multi-step controller, drift 추적, rollback 통합
6. FEAK 1회·반복 FEAK·LLM self-refine·휴리스틱·TVM agent 비교 평가

## 9. 근거 문서

- [`PROJECT_CONTEXT.md`](../docs/PROJECT_CONTEXT.md): 연구 배경과 확정 설계
- [`IMPLEMENTATION_MVP.md`](../docs/IMPLEMENTATION_MVP.md): MVP 최초 구현 범위
- [`DIAGNOSER_INTEGRATION.md`](../../docs/DIAGNOSER_INTEGRATION.md): 실제 채점기 연결
- [`SCORER_NOISE_M_SWEEP.md`](SCORER_NOISE_M_SWEEP.md): 채점기 노이즈 측정
- [`SOFT_THRESHOLD_SWEEP_20.md`](SOFT_THRESHOLD_SWEEP_20.md): 연속 점수 threshold 검토
- [`EMBEDDING_MODEL_EVAL_2026-07-19.md`](EMBEDDING_MODEL_EVAL_2026-07-19.md):
  의미 보존 모델 비교
- [`STAGE_A_BGE_100_RESULTS_2026-07-20.md`](STAGE_A_BGE_100_RESULTS_2026-07-20.md):
  최종 100편 3-run 비교
- [`DOF_DEFINITION_GUARD_2026-07-22.md`](DOF_DEFINITION_GUARD_2026-07-22.md):
  핵심어 정의 문장 보호와 out-of-sample 검증
