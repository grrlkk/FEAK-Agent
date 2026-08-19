# CODEX 작업 요청 — corruption 데이터 rule v4 (대량 생성 전 마지막 정비)

> 작성: 2026-08-12 (Claude). 기준 커밋 `8c5b62f` (`corruption_data`).
> 선행 문서: `docs/TASK_데이터생성_단계.md`(STEP 정의·확정사항),
> `CORRUPTION_RULEV3_STEP1_3_RESULTS_2026-07-29.md`(v3 결과),
> `G2_TWO_LLM_BLIND_REVIEW_2026-08-12.md`(2-LLM 블라인드 평가).
>
> **이 문서는 `TASK_데이터생성_단계.md`의 STEP1~3을 대체하지 않는다.** 그 STEP들은
> 완료됐고, 이 문서는 완료 후 발견된 결함을 고치는 **STEP 3.5**에 해당한다.
> STEP4(대량 생성)·STEP5(사람 평가) 정의는 원 문서를 계속 따른다.

## 왜 지금 대량 생성으로 가면 안 되는가 (근거 데이터)

Claude가 블라인드 50쌍을 직접 평가(48/50, 96%)하고 채택 데이터 42개와
`feak_tc/corruption/operators.py`를 직접 확인해 얻은 결과다.

### 결함 1 — 템플릿 아티팩트로 데이터 50%가 오염됨 (최우선)

`operators.py:351`이 고정 문자열을 주입한다.

```python
repetition = f"{term}, {term}, {term}을 계속 중요하게 생각해야 한다."
```

채택 42개 중 **21개(50%)** 에 이 문자열이 그대로 들어 있다. 더 나쁜 것은
`INJECT_LEX_REPEAT`가 아닌 operator 샘플에도 **20개 중 6개**가 오염돼 있다는 점이다
(체인 이전 stage에서 주입된 것이 누적).

`INSERT_OFFTOPIC`의 rule 버전도 같은 문제다 — `operators.py:124-129`의 고정 문장 4개를
돌려쓴다.

영향: 텍스트를 읽는 TVM(cross-encoder 등)은 이 문자열 하나를 탐지하는 것만으로
데이터 절반을 맞힌다. 일반화 성능은 0이 된다. 현재 GBM sanity가 100%인 것은 9개
feature만 봐서 이 함정을 우회했기 때문이며, 모델을 키우는 순간 문제가 드러난다.

### 결함 2 — SHUFFLE_FLOW가 구조적으로 무력함

채택된 42개의 `target_drop` 분포:

| operator | n | 최소 | 중앙값 | 최대 |
|---|---:|---:|---:|---:|
| INJECT_LEX_REPEAT | 22 | 0.39 | 1.27 | 3.25 |
| DELETE_SPECIFICS | 12 | 0.32 | 0.55 | 1.20 |
| INSERT_OFFTOPIC | 5 | 0.45 | 0.72 | 0.91 |
| **SHUFFLE_FLOW** | **3** | 0.34 | 0.40 | **0.50** |

SHUFFLE_FLOW는 **가장 강하게 훼손된 샘플조차 drop이 0.50**이다. 채택 3개 전부가
노이즈 인접 구간에 있고, margin을 0.5로 올리면 **0개**가 된다.

블라인드 평가에서도 G2H-020(SHUFFLE_FLOW)은 3명의 평가자 중 key가 지목한 쪽을
고른 사람이 **0명**이었다. 문장 위치를 바꿨더니 오히려 글이 자연스러워진 사례다.

### 결함 3 — DELETE_SPECIFICS 라벨 역전

G2H-044는 3명 전원이 key와 반대를 선택했다. 삭제된 span이 심한 반복 구간과 겹쳐,
훼손본이 오히려 읽기 좋아진 사례다. 채점기는 content_2 하락 0.78을 줬지만 사람 판단은
정반대였다.

### 결함 4 — 클래스 불균형과 rule 생성 편중

채택 42개 중 INJECT_LEX_REPEAT가 22개(52%). 생성기별로는 rule/rule_fallback이 23개(55%).

| operator | rule | LLM |
|---|---:|---:|
| INJECT_LEX_REPEAT | 15 | 7 |
| DELETE_SPECIFICS | 9 | 3 |
| SHUFFLE_FLOW | 3 | 0 |
| INSERT_OFFTOPIC | 0 | 5 |

rule 생성이 곧 템플릿 생성이므로 결함 1과 같은 뿌리다.

### 결함 5 — margin 0.3에 라벨 노이즈가 몰려 있음

블라인드 평가에서 문제가 된 4쌍 중 3쌍의 `target_drop`이 0.32/0.34/0.40이었다.
현재 임계 0.3 바로 위 구간이다. drop이 큰 쌍은 3명 모두 100% 일치했다.

---

## 작업 1 — 템플릿 아티팩트 제거 (최우선)

1. `INJECT_LEX_REPEAT`의 rule 구현에서 고정 템플릿 문장을 **폐기**한다.
   대체 방식은 둘 중 하나를 택하고 근거를 기록할 것.
   - (a) 원문에 이미 등장하는 어휘·구를 인접 문장에 자연스럽게 재사용하도록 재작성
   - (b) 이 operator를 **LLM 전용**으로 전환 (rule 모드 제외).
     v3에서 LLM 생성 7개는 템플릿 문제가 없었으므로 (b)가 더 안전하다.
2. `INSERT_OFFTOPIC`의 `_RULE_OFFTOPIC` 고정 리스트도 동일하게 처리한다.
3. **자동 중복 검사기를 추가**한다 (재발 방지 장치, 이번 작업의 핵심 산출물).
   - 생성된 corruption 코퍼스 전체에서 **동일 10-gram이 전체 샘플의 2% 이상에 등장하면
     생성 실패로 처리**하고 리포트에 위반 n-gram 목록을 남긴다.
   - 이 검사는 `scripts/`의 생성 파이프라인 끝에서 자동 실행되어야 한다. 사람이
     기억해서 돌리는 방식은 안 된다.

## 작업 2 — SHUFFLE_FLOW 진단 후 존폐 결정 (작업 3보다 먼저)

**이 결과에 따라 작업 3의 설정이 갈리므로 먼저 수행한다.**

에세이 10편에 대해 문장을 **완전 무작위로 셔플**한 뒤 재채점하여 `organization_1`
하락 폭을 측정한다 (corruption operator를 쓰지 말고, 단순 전체 셔플로).

- 하락이 크면 → 현재 operator가 너무 약한 것. `edits_per_step`을 늘리고 문단 경계를
  넘는 이동, 담화 의존 문장 다중 이동을 허용하도록 강화한다.
- 하락이 작으면 → **채점기가 문장 순서에 둔감**한 것. 그렇다면 SHUFFLE_FLOW를
  메인 체인에서 제외하고, RESTRUCTURE action의 학습 신호 확보 방안은 별도 논의로
  넘긴다. **operator 제거는 사용자 승인 사항이므로 임의로 제거하지 말고 측정값과
  권고안을 보고할 것.**

참고: MVP 100편에서도 RESTRUCTURE는 한 번도 선택되지 않았다. 두 경로가 같은 곳을
가리키므로 이 측정은 taxonomy 자체에 영향을 줄 수 있다.

산출물: `feak_tc_docs/중간정리/SHUFFLE_SENSITIVITY_2026-08-12.md`

## 작업 3 — 채택 기준과 균형 조정

1. `configs/corruption.yaml`의 `measurement.target_drop_min`을 **0.3 → 0.5**로 올린다.
2. `DELETE_SPECIFICS`에 **반복 동반 제거 가드**를 추가한다.
   - 훼손 전후로 어휘 반복 지표(기존 FEAK feature 활용)를 비교해, **반복이 유의하게
     줄었으면 그 샘플을 폐기**한다. G2H-044 유형의 재발 방지다.
   - 폐기 사유를 audit 로그에 남길 것.
3. **operator별 생성 쿼터**를 도입한다. 채택률이 operator마다 크게 다르므로
   (LEX_REPEAT 84.6% vs SHUFFLE_FLOW 14.3%), 생성 단계에서 균형을 맞추지 않으면
   최종 데이터가 한 operator로 쏠린다. 목표는 **최종 채택 데이터에서 어느 operator도
   40%를 넘지 않는 것**.
4. 생성 모드에서 rule 비중을 낮추고 LLM 비중을 올린다 (결함 1과 같은 뿌리).

## 작업 4 — 30편 재생성 및 블라인드 재검증 (rule v4)

작업 1~3 적용 후, **기존과 동일한 30편**으로 체인을 재생성하고 재측정한다.
v2→v3에서 통했던 검증 사이클을 한 번 더 도는 것이며, 대량 생성 전 마지막 관문이다.

보고할 것:
- operator별 생성/채택 수와 채택률 (v3 대비 변화)
- 채택 데이터의 operator 분포 (40% 상한 충족 여부)
- 10-gram 중복 검사 통과 여부
- `target_drop` 분포 (operator별 최소/중앙/최대)

이어서 블라인드 50쌍을 새로 추출하고 재평가한다. 평가 방식은
`G2_TWO_LLM_BLIND_REVIEW_2026-08-12.md`와 동일하게 한다 (key를 보기 전에 판정을
파일로 먼저 고정할 것).

## 작업 5 — STEP4 대량 생성 (작업 4 통과 후에만)

`TASK_데이터생성_단계.md`의 STEP4를 수행하되, 한 가지를 수정한다.

**학습곡선을 GBM 한 가지로만 측정하지 말 것.** 9개 feature짜리 GBM은 수백 쌍에서
조기 포화하므로, 그 곡선만 보고 규모를 정하면 텍스트 기반 TVM으로 갈 때 데이터가
모자란다. 최소한 다음 두 클래스로 곡선을 측정한다.

1. feature-only GBM (기존, baseline으로 논문에 남길 것)
2. 텍스트 기반 pairwise 모델 (BGE-M3 또는 KLUE-RoBERTa cross-encoder)

포화 지점이 다르면 **더 늦게 포화하는 쪽**을 기준으로 목표 규모를 정한다.

---

## 기존 42쌍의 취급

**TVM 학습 데이터로 사용하지 않는다.** 50% 템플릿 오염을 안고 있으므로 파일럿·
검증 기록으로만 보존한다. TVM 학습은 v4 데이터부터 시작한다.

## 하지 않는 것

- operator↔rubric 매핑 변경 (확정 사항)
- SHUFFLE_FLOW의 임의 제거 (측정 후 사용자 승인 필요)
- action taxonomy 변경
- TVM 모델 구조 확정 (데이터 준비 후 별도 논의)
- Global Drift 평가기 착수 (이번 범위 밖 — TVM 데이터에만 집중)
- 작업 4를 건너뛴 대량 생성
- `essay_scoring_llm` 채점기 코드 수정
- 커밋 메시지에 attribution 트레일러 추가

## 완료 기준

- 10-gram 중복 검사가 파이프라인에 내장되고 v4 코퍼스가 통과
- SHUFFLE_FLOW 민감도 측정값과 권고안 보고
- v4 30편 재생성 결과 + operator 분포 40% 상한 충족
- 블라인드 50쌍 재평가 결과
- pytest 전체 통과
- 진행 기록을 `CODEX_TASK_PROGRESS_2026-08-12.md`로 남길 것
