# VERAK v3 — Phase 7 second teacher pilot

현재 활성 실행은 `two_stage`, CHECK 없음이며, 입력 코퍼스는
`config.paths.active_corrupt`의 `corrupt_scope_v2/`다. G_DELETE_SUPPORT가 포함된 글은
범위 결정으로 제외하고, L_CONJ_DROP은 기존 closed-set 접속어를 삭제한 뒤
같은 coarse class가 문두에 돌아오면 복구로 인정한다. 모든 새 후보는 글 단위
instance QC를 거친다. 이전 코퍼스·판정·trajectory는 보존한다.

각 단계 시작과 매 5번째 행동 뒤에 전체 Korean document profile과 글을 보여준다.
나머지는 표지 변화 알림과 바뀐 문단, 양쪽 이웃 한 문장만 갱신한다. Teacher와
정책 추론, SFT export는 같은 8,192토큰 문맥과 생성 예약 1,024토큰을 쓴다.
넘치면 system → KOREAN 인계 → 최신 전체 프로필 → 작업 일지 → 최근 턴 순으로
남기며, 문서 문자열을 잘라 맞추지 않는다. SFT turn 파일은 실제 teacher 입력과
동일하고 현재 assistant 응답만 loss를 갖는다. 누적 transcript는 감사용이다.

R_over는 참조되지 않은 원천 문장에 대해 형태소 거리와 순서 거리를 각각 0.5로
합친다. 문단 소속이 바뀐 문장은 순서 비용 1, 나머지는 해당 문장의 Kendall 역전
쌍 비율이며 문장별로 평균한다. 역할별 비용은 자신의 MOVE가 새로 만든 순서
차이에만 부과한다. GLOBAL 레코드가 없는 SFT 후보는 STOP≤2/R_over=0뿐 아니라
문장 이동·삽입·삭제를 시도하지 않았어야 한다(거절·UNDO도 해당 행동으로 센다).

```bash
python -m verak.v3.cli.teacher_pilot2 build --max-api-calls 0
python -m verak.v3.cli.teacher_pilot2 qc --max-api-calls 7000
python -m verak.v3.cli.teacher_pilot2 finalize --max-api-calls 0
python -m verak.v3.cli.verify_rewards --out verak/v3/outputs/phase7_pilot2/reward_audit
python -m verak.v3.cli.teacher_pilot2 prepare --max-api-calls 0
python -m verak.v3.cli.teacher_pilot2 run --max-api-calls 7000
python -m verak.v3.cli.teacher_pilot2 diagnostic --max-api-calls 0
python -m verak.v3.cli.teacher_pilot2 report --max-api-calls 0
```

QC와 pilot은 같은 SQLite 장부의 $20 한도를 공유한다. 완료된 요청은 재사용하고
SDK 자동 재시도는 꺼져 있다. CLI에는 bulk 생성·SFT·RFT 학습 경로가 없다.
실측 수치와 부족한 QC 표본은 로컬 `imple/reports/V3_PHASE_7_PILOT2.md`에 기록한다.
아래 절은 이전 단계의 구현·실행 계약을 보존한 것이며 최신 결정은 addendum을 따른다.

## 이전 Phase 3b 개요 (보존)

현재 활성 구조는 아래 **Phase 2c** 절의 `StructuralAnalyzer` / `annotate_structural`이다.
Phase 2/2b의 `KoreanStructure`, `annotate`, REF/TOPIC 규칙과 검증 프롬프트는 과거 결과
재현·디버깅 전용으로 보존한다. 새 데이터 준비는 config의 `structural_dependency` 모드로
DEP와 coarse 관계를 사용한다. Phase 3의 surface corruption과 데이터 QC를 추가했으며,
Phase 4의 의미 유사도 복원 판정·보상·학습은 아직 구현하지 않았다.

Phase 1과 1b의 데이터·채점·노이즈 보정에 Phase 2의 바른 기반 한국어 구조,
렌더링, EC 검증 도구를 추가한다. 단계별 실측 결과와 한계는 로컬 Phase 2 보고서에 기록한다.
기존 `verak/src/`, 프롬프트와 테스트는 바꾸지 않는다. 이후 단계의 수정 에이전트,
reward, 학습과 최종 평가는 실행하지 않는다.

## Phase 3b: 승인된 전체 후보 판정과 코퍼스 확정

파일럿 승인 후 진입점은 `verak.v3.cli.finalize_corruptions`다. 기존 100편의 판정과
비용을 재사용하고, 나머지 2,933편에 같은 QC 프롬프트와 `gpt-6.1-sol/high`를 적용한다.
파일럿 파일은 덮어쓰지 않는다. 누적 원장 `outputs/phase3b/full/cost_ledger.json`이
파일럿의 hash·100회·비용을 포함한다. 사용자 재개 결정 이후 비용 상한은 **확인된
usage 비용 + 미확인 timeout 예약액 합계 $50**이다. usage 없는 HTTP 5xx는 사용자
결정에 따라 $0으로 계산한다. 기존 원장은 `cost_ledger_before_retry_approval.json`에
보존하고 새 정책으로 재정산한다.

```bash
python -m verak.v3.cli.finalize_corruptions judge --max-api-calls 7797 --max-cost-usd 50 --workers 4
python -m verak.v3.cli.finalize_corruptions audit --max-api-calls 0
# 모든 후보의 유효한 판정이 완료된 뒤에만:
python -m verak.v3.cli.finalize_corruptions balance --max-api-calls 0
python -m verak.v3.cli.finalize_corruptions score --max-api-calls 0 --scorer-workers 1
python -m verak.v3.cli.finalize_corruptions verify --max-api-calls 0
```

키는 환경변수에서만 읽는다. 원시 응답·usage를 요청별로 즉시 저장하고, 재시작 시
완료된 판정을 복구한다. SDK 자동 재시도는 0이며, 애플리케이션이 5xx/timeout에만
최대 3회 재시도한다(10/40/120초 대기, transport timeout 600초). 요청마다 별도
attempt ID와 원장 항목을 만든다. 한 후보에서 유효한 판정을 받으면 다시 호출하지 않는다.
이미 끝난 1,445편과 기존 실패 155편의 첫 시도를 포함하면 가능한 최대 요청 수는
7,797회다. 이 호출 수 상한과 $50 비용 상한을 함께 적용한다.

동시 요청은 최대 4개다. 최근 완료 요청 50회에서 5xx가 10회를 넘으면 모든 새 요청을
10분 보류한 뒤 재개한다. 대기 상태를 파일로 보존하여 재시작으로 보류를 우회하지 않는다.
요청 전에는 진행 중 요청의 최대 비용도 예약하여 동시 요청으로 상한을 넘지 않게 한다.
5xx의 usage가 있으면 실제 사용량을 계산한다. Timeout 예약액은 재시도가 성공해도
원래 요청의 usage가 확인되기 전까지 유지한다. 로그의 `usage`는 확인 비용,
`timeout`은 미확인 timeout 예약액이다. 원장의 `cost_usd`는 그 합인 예산 검사값이다.

중단 시 `audit`가 `full/partial_after_retry/`에 완료된 판정, 통과한 부분집합,
미판정 실패/미시도 ID와 집계를 저장한다. 이전 중단의 `full/partial/`은 보존한다.
이 파일들은 최종 코퍼스가 아니다. 미판정 후보를 제외하여 균형 표집하거나 채점하지 않는다.

전체 판정 완료 후 `balance`는 모든 record가 통과한 글을 split별
`full/kept_agent_{train,dev}.jsonl`에 보존한다. WORD/SENTENCE/TEXT 국소 record 수로
비중을 계산하고, 각 28–38%가 되도록 글 단위로 비복원 표집한다. TEXT 비중이 1/3을
넘는 글을 우선 줄인다. 정수 최적화로 그 외 글의 제거를 최소화하고, 같은 조건에서
전체 잔존 수를 최대화한 뒤 같은 record 구성 안에서는 seed 41로 표집한다.
GLOBAL record는 세 국소 수준의 비중 계산에서 제외한다.

표집 결과를 채점 전에 hash로 고정한다. `score`는 기존 Phase 1 `KananaScorer`를
호출하여 k=1 expected score를 기록한다. GPU·모델·프롬프트·digit 계산은 설정과
기존 채점기를 그대로 사용한다. 모든 점수가 저장된 경우에만
`data/corrupt/agent_train.jsonl`, `agent_dev.jsonl`을 출력한다. `verify`는 원문/판정
불변성, split 분리, 학습 글 hash 배제, 국소 비중, 점수 provenance를 검사한다.

채점이 `ScoreParseError`로 끝나면 같은 입력·설정·파서로 단독 재현한 결과를
`full/score_parse_diagnostics.json`에 기록한다. 각 항목은 `episode_id`, `status`,
`error_type`, `fingerprint`, `first_lines: [{text: ...}]`를 포함한다. 모든 후보의
일괄 채점이 끝난 뒤 `exclude-unscorable --max-api-calls 0`으로 반복 확인된
형식 오류만 최종 표집에서 제외할 수 있다. 원래 manifest·표집과 전체 kept set은
보존하고, 새 선택은 `balanced_scorable_agent_*.jsonl`에 저장한다. 유효 점수가
있거나 다른 오류인 글을 제외하려는 경우, 또는 제외 후 28–38% 균형이 깨지는
경우는 거부한다. 점수의 크기를 선택 기준으로 사용하지 않는다. 이후 `score`를
다시 실행하면 이미 저장된 점수를 재사용하며, `verify`가 최종 파일을 검증한다.

실측 진행 상태와 미완료 항목은 로컬 `imple/reports/V3_PHASE_3b.md`에 기록한다.
Phase 4는 실행하지 않는다.

## Phase 3b: 개별 변경 필터링 파일럿 (보존)

현재 후보 생성은 `corrupt/instance_policy.py`를 사용한다. `G_VAGUE`,
`L_TRANSLATIONESE`를 제외한 10개 연산자를 유지하며, 낮은 QC 비율만으로 연산자를
끄지 않는다. 극성 표는 `수밖에 없다/않을 수 없다` 계열만 사용한다. 비첫 문장을
삭제하려면 바로 다음 문장이 `이처럼/이러한/이와 같이/따라서/그래서/이 때문에`로
시작해야 한다. 다음 문장은 문단 경계를 넘을 수 있으며 이 사실을 record에 저장한다.
나머지 연산자·바른 분석기·복원 계약은 Phase 3과 같다.

기존 1,011개 적격 원천과 저장된 source Q를 재사용해 원천당 후보 세 개를 만든다.
생성 시 WORD/SENTENCE/TEXT를 균등하게 배정한다. Phase 3 출력·판정·원장은 보존하고
새 출력은 `outputs/phase3b/`에 저장한다. 기존 Bareun cache는 읽기 전용으로 재사용한다.

seed 41로 두 split과 L1–L4, 모든 연산자를 포함하는 100편만 추출한다.
층은 `(split, curriculum, 후보에 등장한 가장 드문 연산자)`이며, 층 안에서 비복원
무작위 추출한다. 추출 확률을 저장해 전체 코퍼스의 잔존 규모·수준별 비중·비용을
가중 추정한다. 판정 결과를 보고 표본을 바꾸지 않는다.

`gpt-6.1-sol/high`에 기존 Phase 3 QC 프롬프트 그대로 에세이당 한 요청을 보낸다.
각 record의 `damage_real`과 `original_is_fix`가 모두 true여야 통과하고,
모든 record가 통과한 에세이만 유지한다. 전체 후보의 추가 판정에는 사용자 승인이
필요하다. 이 CLI는 고정 100편만 처리하며 재실행으로 이미 판정한 글을 다시 호출하지 않는다.

```bash
python -m verak.v3.cli.build_corruption_candidates --split both
python -m verak.v3.cli.verify_corruption_candidates
python -m verak.v3.cli.filter_corruption_pilot prepare --max-api-calls 0
python -m verak.v3.cli.filter_corruption_pilot judge --max-api-calls 100 --workers 4
python -m pytest -q tests verak/tests verak/v3/tests
```

파일럿 요청은 별도 단계의 단일 공유 원장 `phase3b/api_budget.json`에
호출 직전 기록한다. SDK 자동 재시도는 0이며 실패한 요청도 예산에 포함한다.
API 원시 출력, usage/cache 토큰, 표본·후보 hash와 프롬프트 hash를 저장한다.
`pilot_kept.jsonl`도 파일럿 결과일 뿐 최종 학습 코퍼스가 아니다. 모든 후보의
`q_corrupted`는 null로 유지한다. Phase 4는 시작하지 않는다.

## Phase 3: 이전 연산자 단위 QC (재현 전용)

`corrupt/`는 분석기 규칙을 바꾸지 않는다. WORD/SENTENCE/TEXT를 동등하게 표집하고,
DEP는 직전 문장 변경이라는 위치 힌트로만 기록한다. 후속 coupled 복원의 상대 가중치
계약은 CONJ=1, DEP=0.5이며 실제 보상 계산은 Phase 4 범위다.

- 원천은 `view_data.load_episode_examples`를 거쳐 valid의 각 split·장르 내 gold 상위
  사분위(경계 동점 포함), 500–2,500자, Kanana 8점 파싱 성공 조건으로 선정한다.
  gold는 두 human grader의 항목별 평균 합계이며 오프라인 선정에만 사용한다.
- 5개 GLOBAL과 7개 국소 연산자는 표면 문자열·위치만 편집한다. 국소 연산자는
  multi-unit과 익명화 표지 내부를 제외하고 새 표면을 바른으로 재분석한다.
  띄어쓰기는 WORD, 번역투 구문은 SENTENCE로 분류한다.
- 원문 공백과 stable ID를 보존하고 수정마다 역변환 기록을 남긴다. 복원 목표는
  항상 source 위치를 기준으로 한다. 이 역변환 검사는 Phase 4의 유사도 기반
  recovery나 reward 구현이 아니다.
- G_OFFTOPIC donor도 같은 agent split에서 다른 문항·같은 문체로 고른다.
  G_VAGUE는 gpt-5-mini/low로 미리 생성한 캐시만 사용한다. 전체 데이터 생성 시
  캐시가 없는 문장은 이 연산자의 대상이 아니며 추가 GPT 호출을 하지 않는다.
- `--max-api-calls`는 vague+QC의 단일 원장에 적용되고 100이 상한이다. GPT QC는
  gpt-6.1-sol/high로 dev 60편을 보고, 한 요청에 그 글의 모든 record를 묶어
  각 변경을 한 번씩 판단한다. 변경 전후 실제 문맥을 제공하며 예상 손상 라벨,
  gold·Kanana 점수는 제공하지 않는다. 어느 필드든 80% 미만이면 연산자를 끈다.
- 최종 builder는 QC gate를 읽어 기준 미달 연산자를 제외하고 새로 구성한다.
  private source/record는 supervision 용도이며 향후 에이전트 입력으로 보내지 않는다.
  최종 파일은 각 원천당 2편, L1–L4 계약, 3,000 compact 토큰 한도, Q 저장을 검사한다.

실행 순서(키는 프로세스 환경변수로 설정):

```bash
python -m verak.v3.cli.prepare_corruption_sources --split both
python -m verak.v3.cli.qc_corruptions vague --max-api-calls 100
python -m verak.v3.cli.build_corruptions --split agent_dev --per-essay 2 --seed 13 --pre-qc --defer-scoring --max-api-calls 100 --out verak/v3/outputs/phase3/pre_qc_agent_dev.jsonl
python -m verak.v3.cli.probe_corruptions
python -m verak.v3.cli.qc_corruptions judge --pool verak/v3/outputs/phase3/pre_qc_agent_dev.jsonl --max-api-calls 100
python -m verak.v3.cli.build_corruptions --split agent_train --per-essay 2 --seed 13 --max-api-calls 100 --out verak/v3/data/corrupt/agent_train.jsonl
python -m verak.v3.cli.build_corruptions --split agent_dev --per-essay 2 --seed 13 --max-api-calls 100 --out verak/v3/data/corrupt/agent_dev.jsonl
python -m verak.v3.cli.verify_corruptions
python -m pytest -q tests verak/tests verak/v3/tests
```

`--defer-scoring`은 Q가 없는 준비 파일임을 stats에 표시한다. `--score-only`로 실제
Kanana 채점을 완료해야 최종 검증을 통과한다. score cache는 Phase 1에서 독립된
Phase 3 사본으로 시작하며, 모델·점수 계산식은 동일하다.
QC 후 한 수준의 국소 연산자가 모두 탈락하면 최종 builder는 중단한다. 사용자 결정 없이
균등 구성 조건을 완화하지 않는다. 불균형을 명시적으로 허용받은 제한 데이터에만
`--allow-unbalanced`를 사용하며, 해당 파일은 균등한 최종 데이터로 표시하지 않는다.
로컬 결과는 `outputs/phase3/`, 최종 데이터는 `data/corrupt/`, 수치 보고서는
`imple/reports/V3_PHASE_3.md`에 저장한다. LLM QC 수치는 사람 정확도나 에이전트 성능이 아니다.

## 데이터 경계

- 에이전트 예제는 `valid.jsonl`에서만 읽는다. 반환 객체에는 문항·원문·메타데이터만 있고
  gold 점수·피드백·핵심 키워드는 없다.
- 준비 명령의 train/test 읽기는 사용자 승인된 해시 감사다. 본문을 저장하거나 모델에 보내지 않는다.
  이후 실행은 생성한 hash index로 누수를 검사하며 train/test 파일을 읽지 않는다.
- seed 13으로 문항 해시를 섞어 약 80/20으로 나눈다. valid는 삭제하지 않는다.
- `seen_by_scorer`는 문항이 train에서 관찰됐다는 뜻이며 에세이 본문 중복을 뜻하지 않는다.
- `eval_exclusions.json`의 1-based test 행은 Phase 9에서 반드시 제외해야 한다.
- 피드백 항목명은 장르 라벨을 만드는 오프라인 작업에만 사용한다. 런타임은 `genres.json`의
  라벨만 사용하고 피드백·항목명 감사 파일을 모델에 전달하지 않는다.
- 데이터, API 원문, 캐시와 실험 출력은 로컬 파일로 보관하고 Git에서 제외한다.

## 실행

저장소 루트에서 `feak_agent` 환경으로 실행한다. GPT·바른의 키는 실행 프로세스의
`OPENAI_API_KEY`, `BAREUN_API_KEY` 환경변수에 준비한다. v3 모듈은 `.env`를 읽지 않는다.

```bash
python -m verak.v3.cli.prepare_phase1 --max-api-calls 700
python -m pytest -q verak/v3/tests
python -m verak.v3.cli.verify_scorer --n 200 --repeat 20 --gpu 1
python -m verak.v3.cli.calibrate_scorer --split agent_dev --n 100 --max-api-calls 700 --workers 4
python -m pytest -q
```

기존 pytest 설정은 `tests`와 `verak/tests`만 수집한다. 따라서 새 테스트를 명시적으로
실행해야 한다. 실제 corpus metadata 검사는 준비 파일이 없는 checkout에서는 skip된다.
합성 데이터에 대한 누수·예산·채점·노이즈 계약 검사는 항상 실행된다.

## 채점 계약

`score/kanana.py`의 `KananaScorer.score(question, text)`는 eight expected scores,
argmax integers, mean Q, genre, cache 여부와 입력 길이를 반환한다.

- frozen Kanana + scorer LoRA를 지정 GPU 한 대에만 로드한다.
- greedy로 첫 줄만 생성하고 첫 개행에서 멈춘다. 점수 8개 이외의 첫 줄은 오류다.
- 첫 줄을 teacher forcing해 각 숫자 직전 위치의 raw logits를 읽는다.
- 1–9의 단일 토큰 ID 16–24만 정규화한다. `01`·`001`은 포함하지 않는다.
- 3,072 입력 토큰을 초과하면 오류다. 절단하지 않는다.
- 같은 `(question, text)`의 SHA-256을 캐시 키로 쓰며, 모델/설정 fingerprint가 다르면
  기존 캐시를 거부한다. 반복 검사는 `use_cache=False`로 실제 계산을 비교한다.
- 모델 로더와 기대점수 로직은 `essay_scoring_llm.soft_sc`에서 필요한 부분을 복사·변경했다.
  원본 패키지와 그 설정은 변경하지 않는다.

## API 예산과 노이즈

`outputs/phase1/api_budget.json`은 Phase 1 전체의 누적 요청 수다. 각 CLI의
`--max-api-calls`는 이 누적값에 적용된다. 재실행·재시도·병렬 worker도 같은 ledger를 사용한다.
700을 넘는 값은 거절한다. 요청 전에 lock을 잡아 예약하므로 실패·중단된 요청도 보수적으로 센다.
SDK 자동 재시도는 꺼져 있으며 모델은 gpt-5-mini/low다.

기존 OpenAI 어댑터의 Responses/JSON Schema/`store=False` 동작을 재사용한다.
[공식 Structured Outputs 문서](https://developers.openai.com/api/docs/guides/structured-outputs)와
[GPT-5-mini 문서](https://developers.openai.com/api/docs/models/gpt-5-mini)를 확인했다.

노이즈 표본은 agent_dev의 고정된 benchmark에서 장르별로 선택한다. 바른으로 문장 경계를
확인해 한 문장만 ±10% 길이로 바꾼다. 숫자·익명화·문체·문장 수를 검사하고 별도 GPT 요청으로
의미·문체·새 정보 유무를 확인한다. 같은 GPT 계열의 검증이므로 사람 정답으로 표현하지 않는다.
전체 글에서 공백 하나를 추가한 변형도 비교한다.

`noise_floor = 2 * std(부호 있는 paraphrase ΔQ)`이며 표준편차는 ddof=0이다.
`|ΔQ|`의 평균·표준편차와 부호 있는 ΔQ의 통계는 구분해 전체·장르별로 저장한다.
API 응답·변형·채점 결과는 중간 저장하며 같은 표본으로 재개한다.

## 주요 파일

| 파일 | 역할 |
|---|---|
| `data_policy.py` | 오프라인 장르 라벨, 문항 분할, hash 감사, validation-only 입력 |
| `score/loading.py`, `score/kanana.py` | 한 GPU의 frozen scorer와 기대점수·캐시 |
| `api.py` | 환경변수 전용 API 연결과 Phase 1 누적 호출 한도 |
| `calibration.py` | 의미 보존 변형의 검사와 노이즈 통계 |
| `cli/` | metadata 준비, 200/20 검증, 100편 보정 |
| `tests/` | split 누수·train 본문 배제·숫자 분포·결정성·예산·보정 계약 |

최종 결과는 로컬 `imple/reports/V3_PHASE_1.md`에 기록한다.

## Phase 1b: 장르별 분할과 평균 채점

Phase 1을 승인한 뒤에는 다음 명령으로 분할을 한 번 갱신한다.

```bash
python -m verak.v3.cli.resplit_by_genre
python -m pytest -q verak/v3/tests
python -m verak.v3.cli.calibrate_averaged --repeat 20
```

재분할은 원래 `splits.json`의 바이트를 `splits_v1.json`에 보존하고, seed 13으로
각 장르의 문항을 약 80/20으로 나눈다. 원천 valid는 유지하며 train/test를 읽지 않는다.
이후에는 기존 `prepare_phase1`이 새 분할을 덮어쓰지 못하도록 거절한다.

`KananaScorer.score_averaged(question, text, k)`는 원문과 k−1개의 공백 변형을 각각
기존 `score`로 채점한 Q의 평균을 반환한다. 변형은 익명화 표지 밖의 서로 다른 단어
경계에 공백 하나만 추가한다. 문단 경계는 유지한다. seed 13으로 모든 후보 위치를
섞은 뒤 앞에서 선택하므로 k=1,3,5는 중첩 집합이다. 경계가 부족하면 오류이며
같은 입력을 중복해서 채워 k를 늘리지 않는다. 개별 score의 계산 방식은 유지한다.

실험은 새 dev를 뽑지 않고 **기존 보정 100편과 저장된 변형**을 그대로 사용한다.
재분할 뒤 소속이 달라진 표본도 원래 실험의 쌍을 유지하고 소속 이동을 기록한다.
GPT·바른을 호출하거나 새 바꿔쓰기를 생성하지 않는다. 각 입력의 다섯 점수를
캐시 없이 한 번 계산하고, 앞 1/3/5개의 평균으로 noise를 비교한다.

시간은 같은 100편 원문의 처음 k개 실제 채점 호출 시간을 합산해 보고한다.
모델 로딩과 공백 후보 생성·집계의 작은 부가 시간은 제외한다. 20편은 다시 캐시
없이 계산하며 기대점수 벡터와 k별 Q가 일치해야 한다. 이전 Phase 1의 k=1 점수도
모든 원문·변형에서 일치하는지 확인한다.

실험과 재개 기록은 `outputs/phase1b/`, 집계는 `data/scorer_noise_averaged.json`에
저장한다. 원래 `scorer_noise.json`은 변경하지 않는다. k=5의 전체 paraphrase noise
floor가 k=1보다 25% 이상 줄고 결정성 검사가 통과해야 Phase 2를 진행한다.
통과하면 k=5 noise floor의 110% 이하인 가장 작은 k를 선택한다. 실패하면 k=1을
유지하고 보고한 뒤 멈춘다. 이 중단 이후 사용자가 Phase 1b를 승인하고 Phase 2 진행을
명시적으로 허용했다. 현재 설정은 `average_k=1`, `score`이며 평균 채점을 사용하지 않는다.
품질 보상과 CHECK는 장르별 noise floor를 쓰고, 미상 장르는 기존 전체 값으로 fallback한다.
실제 reward·환경 구현은 각각 Phase 4·5에 맡긴다.

## Phase 2: 한국어 구조와 EC 검증

`ko.KoreanStructure.from_config(config).analyze(text)`는 기존 바른 profile 위에
문장 ID, 문단, 문체, EC 후보, 접속부사, 주어·화제·선행사, 주절 극성·양태를 붙인다.
선행사는 같은 문단의 앞 두 문장을 확인하며, REF/TOPIC/REL 연결과 불확실성 표시를 만든다.
사람 정답으로 보장하는 구문·담화 분석기가 아니며, 모든 후보와 원본 offset을 보존한다.
`render`는 기본으로 모든 에세이에 compact 형식 하나만 제공한다. 대표 문체는 머리말에
한 번만 쓰고, 개별 문체 차이·생략 주어의 상태/대상·은/는 화제·관계 있는 EC·문두 접속어·
모든 REF/TOPIC/REL 연결·비기본 극성/양태·불확실성 `?`를 보존한다. compact는 절단하지 않는다.
`render(..., compact=False)`의 full 뷰는 디버깅 전용이며 표시용 줄임표가 원문을 바꾸지 않는다.

```bash
python -m verak.v3.cli.discover_ec_judge --max-api-calls 220
python -m verak.v3.cli.prepare_phase2
python -m verak.v3.cli.prepare_views --workers 4
VERAK_LIVE_TAGS=1 python -m pytest -q verak/v3/tests/test_ko_annotation.py
python -m pytest -q tests verak/tests verak/v3/tests
```

모델 조회는 성공 응답을 캐시하고 GPT-6.1 Sol, GPT-6 Sol 순으로 **실제 목록에 있는 ID**만
선택한다. 다른 모델로 대체하지 않는다. 바른 외 분석기와 새 패키지를 설치하지 않는다.
새 agent_dev 전체에서 EC 문장 모집단을 만들고 seed 23으로 100문장을 추출한다.
`data/ec_check.jsonl`은 EC를 강조한 형태소·태그와 사전 관계 후보를 담는다.
20편 compact 검토 자료(논증 최소 7편)는 `outputs/phase2/compact_review/`에 저장한다.
`outputs/phase2/renders/`의 full 뷰는 디버깅용이며 episode에 사용하지 않는다.

§6.3의 [ASK]는 사용자 결정으로 해결했다. `view_token_budget=3000`은 compact 뷰의
hard limit이다. `prepare_views`는 valid에서 파생된 agent_train/dev 전체를 검사하고
`data/view_eligibility.json`, `data/view_exclusions.json`을 작성한다. 원천 valid와
문항 split은 유지한다. 토큰 수 분포는 제외 전 모집단에서 측정한다.

Phase 3 이후 v3 데이터 구성에는 **`view_data.load_episode_examples(config, split)`**를 사용한다.
이 로더는 한도 초과 원문을 제외하고, 색인 누락·소스/분할/annotation/렌더러/토크나이저 변경 시
오류를 낸다. `data_policy.load_examples`는 기존 Phase 1 재현과 원천 감사용이며 8,000편을 유지한다.
새 변형도 `render_with_budget`의 `eligible`을 확인해야 하며 초과한 뷰를 절단해 통과시키면 안 된다.

EC 태깅 검증은 episode 뷰 예산과 독립적이다. 고정된 원래 100문장 표본을 유지하고
아래 두 단계를 진행한다.

```bash
python -m verak.v3.cli.judge_ec --stage pilot --max-api-calls 220 --workers 4
# 첫 10문장의 두 판정에서 산정한 예상 총비용이 $20 미만일 때만:
python -m verak.v3.cli.judge_ec --stage remaining --max-api-calls 220 --workers 4
```

두 판정은 같은 프롬프트를 별도 요청에 보내며 이전 응답은 입력하지 않는다.
모든 EC의 세 bool이 양쪽 모두 true일 때만 `llm_ok=true`다. `human_ok`는 코드가 채우지 않는다.
판정 모델은 `ec_judge.model`, reasoning은 high로 별도 설정한다. teacher는 계속 gpt-5-mini/low다.

`outputs/phase2/api_budget.json`의 220회 한도를 모델 조회·파일럿·나머지 판정·실패 요청이
공유한다. Phase 1 장부와 분리하며 재시작해도 누적한다. SDK 자동 재시도는 꺼져 있다.
파일럿/실제 비용은 요청한 $2/M input, $10/M output으로 계산하며 reasoning은 output에
이미 포함되어 있으므로 이중 가산하지 않는다. 캐시 할인은 반영하지 않은 계산값이다.

집계는 LLM-verified로 표기한다. 두 판정의 문장/토큰별 bool 일치율, 필드별 양측 true 비율,
불일치 문장 ID를 기록하며 자유로운 note 문장의 일치 여부는 세지 않는다.
실행 현황과 [ASK]는 로컬 `imple/reports/V3_PHASE_2.md`를 확인한다.

## Phase 2b: WORD / SENTENCE / TEXT 동등 검증

세 수준의 가중치는 `structure_policy.level_weights`에 각각 1로 기록한다.
WORD는 EC 관계·극성/양태·초점조사, SENTENCE는 REF/TOPIC과 문두 REL,
TEXT는 문장 종결과 글의 대표 문체다. structured output의 `field_levels`와
`cohesion_change_levels`에 수준을 명시하며, 렌더링에는 수준 이름을 추가하지 않는다.
여러 수준에 걸친 `uncertain`은 구성 필드마다 수준을 기록한다.

EC 다음 VX(허용 조사 경유 포함)는 AUX이며 관계 후보가 없다. 한 후보 EC는
UNAMBIGUOUS, 복수/미등재 후보는 AMBIGUOUS다. 문두 접속어에는 띄어쓰기 변형을 허용하되
접속어 다음 단어까지 붙인 모든 입력을 임의 분리하지 않는다. 어미 사전은 의문형·구어형과
EF 뒤 요/JX를 지원한다. 한 바른 문장에 EF가 2개 이상이면 `multi_unit=true`이며,
각 EF의 관측을 저장한다. 서로 다른 EF 문체가 있으면 mixed다. 내포 EF까지 세는 한계는
검토 보고서에 남기며, 이를 사람 정확도라고 해석하지 않는다.

형식명사·시간/담화 표현 목록과 관형절 규칙으로 주어 후보를 걸러낸다. 익명 표지는
표면 문자열이 같아도 발생 위치별 개체 ID를 부여한다. 동일인을 추정해 합치지 않는다.
REF/TOPIC의 HIGH는 단일 후보·같은 문단·앞 2문장·비복수 단위라는 규칙을 만족한 뜻이며,
의미상 정확성을 보장하지 않는다. LOW는 compact에서 `?`로 표시한다.

후속 사용 계약은 `ko/levels.py`에 둔다. 관계에는 UNAMBIGUOUS EC와 문두 접속어만,
의존 연결에는 HIGH만, corruption 대상에는 비복수 단위만 허용한다. 실제 Phase 3
연산자·피드백·복원 학습은 아직 구현하지 않는다. 이 조건을 만족해도 아래 LLM gate를
통과하지 못한 필드를 후속 연산자로 활성화하면 안 된다. 초점조사의 의미 분류는 이번
LLM gate의 검증 항목이 아니므로 검증된 필드로 취급하지 않는다.

```bash
python -m verak.v3.cli.prepare_phase2b --workers 4
python -m verak.v3.cli.judge_structure --stage pilot --max-api-calls 560 --workers 4
# 세 수준 각 5개 × 2회 파일럿의 예상 총비용이 $15 미만인 경우에만:
python -m verak.v3.cli.judge_structure --stage remaining --max-api-calls 560 --workers 4
```

준비 CLI는 Phase 2 판단 파일·장부·보고서의 SHA-256을 보존하고, 기존 바른 캐시로
compact 예산 색인을 갱신한다. 같은 20편은 `outputs/phase2b/compact_review/`에 다시 쓴다.
새 표본은 seed 29로 고정한 `data/structure_check_phase2b.jsonl`이다. 원래 Phase 2
100문장과 겹치지 않으며 세 검증 사이에도 중복 문장을 두지 않는다.

- WORD 80문장: 관계 후보 포함 40개와 비기본 극성/양태 포함 40개를 중복 없이 추출한다.
  두 필드가 모두 적용되는 문장은 두 필드를 함께 판정하며, 비적용 필드는 null이다.
- SENTENCE 120문장: 주어 생략 HIGH REF 60개, LOW REF 20개, 문두 접속어 40개.
- TEXT 80문장: 의문/구어 종결을 포함하는 20편에서 4개씩 뽑는다. 최소 20개는 해당 종결이다.

judge는 별도 `structure_judge` 설정의 gpt-6.1-sol/high다. 대상 문장과 같은 문단의
앞 2문장만 공통 문맥으로 제공한다. 두 요청에 동일한 prompt hash를 기록하고 이전 판단이나
HIGH/LOW 등급은 모델에 보여주지 않는다. `human_ok`는 비워 둔다.
`outputs/phase2b/api_budget.json`의 **560회는 파일럿을 포함한 두 실행의 공유 한도**다.
완전 표본에 정확히 560회가 필요하므로 자동 재시도는 없으며, 실패 시 예산을 늘리지 않는다.

`outputs/phase2b/results.json`은 양쪽 true 비율, bool 일치율, 불일치 ID, 토큰/비용,
필드·수준별 Phase 3 gate를 저장한다. gate는 WORD 관계/극성·양태 각각 85%,
SENTENCE HIGH 선행사 70% 및 문두 관계 85%, TEXT 문체 90%다. LOW는 진단용이다.
검증 항목의 표본 구성과 문맥 길이가 다르므로 세 수준의 단순 순위나 일반 정확도로
해석하지 않는다. 결정과 잔여 문제는 로컬 `imple/reports/V3_PHASE_2b.md`를 확인한다.

## Phase 2c: 구조적 의존성과 거친 관계

`ko.StructuralAnalyzer.from_config(config).analyze(text)` 또는 캐시용
`ko.annotate_structural(text, profile)`을 사용한다. `sentence_ids`를 전달하면 문장 이동
후에도 같은 ID를 보존한다. 이 인터페이스에는 선행사 정체성에 관한 활성 필드/연결이 없다.
`include_debug=True`와 `to_dict(include_debug=True)`를 모두 명시한 경우에만 과거 분석이
`debug.phase2b_antecedents_debug_only` 아래에 나온다. judge나 compact 뷰에는 넣지 않는다.

- WORD: CONDITION / CAUSE / ADVERSATIVE / PURPOSE. 대조와 양보를 합친다. AUX,
  고정 표현, 허가 구성은 제외하며 같은 coarse class 안의 교환을 허용하지 않는다.
  `-기 때문에`는 EC로 위장하지 않고 ETN+NNB+JKB construction으로 보존한다.
  기존 주절 polarity/modality 패턴은 그대로 사용한다.
- SENTENCE: NP+이/가/은/는/도, 익명화 표지, 지시·대명사를 보존한다. ETM 필터를 쓰지 않는다.
  생략이 불확실하면 `subject_omitted=false`, `omission_uncertain=true`다. 내포 명사구가
  있으면 주절 생략을 놓칠 수 있는 보수적 관측이며 완전한 구문 분석이라고 주장하지 않는다.
  `predecessor_id`는 문서 순서상 직전 문장이다. 문단 경계에서는 `cross_paragraph=true`,
  문서 첫 문장은 predecessor가 null이고 뷰에서는 START다. 생략 문장마다 DEP를 만든다.
  `dependency_changes(source, current)`는 stable ID를 비교해 직전 문장 변경 사실만 알린다.
  특정 인물이 선행사라거나 연결이 깨졌다고 단정하지 않는다.
- SENTENCE의 문두 접속어는 사용자 지정 17개 표현의 5개 coarse class만 검증용 REL에 쓴다.
  사전의 나머지 접속어는 `?관찰용`으로 표시하며 관계 복원에 쓰지 않는다.
- TEXT: 인용·내포 EF를 제외한 마지막 주절 EF의 문체만 사용한다. 모호한 -ㄹ까/-니는
  앞 문장 최종 문체가 해/한다일 때 이를 사용하고, 불명확하면 unknown이다.
  EF가 여러 개라는 `multi_unit`은 별도 플래그이며 문체 판정에 섞지 않는다.
  대표 문체는 알려진 문장 문체의 최빈값(동률은 unknown), off-style은 그와 다른 확정 문체다.

수준별 가중치는 모두 1이다. 후속 corruption의 대상은 계속 비복수 단위에 한정한다.
**Phase 4 복원 계약만 기록**했다: 원래 predecessor를 되돌리거나 명시 주어를 넣으면 DEP 복원이다.
아직 Phase 3 연산자, Phase 4 reward/recovery 실행기, Phase 5 환경을 구현하지 않는다.
초점조사·관찰용 접속어는 이번 검증 gate의 성공 항목으로 취급하지 않는다.

```bash
python -m verak.v3.cli.prepare_phase2c --workers 4
python -m verak.v3.cli.judge_phase2c --stage pilot --max-api-calls 470 --workers 4
# 체크당 5개 × 2회 = 40회 파일럿의 예상 총비용이 $10 미만일 때만:
python -m verak.v3.cli.judge_phase2c --stage remaining --max-api-calls 470 --workers 4
python -m pytest -q tests verak/tests verak/v3/tests
```

seed 31로 생략 80(각 bool 40), 닫힌 집합 접속어 40(문단 첫 문장 10), coarse EC 50,
문체 60(15편×4, 내포 의문/인용 포함 최소 15) 문장을 고정한다. 이전 EC/Phase 2b 표본과
새 네 검증 사이의 문장 중복을 금지한다. 문맥은 문단 경계를 포함해 직전 두 문장이다.
문체에는 대표 문체도 제공한다. 새 데이터는 `data/structure_check_phase2c.jsonl`,
검증·렌더링·비용 기록은 `outputs/phase2c/`에 로컬로 저장한다.

`phase2c_judge`만 gpt-6.1-sol/high이며 기존 teacher/EC/Phase 2b 모델 설정은 유지한다.
두 실행은 파일럿을 포함한 **470회 공유 장부**를 쓴다. 정상 완료에는 460회가 필요하며,
SDK 재시도는 없고 실패도 예약 횟수에 포함한다. 키는 프로세스 환경변수에서만 읽는다.
두 요청은 같은 prompt hash를 사용하고 상대 응답을 받지 않는다. 모든 `human_ok`는 null이다.
비용은 input $2/M, output $10/M으로 계산하고 reasoning을 output에 중복 가산하지 않는다.

gate는 WORD coarse 관계 85%, SENTENCE 생략 90% 및 접속 관계 85%, TEXT 문체 90%다.
**LLM-verified 양쪽 true 비율**이며 사람 정확도가 아니다. 생략 주어의 referent_type은
진단 분포로만 보고하고 gate에 넣지 않는다. 두 bool의 일치율과 이 진단 분류의 일치율은
따로 보고한다. 결과와 남은 한계는 로컬 `imple/reports/V3_PHASE_2c.md`에 기록한다.
