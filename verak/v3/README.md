# VERAK v3 — Phase 1 / 1b

Phase 1과 1b의 데이터 분할, 장르 메타데이터, 점수 전용 Kanana, 노이즈 보정과
평균 채점 비교를 구현한다.
기존 `verak/src/`, 프롬프트와 테스트는 바꾸지 않는다. 이후 단계의 수정 에이전트,
corruption, reward, 학습과 최종 평가는 실행하지 않는다.

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
유지하고 보고한 뒤 멈춘다. 후속 reward와 CHECK 결정은 config에 기록하고 실제
reward·환경 구현은 각각 Phase 4·5에 맡긴다.
