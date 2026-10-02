# VERAK 범위 제한 수정 루프

현재 기본 실행은 **Kanana + 바른 → Planner → Reviser → RV → 채택/유지 → 반복**이다.
기존 소스 모듈 안에 구현했으며 P1 비교 실험은 `--mode single`로 재현한다.

```bash
conda activate feak_agent
python scripts/check_env.py
python -m verak.src.run_single --limit 5 --max-steps 5 --output-dir verak/outputs/logs/scope_run1
less verak/outputs/logs/scope_run1/summary.md
```

기본 입력은 `data/data_jsonl/valid.jsonl`, 계획·수정·검증 모델은 `gpt-5-mini/low`다.
점수는 기존 Kanana FT가 생성하는 1–9 정수 8개다. 문항과 에세이를 전달하며 키워드는 제외한다.

## 현재 역할과 제약

- **Planner:** `goal`, `scope`, `target`, `action`, `preserve`와 원문 근거를 정한다.
  Kanana의 낮은 항목부터 실제로 고칠 문제를 검토한다. 점수가 낮아도 근거가 없으면 다른 항목을 본다.
  문제를 해결할 수 있는 가장 작은 범위를 지시하고 그 이유를 기록한다. 최소성 자체는 의미 판단이며 기계적 증명이 아니다.
  `morpheme`은 바른 표면 오프셋, `sentence`는 한 문장, `span`은 연속된 여러 문장,
  `paragraph`는 줄바꿈 기준 문단, `document`는 전체 글이다. 모델은 출처 ID를 선택하고 코드가 오프셋을 계산한다.
- **Reviser:** `ADD`는 지정 구간 앞/뒤에 새 내용만 삽입하고, `DELETE`는 구간만 삭제한다.
  `REWRITE`는 그 구간만 교체한다. 전체 글을 포함한 REWRITE는 기본 금지다.
  `REORDER`는 span/paragraph의 문장 또는 document의 문단(한 문단이면 문장)을 순열로 재배열한다.
  문장 내용·공백 구분자·범위 밖 글은 코드가 보존한다. 익명화 표지의 종류별 개수도 검사한다.
- **RV:** `goal`, `selectivity`, `preservation`, `korean_consistency`를 pass/fail/unknown으로 판정한다.
  전후 전체 글과 문항으로 문맥을 확인하되, 바른 정보는 실제 diff에 걸친 형태소와 해당 문장의 문체 후보만 제공한다.
  전체 프로필·다른 문장의 형태소·선행사 목록·점수·이전 판정은 제공하지 않는다. 전역 비교 호출은 별도로 추가하지 않는다.
- **Loop:** 기계 제약과 RV 네 항목이 모두 통과하면 채택한다. 실패/불확실이면 직전 채택 상태에서 다른 계획을 시도한다.
  후보는 RV 전에 바른으로 분석하고, 그 프로필을 채택 시 다음 상태에서 재사용한다.
  RV 후 Kanana로 후보를 채점한다. 거절 후보도 분석용 점수를 남기지만 현재 상태 점수로 사용하지 않는다.
  동일 텍스트의 점수는 실행 중 재사용한다. 점수 증감은 RV 결정을 바꾸지 않는다.

`plan=null`, 동일 상태의 반복 계획, 호출 예산, 최대 step에서 STOP한다. 이전 채택 상태로 돌아가는 후보도 거절한다.
기본 최대 5 step(계획 실패·거절 포함), 기본 글 전체 재작성 금지다. 설정은 `config.yaml`의 `loop`에 있다.
API·인용 형식 오류의 동일 요청 재시도는 RV에만 최대 2회이며, 실패하면 unknown으로 기록한다.
후속 분석/점수 오류가 나도 마지막 채택 글을 원문으로 되돌리지 않는다. 점수 실패는 누락과 오류로 명시한다.

## 단계별 기록

- `trajectory.jsonl`: 현재 글, 계획, 후보, 실제 diff, 변경 부분 바른 정보, RV, 채택 결정, 다음 상태.
  `kanana_before/candidate/after`에는 피드백까지, `scores_before/candidate/after`에는 8개 점수를 기록한다.
  **candidate는 거절된 후보도 포함하고 after는 실제 채택 상태다.** STOP/오류 단계의 미실행 항목은 null이다.
- `events.jsonl`: 후보 생성과 채택 결정을 즉시 저장하므로 후속 채점 도중 중단돼도 확인할 수 있다.
- `calls.jsonl`: sample/step, 실제 요청·원시 응답·사용량·재요청 기록.
- `results.jsonl`, `report.json`, `summary.md`: 최종 채택 글, 단계별 전후 비교, 종료 이유, 호출 집계.
- `manifest.json`: 입력·설정·프롬프트·소스 해시. 출력은 Git에 포함하지 않는다.

맞춤법 API와 아래 P1의 세 조건 비교는 loop에서 호출하지 않는다. 바른 형태소 분석과 네 기준 RV만 사용한다.
자동 테스트나 RV 통과율은 사람 평가·실제 품질 향상·held-out 성능의 증거가 아니다.

## 보관된 P1 비교 실험

한 글에서 목표 하나와 후보 하나를 만들고, **같은 수정 쌍**을 세 판단 입력 조건으로 비교한다.
반복 수정이나 실제 채택은 하지 않는다. 기존 `feak_tc/agent` 파일럿과 별도로 실행한다.
작업 범위는 `imple/VERAK_CLI_TASK_P1.md`, 자료 구조와 판단 요건은 `imple/VERAK_IMPL_SPEC_v2.md`를 따른다.
사용자가 변경한 설정은 [NOTES.md](NOTES.md)에 기록했다.

## 실행

저장소 루트에서 기존 환경을 사용한다.

```bash
conda activate feak_agent
python scripts/check_env.py
python -m verak.src.run_single --mode single --limit 5 --output-dir verak/outputs/logs/p1_run1
less verak/outputs/logs/p1_run1/summary.md
```

기본 입력은 `data/data_jsonl/valid.jsonl`이다. 처음 5편을 연결 검사하며 test를 평가하지 않는다.
출력 폴더는 매번 새로 지정한다. 기본 모델은 수정·판단 모두 `gpt-5-mini`, reasoning low다.
서로 다른 계열 조건은 현재 GPT API를 쓰라는 사용자 지시에 따른 예외다.
키는 기존 로컬 `.env`에서 읽으며 파일에 키 값을 넣거나 로그에 출력하지 않는다.

채점기만 확인하려면 다음과 같이 실행한다. 키워드 줄 없이 원시 정수 8개와 피드백 8개를 검사한다.

```bash
python -m verak.src.run_single --diagnose-only --limit 3 --output-dir verak/outputs/logs/t6_run1
```

이미 생성한 쌍의 판단만 다시 실행할 수도 있다. 이 모드는 채점·목표·수정·맞춤법 API를 호출하지 않는다.
`--pairs`에는 report.json에 기록된 실제 `pair_path`를 사용한다.

```bash
python -m verak.src.run_single --pairs verak/data/pairs/실제파일명.jsonl --limit 5 --output-dir verak/outputs/logs/judge_run1
```

## 구조

1. 바른이 원문 오프셋과 형태소·문체·연결 관계·선행사 **후보**를 기록한다. 자동 공백 보정은 끈다.
2. 기존 Kanana FT 모델의 생성문에서 정수 8개와 루브릭별 피드백을 읽는다. 이 경로는 RF 보정값을 쓰지 않는다.
3. 가장 낮은 점수의 루브릭 하나를 코드로 정하고, LLM이 원문 인용·의도·시작/끝 문장을 지정한다.
   코드는 중간 문장을 모두 포함해 연속 범위의 `target_sents`와 `span_before`를 계산한다.
   인용은 LLM이 선택한 출처 ID에서 가져온다. 피드백 문장을 원문으로 잘못 복사하는 일을 막는다.
4. 수정기는 범위의 replacement만 반환한다. 프로그램이 원문에 교체해 범위 밖 UTF-8 문자열을 보존한다.
5. 문장 정렬·형태소 비교·문자열 차이를 기록하고 맞춤법 전후 검사 결과를 결합한다.
6. 동일한 쌍에 아래 조건마다 국소 판단 1회, 전역 판단 1회를 수행한다. 같은 쌍의 A/B 순서는 동일하다.

| 조건 | 추가 정보 |
|---|---|
| criteria_only | 문항·전후 전체 글·목표·판단 요건 |
| surface_diff | 위 + 문자열 변경 위치·표현 |
| korean | 위 + 형태소·변경 유형·해석 후보·불확실성·맞춤법 fixed/introduced |

판단 프롬프트 본문·모델·호출 구조는 같다. 채점 점수·정답·수정기 해설·앞선 판정을 보내지 않는다.
후보 범위와 익명화 표지, 명시된 JSON 문항 제약만 기계 검사한다. 추정 문체나 관계 라벨로 기각하지 않는다.
문항의 자유 문장 전체를 기계 규칙으로 자동 번역하지 않는다.

`hard_ok`이며 goal_valid/goal_improved/selective/meaning이 모두 pass이고,
global이 better 또는 equal이면 조건별 accept=true다. 무변경·빈 후보는 기각한다.
최종 returned_text는 항상 원문이다. API·파싱 오류 복구 최대 2회는 후보 재수정과 구분한다.
목표와 수정 생성에는 재요청이 없다. 판단 실패는 unknown, 맞춤법 실패는 unavailable로 진행한다.

## 보관과 재현

- `manifest.json`: 입력 경로·설정·프롬프트·소스 해시·고정 쌍 경로.
- `events.jsonl`: 단계별 상태와 생성 직후 후보. 판단 전에 실패해도 후보를 확인할 수 있다.
- `calls.jsonl`: 실제 API 요청·원시 응답·사용량·시간·오류·재요청·조건.
- `results.jsonl`: 진단·목표·후보·Units·맞춤법·조건별 판단·채택 계산·원문 반환.
- `report.json`, `summary.md`: 실행 집계와 CLI에서 읽을 수 있는 전후 비교.
- `data/pairs/*.jsonl`: 판단 전 저장한 고정 쌍·목표·프로필·변경 정보·맞춤법 결과.

입력·쌍·출력·캐시는 Git에서 제외한다. 학습 자료 중복이나 사람 판정 일치도는 연결 검사가 검증하지 않는다.
형태 복원으로 형태소 표기와 원문의 글자가 다를 수 있고 문장 정렬은 모호할 수 있다. 해당 불확실성을
판단기에 전달하며 규칙을 의미 보존의 정답으로 사용하지 않는다.

## 테스트

```bash
python -m pytest -q
VERAK_LIVE_TAGS=1 python -m pytest -q -s verak/tests/test_tags.py
VERAK_LIVE_TAGS=1 python -m pytest -q verak/tests/test_analysis.py
VERAK_LIVE_SPELL=1 python -m pytest -q -s verak/tests/test_spellcheck_live.py
```

일반 테스트는 실제 서비스 검사를 skip한다. T1의 실제 태그 검사를 먼저 실행한 뒤 사전을 확장한다.
명세에서 제외한 반복 루프·재수정·누적 점검·rollback·RAG·기준선은 구현하지 않았다.
