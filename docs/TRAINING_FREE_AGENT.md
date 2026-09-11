# 학습 없는 로컬 글쓰기 루프

2026-09-11 사용자 지시로 신규 RV 학습을 보류하고 전체 루프를 먼저 실행한다.
진단기는 기존 Kanana + 학습된 LoRA·보정 모델과 독립 FEAK 자질 계산기를 그대로 사용한다.
Planner, Generator, 두 축 RV, Trajectory Guard는 하나의 일반 로컬 Kanana 모델을 공유한다.
기본 모델은 `kakaocorp/kanana-1.5-8b-instruct-2505`이며, 이 역할들에는 진단기 adapter를 적용하지 않는다.
이 실행 경로에는 학습, corruption 생성, 외부 LLM API 호출이 없다.

## 실행

기존 `essay_scoring_llm` 설치, Kanana adapter·보정 모델, FEAK 자질 계산 환경
(Bareun/UTagger 등), 캐시된 Kanana-1.5-8B-Instruct-2505가 필요하다.
의존성은 기존 `requirements-kanana.txt`를 재사용한다. 모델은 `local_files_only=True`로
읽으며 자동 다운로드하지 않는다. 설정에서 다른 로컬 Instruct 모델 경로를 지정할 수도 있다.

```bash
python scripts/run_agent.py \
  --text-file examples/local_agent_essay.txt \
  --question "학교에서 휴대전화 사용에 관한 자신의 주장과 이유를 쓰시오." \
  --kanana-device 1 --llm-device cuda:2 \
  --max-steps 3 --candidates 2 \
  --output experiments/results/local_agent.json
```

GPU 번호는 실행 환경에 맞게 지정한다. 기존 Kanana 로더가 `CUDA_VISIBLE_DEVICES`를
변경하므로 진단기는 별도 persistent process에서 재사용한다. 일반 Kanana와 채점기의 GPU 설정이
서로 영향을 주지 않도록 한 것이며 채점기·자질 계산 코드는 수정하지 않는다.
기본 설정에서 feature subprocess와 BGE-M3는 CPU를 사용한다. `FEAK_EMBEDDING_DEVICE`로
임베딩 장치를 변경할 수 있다. 기존 의미 유사도 함수가 임베딩을 사용할 수 없으면 token
retention으로 대체하며 사용된 방식은 후보 metadata에 기록한다.

`configs/agent_local.yaml`에서 모델, GPU, 생성 예산, RV 통과 조건, FEAK 회귀 한계를 정한다.
`m=3`은 실행 확인용이며 기존 측정 조건은 `--kanana-m 10`이다. 이 설정의 임계값은 잠정값이다.
한 LLM은 한 번만 로드되고 역할별 프롬프트로 호출된다. 따라서 서로 독립된 평가자라는 주장은 하지 않는다.

모델 없이 제어 연결만 점검하려면:

```bash
python scripts/run_agent.py --offline-smoke \
  --text-file examples/local_agent_essay.txt --question "학교의 휴대전화 사용 규칙" \
  --output experiments/results/local_agent_offline.json
```

이 모드의 진단·RV·가드는 명시적인 stub이며 품질 평가 결과가 아니다.

## 웹에서 실행 과정 보기

```bash
python scripts/run_agent_web.py --port 8765
```

브라우저에서 `http://localhost:8765`를 연다. 원격 서버를 VS Code로 사용한다면 Ports 탭에서
8765번 포트를 전달한 뒤 연다. 다른 SSH 환경에서는
`ssh -N -L 8765:127.0.0.1:8765 USER@SERVER`로 전달한다.
기본 바인딩은 `127.0.0.1`이고 인증 없는 개인용 도구이므로 공개 인터넷에 노출하지 않는다.

1. 과제와 글을 입력하거나 **예제 불러오기**를 누른다. `.txt` 업로드도 가능하다.
2. 단계·후보 수를 선택하고 **수정 과정 시작**을 누른다. 기본은 실제 Kanana 실행이다.
3. 현재 처리 단계, 원문 대비 수정 표시, 8개 rubric 점수, 후보별 두 축 RV와 판정 근거를 확인한다.
4. 타임라인에서 채택·거절·경로 가드·복구를 보고, 후보나 저장 지점을 눌러 전체 글을 비교한다.
5. **기록 저장**으로 전체 JSON을 내려받는다. **이전 실행 다시 보기**와 새로고침 복원이 지원된다.

**화면 데모 · 모델 없이**는 UI 확인용 stub이다. 화면에 모의 판정으로 표시하며 실제 모델 결과와
구분한다. 실제 실행은 첫 모델 로딩과 각 후보의 진단에 수 분이 걸릴 수 있다. UI는 토큰 스트리밍이
아니라 단계·후보 완료 이벤트를 약 1.3초 간격으로 읽는다. 페이지를 닫아도 서버가 살아 있으면 계속 실행된다.

동시에 한 글만 실행한다. 중단하면 이 서버가 시작한 작업과 자식 프로세스만 종료하고, 경로 가드를
통과한 마지막 글을 보관한다. 검증 전 채택 상태는 최종문으로 반환하지 않는다. 원본·후보·실패 이력은
`experiments/results/web_runs/<실행 ID>/`의 `request.json`, `result.json`, `result.events.jsonl`,
`process.log`에 남는다. 서버 재시작 시 기록을 다시 읽는다. 서버를 비정상 종료한 경우에는 이전 GPU
자식 프로세스가 남아 있는지 확인하고 새 실행을 시작한다.

기존 `experiments/results/local_agent*.json`도 기록으로 표시한다. 과거 Qwen 실행은 Qwen으로
표시하며 새 Kanana 실행으로 재라벨링하지 않는다. 설정은 `--config`, 저장 위치는 `--runs-dir`,
기존 기록 위치는 `--history-dir`로 지정한다. 웹 자체는 Python 표준 라이브러리만 사용하며
Node 빌드나 별도 웹 프레임워크 설치가 필요 없다.

## 모듈과 결정

1. Kanana로 현재 글을 진단하고 독립 자질을 계산한다. 같은 글의 진단은 세션 내 재사용한다.
2. 로컬 Planner가 `problem`, 정확한 `target_span`, `instruction`, `preserve`를 작성한다.
   한 단계의 여러 후보는 같은 수정 요구를 공유한다. 거절·복구 이력은 다음 계획에 전달된다.
3. 기존 span patcher가 후보를 생성한다. 원문과 수정문을 모두 보관하고 구조 검사를 한다.
4. 후보를 재진단하고 기존 9개 transition feature를 계산한다. RV에는 과제, 수정 전 글,
   수정 요구·보존 조건, 수정 후 글만 전달한다. FEAK 점수·gain·정답 복원문은 넣지 않는다.
5. RV는 `target_fulfillment`, `preservation`의 `pass/partial/fail`과 축별 근거를 반환한다.
   판단 불가는 `label: null`로 남긴다. 기본 controller는 두 축 모두 `pass`일 때만 채택 가능하다.
6. 기존 휴리스틱과 hard constraints로 후보를 비교한다. 동일 target의 점수 변화·편집량이 0인
   NO_OP보다 점수가 높고 RV를 통과한 후보 중 최고를 채택한다. RV pass만으로 채택하지 않는다.
7. 채택한 전체 글을 원문·최고 안전 checkpoint·이력과 비교해 로컬 경로 가드를 실행한다.
   원래 주장·조건 보존과 전체 흐름을 확인하며, FEAK 평균점수의 checkpoint 대비 누적 하락도 본다.
   가드 실패·불확실·호출 실패이면 이미 채택한 상태를 최고 안전 checkpoint로 복구한다.
8. 연속 실패, 복구 횟수, 최대 단계, LLM 호출 예산에 따라 종료한다. Planner가 더 고칠 근거가
   없다고 판단해도 종료한다. 이전 채택·복구 상태로 돌아가는 동일 후보는 거절한다.

`reject`는 경로 채택 전 후보 거절, `rollback`은 채택된 경로의 복구다.
best checkpoint는 가드를 통과한 상태 중 FEAK 연속 점수 평균이 가장 높은 상태다.
가드 호출 예산이 소진되면 검증하지 못한 수정문을 최종 결과로 내보내지 않는다.
입력이 context 예산을 넘으면 글을 조용히 잘라 평가하지 않고 오류로 기록한다.

## 출력

- `local_agent.json`: 원문·최종문, 전체 checkpoint와 진단, 이벤트, 설정, 로컬 LLM 원응답·호출 수.
- `local_agent.events.jsonl`: 진행 중 즉시 기록되는 계획·후보·채택·거절·재계획·가드·복구·종료 로그.

기존 파일을 덮어쓰지 않으므로 재실행은 새 출력 경로를 사용한다. 입력 파일은 수정하지 않는다.
`accepted`는 가드 검사 이전의 채택 횟수를 포함한다. 실제 남은 상태는 `final_checkpoint_id`와
`parent_id`를 따라 확인하며, 복구된 상태도 기록에 남는다. `status=error`는 CLI 종료 코드 1이다.
`stop_reason=llm_call_budget`는 설정된 예산으로 종료했다는 뜻이며 품질 수렴을 의미하지 않는다.

## 선택적인 우수 사례 검색

사례 없이도 루프를 실행할 수 있다. 실제 train corpus가 준비되어 있으면 `--exemplars`와
`--source-group`을 지정한다. JSONL 한 행의 형식:

```json
{"essay_id":"train_1","source_group":"source_family_1","split":"train","text":"우수 사례 글", "rubrics":{"content_2":8.0}}
```

점수 기준을 통과한 rubric별 사례를 가벼운 어휘 유사도로 검색한다. 현재 글과 같은 essay ID,
같은 원문·파생 그룹, 가까운 중복문은 제외한다. `train` 외 행은 오류다. 정확한 분할·파생 그룹
부여와 corpus 선별은 입력 준비 단계의 책임이며, 이 검색기가 오염되지 않은 분할을 자동 보장하지 않는다.
검수용 전문가 피드백이나 참고 정답은 이 runtime 입력 형식에 포함하지 않는다.

## 이후 교체 지점

`run_agent(..., diagnoser=..., roles=...)`로 연결한다. 로컬 모델은
`LocalJSONClient`, RV는 `roles.verify`, 계획은 `roles.plan`, 가드는 `roles.guard`다.
향후 RV 학습이나 모델 교체 시 기존 입력·두 축 출력 계약을 유지해 해당 모듈만 교체할 수 있다.
기존 4축 파일럿 schema를 2축으로 덮어쓰거나 과거 데이터를 재라벨링하지 않는다.

현재 목표는 실제 입력으로 전체 경로가 작동하는지 확인하는 것이다. 품질 개선, 학습 RV의 필요성,
한국어 글쓰기 에이전트의 최초성은 별도의 선행연구 검토·비교 실험·사람 평가가 필요하다.

실제 실행과 확인된 한계는 [2026-09-11 실행 기록](LOCAL_AGENT_SMOKE_2026-09-11.md)을 참고한다.
전체 Kanana 전환 및 웹 검증은 [Kanana 웹 실행 기록](KANANA_WEB_SMOKE_2026-09-11.md)에 정리했다.
