# 로컬 전체 루프 실행 확인 — 2026-09-11

신규 학습 없이 기존 Kanana 진단기와 일반 로컬 Qwen으로 실제 글쓰기 루프를 실행했다.
실험용으로 작성한 소수 예제의 연결 점검이며, 품질 비교나 RV 정확도 측정은 아니다.
사용자가 요청한 범위에 따라 외부 LLM API와 RV 학습은 사용하지 않았다.

## 실행 환경

- 진단: 기존 `essay_scoring_llm` Kanana + 학습된 LoRA·보정 모델, GPU 1, `m=3`, `chunk_m=1`.
- Planner·Generator·RV·경로 가드: 캐시된 `Qwen/Qwen2.5-7B-Instruct`, GPU 2, NF4 4비트 추론.
- 기존 독립 FEAK 자질 29개: CPU subprocess, 기존 로컬 Bareun/UTagger 사용.
- 의미 유사도: 캐시된 BGE-M3, CPU. 진단과 생성 모델은 각각 한 번 로드 후 재사용.
- 우수 사례 corpus: 미지정. 사례 없이 계획했으며 실제 RAG 성능은 확인하지 않았다.

## 실행 중 확인하고 수정한 문제

기존 Kanana 로더가 `CUDA_VISIBLE_DEVICES`를 변경하여 같은 프로세스의 Qwen에
`invalid device ordinal` 오류를 일으켰다. 채점기 코드를 수정하지 않고 별도 persistent worker로
진단을 격리하여 해결했다. 초기 실패 로그도 `local_agent_smoke.json`에 보존했다.

인권 예제의 초기 실행에서는 Qwen이 원문에 없는 1인칭 목격담을 만들고 RV·가드도 통과시켰다.
이에 Planner·Generator·RV·가드 지시에 원문에 없는 경험을 사실로 추가하지 말 것을 명시했다.
같은 상태에서 실패한 action/span을 그대로 반복하는 계획도 생성 전에 거절하도록 보강했다.
이 조치는 확인된 실패 유형에 대한 개발 수정이며, 경험 날조나 전역 훼손을 항상 탐지한다는
검증 결과가 아니다. 이전 출력은 비교·실패 분석용으로 그대로 보존했다.

## 자동 검증

`python -m pytest -q`: **197 passed**. 기존 SWIG 관련 deprecation warning 2건이 있었다.

반복 채택·재진단 재사용, NO_OP 우세 시 거절, RV fail/partial/null 거절, 재계획,
이전 안전 checkpoint 복구, 가드 실패·예산 소진 시 복구, 중복 후보·실패 계획 반복 차단,
JSON schema 실패, 로컬 모델 누락, API callback 미호출, train-only 사례 필터를 확인했다.
rollback의 정확한 snapshot 복구는 통제된 테스트와 아래 최종 실제 실행에서 모두 확인했다.

## 실제 실행 기록

원문·후보·진단·판정 근거·checkpoint·로컬 모델 원응답은 아래 로컬 산출물에 보존한다.
이 파일들은 Git에 업로드하지 않는다.

- `experiments/results/local_agent_smoke_v2.json`: 휴대전화 규칙 글, 2단계.
  정상 종료했으며 후보의 품질 회귀·NO_OP 대비 낮은 점수로 원문을 유지했다.
- `experiments/results/local_agent_human_rights.json`: 보강 전 인권 글, 3단계.
  2회 채택, 경로 가드 통과, 최대 단계로 종료. 위 목격담 생성 실패를 포함하므로
  이 결과를 타당한 품질 개선 사례로 사용하지 않는다.
- `experiments/results/local_agent_final_smoke.json`: 보강 후 같은 인권 글, 최대 2단계.
  **2회 채택 후 1회 rollback, checkpoint 1로 복구, `max_steps`로 정상 종료**했다.
  1단계 후보 2개는 RV를 통과했고, 2단계에서는 후보 하나를 NO_OP 대비 낮은 점수로 거절했다.
  다른 후보는 채택 후 경로 가드가 preservation=fail로 판정하여 복구했다.
  Kanana 진단 5회(원문 + 후보 4개), 로컬 Qwen 호출 12회였으며 후보 4개의 재진단과 RV 근거를 저장했다.

최종 가드의 실패 사유는 추가된 예시가 원문과 일치하지 않는다는 것이었다. 허용된 설명 추가와
실제 보존 훼손을 구분하는 데 여전히 판단 오류가 있을 수 있으므로, 이 rollback을 사람이 확인한
유해 수정 탐지 성공으로 보고하지 않는다. 확인한 사실은 국소 채택 이후 별도 전역 판정으로
이전 글이 정확히 복구되고 예산 내에서 종료됐다는 것이다.

## 재현

```bash
python scripts/run_agent.py \
  --text '인권은 모든 사람이 태어날 때부터 가지는 권리이다. 사람들은 누구나 소중하다. 그러므로 다른 사람을 존중해야 한다. 하지만 주변에서는 친구의 생각이 다르다는 이유로 무시하는 일이 있다. 나는 이런 일이 줄어들어야 한다고 생각한다. 서로 존중하는 것은 중요하다. 존중이 중요하기 때문에 서로 존중하는 마음이 중요하다. 학교에서도 학생들이 인권을 존중하는 방법을 배울 필요가 있다.' \
  --question '인권의 의미와 학교에서 인권을 존중할 수 있는 방법을 설명하시오.' \
  --max-steps 2 --candidates 2 \
  --output experiments/results/local_agent_new_run.json
```

모델 샘플링과 채점 노이즈가 있으므로 같은 판단·문장을 보장하지 않는다. 특히 `m=3`의
소수 실행과 같은 모델의 생성·자체 검증 결과만으로 품질 향상이나 독립 검증 효과를 주장하지 않는다.
이후 비교 실험에서 `m=10` 조건, 별도 글과 사람 검수를 사용할 수 있다.
