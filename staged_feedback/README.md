# staged_feedback

총괄 초기 판단 → 턴별 피드백 → **대상 범위 수정** → **총괄 검토·복원**으로 이어지는 한국어 글쓰기 실험이다. 기존 Kanana N32·자질·KB 어댑터를 읽기 전용으로 사용한다.

거시 이슈의 `macro_type`은 `restructure` 또는 `student_input`이다. 학생 과제는 결과 맨 앞에 안내하고 나머지 수정을 계속한다. 미시 이슈의 `restructure_issue_ids`에 지정된 미해결 재배열 이슈가 있을 때만 그 문장의 수정을 미룬다. 관계없는 미시 수정은 진행한다. 선택 제안은 `optional_suggestions`로 받아 결과 맨 끝에 둔다.

학생 과제는 총괄 초기 판단·검토만 만든다. 피드백 에이전트는 `needs_student_input`, 수정 에이전트는 `cannot_apply`로 총괄에게 알린다. 후자는 해당 범위를 그대로 둔다. 문장·문단 ID를 포함한 현재 글을 피드백 에이전트에 제공한다. 미시는 대상 문장, 거시는 대상 문단을 선택하며, ID 종류가 맞지 않아도 인용이 유일하게 일치하면 해당 단위로 찾아 연결한다. 서로 겹치는 범위는 묶는다. 같은 이슈의 인접 문단은 함께 재배열할 수 있고 떨어진 범위 사이의 글은 포함하지 않는다. 수정 에이전트에는 범위 원문·앞뒤 한 문장·해당 피드백만 전달한다. 전체 글·전체 수정 이력은 주지 않는다. 범위 수정본을 원래 위치에 끼워 넣고, 범위 밖 바이트는 그대로 둔다. 범위 안의 분할·병합·삭제는 자동 거부하지 않는다.

총괄은 항목별 `applied / partial / not_applied`와 이유, `new_issues`, `reopened`, 다음 계획을 정한다. `remaining_turns`는 **현재 검토 이후 앞으로 실행할 횟수**이며 종료는 0이다. 초기 계획에서는 시작할 턴 수다. 코드가 한 턴을 소모한 뒤 총괄이 남은 횟수를 조절한다. 턴별 개수의 고정 상한은 없고 이슈마다 피드백 한 항목이다. `single_pass`는 현재 실행 가능한 이슈를 한 턴에 처리한다.

네 관찰은 자동 거부를 일으키지 않는다. 총괄이 불필요한 변경이라고 판단해 `restore`에 이전 문장 ID·대응하는 현재 문장 ID를 적으면 코드가 **이전 버전의 실제 문장**으로 되돌린다. 삭제 복원은 현재 문장 ID 목록을 비우고 삽입 기준 문장을 지정한다. 복원 지시·실행 결과·복원 후 글을 모두 기록한다. 위치를 해석할 수 없는 지시는 사유를 기록하며 후보를 임의로 되돌리지 않는다. 비식별 치환·복원은 유지한다.

네 역할 프롬프트는 2026-10-16 사용자가 제공한 원문이다. 한 호출 비교 조건은 `single_call_staged.txt`와 별도 호출 역할로 분리했으며 주 수정 프롬프트에 섞지 않는다.

[제거·유지한 모든 규칙과 도입 이력](RULES.md)에 실행·입출력 제한도 구분해 적었다. 최신 사용자 요청이 과거 코드 검증 방법보다 우선한다.

## 실행

프로젝트 루트에서 `PYTHONPATH=staged_feedback/src`를 사용하거나 이 하위 프로젝트를 설치한다. 기본 비용 한도는 **$3**, 모델은 기존 `gpt-6-luna`, 총괄 low·피드백/수정 none이다. 기존 최대 4턴은 유지하며 변경 이력에 명시한다.

```bash
# 별도로 승인된 새 실험: 같은 선정표, 새 비용 원장
PYTHONPATH=staged_feedback/src python -m staged_feedback prepare \
  --repeat-run staged_feedback/runs/student_gate6_completed_20261016 \
  --run-dir staged_feedback/runs/fulltext_feedback6_20261016
PYTHONPATH=staged_feedback/src python -m staged_feedback live \
  --run-dir staged_feedback/runs/fulltext_feedback6_20261016

# 과거 기록을 변경하지 않는 Markdown 로그 생성, API/GPU 호출 없음
PYTHONPATH=staged_feedback/src python -m staged_feedback log \
  --run-dir staged_feedback/runs/student_gate6_completed_20261016 \
  --log-dir staged_feedback/reviews/student_gate6_completed_20261016

PYTHONDONTWRITEBYTECODE=1 python -m pytest -q -p no:cacheprovider staged_feedback/tests
```

같은 실험에서 실패를 재시도할 때는 `--from-run`으로 기존 비용 원장을 공유한다. `--repeat-run`은 별도 비용 승인을 받은 새 실험에만 사용한다. 이전 원자료·소스 스냅샷은 보존한다. 결과·학생 원문·선정표·캐시·키는 커밋하지 않는다.

## 산출물

- `readable/WORK_LOG.md`와 편별 Markdown: 초기 판단, 매 턴 피드백·범위별 수정·전후 문장·관찰·총괄 검토·복원·남은 턴 수, 학생 과제, 호출 수, 선택 제안.
- `calls.jsonl`, `events.jsonl`, `initial_plan.json`, `revision_turn_N.json`, `reviewed_turn_N.json`, `result.json`: 입력·출력·검토 원자료. 총괄 검토가 실패해도 범위 수정 후보는 먼저 저장된다. reviewed_turn에는 복원 후 글도 저장한다.
- `anonymization.jsonl`: 역할별 치환·복원 대응표. 기존 native validator는 호출하지 않는다.
- `summary.json`, `comparison.json`, `report.html`, `budget.json`: 비용·호출·계획 변화·이전 실행 비교.

거부 0건은 품질 통과율이 아니다. 해결은 총괄 판단이고, 삭제 관찰은 표면 정렬이며 의미 손실을 확정하지 않는다. 원문 사전 테스트는 새 방식에서 미실시다. 과거 로그의 테스트 결과와 새 총괄 판단은 구분해서 읽는다.
