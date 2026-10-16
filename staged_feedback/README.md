# staged_feedback

총괄 초기 판단 → 턴별 피드백 → **전체 글 수정** → **총괄 직접 검토**로 이어지는 한국어 글쓰기 실험이다. 기존 Kanana N32·자질·KB 어댑터를 읽기 전용으로 사용한다.

수정 에이전트는 `revised_text`와 학생 과제를 반환한다. 문장 수·패치 인용·연산 ID·이슈 ID 참조 강제, 편집 실행기, 코드 테스트, 거시 테스트 호출, 재시도·거부·되돌리기를 제거했다. 코드는 문장 변화와 네 관찰(학생 문장 삭제, 앞 턴 수정 문장 재변경, 비식별 표시 변화, 피드백 대상 밖 변경)만 저장해 총괄에게 준다. 비식별 치환·복원은 유지한다.

총괄은 각 피드백을 `applied / partial / not_applied`로 판단하고 이유를 적는다. 새 이슈·학생 과제·계획 변경을 직접 결정한다. 턴별 `feedback_count`는 총괄이 정하며 고정 상한은 없다. `single_pass`는 모든 수정 가능한 이슈를 한 턴에 배정한다. 학생의 새 내용·경험·의도가 필요한 과제는 수정 피드백에서 분리하며, 거시 학생 과제가 막으면 먼저 학생에게 제시하고 종료한다.

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

- `readable/WORK_LOG.md`와 편별 Markdown: 초기 판단, 매 턴 피드백·전후 문장·관찰·항목별 총괄 검토, 학생 과제, 호출 수.
- `calls.jsonl`, `events.jsonl`, `initial_plan.json`, `revision_turn_N.json`, `result.json`: 입력·출력·검토 원자료. 총괄 검토가 실패해도 직전 수정본은 먼저 저장된다.
- `anonymization.jsonl`: 역할별 치환·복원 대응표. 기존 native validator는 호출하지 않는다.
- `summary.json`, `comparison.json`, `report.html`, `budget.json`: 비용·호출·계획 변화·이전 실행 비교.

거부 0건은 품질 통과율이 아니다. 해결은 총괄 판단이고, 삭제 관찰은 표면 정렬이며 의미 손실을 확정하지 않는다. 원문 사전 테스트는 새 방식에서 미실시다. 과거 로그의 테스트 결과와 새 총괄 판단은 구분해서 읽는다.
