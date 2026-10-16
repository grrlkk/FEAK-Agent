import hashlib
import json

from staged_feedback.events import save_json
from staged_feedback.readable import write_logs
from test_engine import execute


def test_new_log_has_feedback_diffs_judgments_and_does_not_mutate_source(tmp_path, observer):
    run, output = tmp_path / "run", tmp_path / "review"
    execute(run / "essays/synthetic", observer)
    save_json(run / "selection.json", {"essays": [{"essay_id": "synthetic", "genre": "정서",
        "human_original_total": 40, "stratum": "middle"}]})
    before = {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in run.rglob("*") if p.is_file()}
    combined = write_logs(run, output)
    body = (output / "synthetic.md").read_text()
    assert "[총괄 초기 판단]" in body and "[1턴]" in body and "[결과]" in body
    assert "대상 문장:" in body and "전:" in body and "→ 후:" in body
    assert "반영. 이유:" in body and "observational_flags" not in body
    assert "```json" not in body and '"items":' not in body
    assert all(hashlib.sha256(__import__('pathlib').Path(p).read_bytes()).hexdigest() == value for p, value in before.items())
    assert json.loads((output / "provenance.json").read_text())["source_unchanged"]


def test_legacy_log_keeps_rejections_tests_and_does_not_invent_review_reasons(tmp_path, observer):
    run, output = tmp_path / "run", tmp_path / "review"
    result, _, _, _ = execute(run / "essays/synthetic", observer)
    turn = result["turns"][0]
    turn["review"] = {"resolved_issue_ids": [], "new_issues": [], "dispositions": [],
                      "turn_budget": 1, "continue_work": False, "reason": "부분 수정 후 종료"}
    turn["operations"] = [{"operation": {"id": "o1", "quote": "책을 읽을수 있다.", "replacement": "읽었다."},
        "applied": False, "retry": False, "reason_code": "preserve", "reason": "기존 보호 문장에 닿았습니다."}]
    turn["tests"] = [{"issue_id": "i1", "status": "error", "reason": "인용 추적 오류", "checklist": "표기 확인"}]
    result["baseline_tests"] = [{"issue_id": "i1", "status": "fail", "reason": "띄어쓰기 미수정", "checklist": "표기 확인"}]
    result.pop("review_method")
    save_json(run / "essays/synthetic/result.json", result)
    write_logs(run, output)
    body = (output / "synthetic.md").read_text()
    assert "거부 o1" in body and "항목별 이유 필드는 당시 스키마에 없었습니다" in body
    assert "당시 코드·거시 테스트의 별도 판정" in body and "검사 오류" in body
