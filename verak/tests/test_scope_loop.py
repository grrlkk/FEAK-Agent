"""State transitions and source-edit contracts, independent of model preferences."""

import json
import re
from copy import deepcopy
from pathlib import Path

import pytest

from feak_tc.runtime.openai import CallBudgetExceeded
from verak.src.change_info import changed_korean_info, surface_diff
from verak.src.diagnoser import resolve_scope_plan, target_catalog
from verak.src.judge import judge_scoped
from verak.src.reviser import apply_scoped_edit, hard_checks
from verak.src.run_single import process_loop, render_loop_summary, load_config
from verak.src.schemas import (Profile, Sentence, Token, RUBRICS, ScopePlanSelection,
                               TargetSelection, ScopePlanResponse, Replacement, ScopeJudgment)


class AnalysisFixture:
    def __init__(self):
        self.calls = []

    def profile(self, text):
        self.calls.append(text)
        sentences = []
        cursor = 0
        for p, line in enumerate(text.splitlines(keepends=True), 1):
            for n, match in enumerate(re.finditer(r"[^.!?\r\n]+[.!?]?", line), 1):
                content = match.group().strip()
                if not content:
                    continue
                start = cursor + match.start() + len(match.group()) - len(match.group().lstrip())
                end = start + len(content)
                tokens = [Token(char, "JKS" if char == "가" else "NNG", i, i + 1)
                          for i, char in enumerate(text[start:end], start) if not char.isspace()]
                sentences.append(Sentence(f"{p}.{n}", p, start, end, content, tokens,
                                          style_candidates=["해라체"]))
            cursor += len(line)
        return Profile(sentences, ["해라체"], "fixture", "1")


def selection(scope="sentence", action="REWRITE", first="s:1.1", last=None,
              position="replace", goal="중복 조사를 고친다"):
    return ScopePlanSelection(rubric="어법적절성", goal=goal, scope=scope, action=action,
        target=TargetSelection(first_id=first, last_id=last or first, position=position),
        preserve_ids=["s:1.1"], evidence_ids=["draft:1.1"], minimal_scope_reason="문제 부분만 수정")


def resolve(text, draft=None, **kwargs):
    return resolve_scope_plan(draft or selection(), "발표에 대해 쓰시오", text,
                              AnalysisFixture().profile(text), **kwargs)


def verdict(label="pass"):
    return ScopeJudgment(goal="pass", selectivity="pass", preservation=label,
                         korean_consistency="pass", issues=[])


def test_add_inserts_only_and_preserves_every_original_character():
    text = "  앞이다.\r\n뒤다.  "
    plan = resolve(text, selection(action="ADD", position="after"))
    after = apply_scoped_edit(text, plan, {"replacement": " 연결 설명이다."})
    assert after == "  앞이다. 연결 설명이다.\r\n뒤다.  "
    assert hard_checks(text, after, plan)[0]
    assert not hard_checks(text, "다른 글이다.", plan)[0]


def test_morpheme_delete_and_rewrite_are_bounded_by_source_offsets():
    text = "학생들이가 발표했다. 다른 내용이다."
    plan = resolve(text, selection(scope="morpheme", action="DELETE", first="m:1.1:4"))
    after = apply_scoped_edit(text, plan, {"replacement": ""})
    assert after == "학생들이 발표했다. 다른 내용이다."
    with pytest.raises(ValueError, match="DELETE"):
        apply_scoped_edit(text, plan, {"replacement": "새 내용"})
    plan = resolve(text)
    after = apply_scoped_edit(text, plan, {"replacement": "학생들이 발표했다."})
    assert after == "학생들이 발표했다. 다른 내용이다."
    with pytest.raises(ValueError, match="Stale"):
        apply_scoped_edit(after, plan, {"replacement": "학생들이 발표했다."})


@pytest.mark.parametrize("scope,first,last", [
    ("document", "document", None), ("paragraph", "p:1", None),
    ("span", "s:1.1", "s:1.2"),
])
def test_full_rewrite_cannot_be_disguised_as_paragraph_or_span(scope, first, last):
    text = "  처음이다. 다음이다.  "
    draft = selection(scope=scope, first=first, last=last)
    with pytest.raises(ValueError, match="Whole-document"):
        resolve(text, draft)
    plan = resolve(text, draft, allow_document_rewrite=True)
    assert "새 글" in apply_scoped_edit(text, plan, {"replacement": "새 글"}, allow_document_rewrite=True)


@pytest.mark.parametrize("draft", [
    selection(first="s:9.9"), selection(scope="morpheme"),
    selection(scope="span", first="s:1.2", last="s:1.1"),
    selection(action="ADD"), selection(position="before"),
    selection(action="REORDER"),
])
def test_invalid_targets_scopes_and_actions_are_rejected(draft):
    with pytest.raises(ValueError):
        resolve("앞이다. 뒤다.", draft)


def test_reorder_moves_existing_sentences_without_generating_new_text():
    text = "첫째다.  둘째다.\r\n마지막이다."
    plan = resolve(text, selection(scope="paragraph", action="REORDER", first="p:1"))
    after = apply_scoped_edit(text, plan, {"order": ["s:1.2", "s:1.1"]})
    assert after == "둘째다.  첫째다.\r\n마지막이다."
    for order in (["s:1.1", "s:1.1"], ["s:1.2"], ["new", "s:1.1"]):
        with pytest.raises(ValueError):
            apply_scoped_edit(text, plan, {"order": order})


def test_document_reorder_preserves_paragraphs_and_crlf_gaps():
    text = "  첫 문단이다. 다음 문장이다.\r\n\r\n둘째 문단이다.  "
    plan = resolve(text, selection(scope="document", action="REORDER", first="document"))
    assert apply_scoped_edit(text, plan, {"order": ["p:3", "p:1"]}) == (
        "  둘째 문단이다.\r\n\r\n첫 문단이다. 다음 문장이다.  ")


def test_changed_morphology_excludes_unrelated_sentences_and_antecedents():
    before = "학생들이가 발표했다. 비밀후보는 유지한다."
    after = "학생들이 발표했다. 비밀후보는 유지한다."
    analyzer = AnalysisFixture()
    left, right = analyzer.profile(before), analyzer.profile(after)
    left.sentences[0].antecedent_candidates = [{"text": "SECRET_ANTECEDENT"}]
    info = changed_korean_info(left, right, surface_diff(before, after))
    assert [s["id"] for s in info["before"]] == ["1.1"]
    assert [t["form"] for t in info["before"][0]["tokens"]] == ["가"]
    assert "SECRET_ANTECEDENT" not in json.dumps(info)
    assert "dominant_style" not in info and "비밀" not in json.dumps(info, ensure_ascii=False)


class ScriptedLLM:
    def __init__(self, plans=(), patches=(), judgments=()):
        self.plans, self.patches, self.judgments = iter(plans), iter(patches), iter(judgments)
        self.calls = []

    def start_sample(self, sample_id):
        pass

    def close(self):
        pass

    def request(self, schema, prompt, payload, **kwargs):
        self.calls.append((kwargs["role"], deepcopy(payload)))
        role = kwargs["role"]
        if role == "planner":
            value = next(self.plans)
            if isinstance(value, Exception):
                raise value
            response = ScopePlanResponse(plan=value, reason="검증된 근거" if value else "더 고칠 문제 없음")
        elif role == "revise":
            response = Replacement(replacement=next(self.patches))
        elif role == "rv":
            value = next(self.judgments)
            if isinstance(value, Exception):
                raise value
            response = value
        else:
            raise AssertionError(role)
        if kwargs.get("validate"):
            kwargs["validate"](response)
        return response


class ScorerFixture:
    def __init__(self, fail_on=None):
        self.calls, self.fail_on = [], fail_on

    def diagnose(self, question, text):
        self.calls.append(text)
        if text == self.fail_on:
            raise RuntimeError("score unavailable")
        return {"scores": {r: (9 if "학생들은" in text else 3) for r in RUBRICS},
                "feedback": {r: "원문에서 확인하라" for r in RUBRICS}}


def run(llm, scorer=None, max_steps=5, text="학생들이가 발표했다. 다른 내용이다.", on_event=None):
    saved, analysis, scorer = [], AnalysisFixture(), scorer or ScorerFixture()
    result = process_loop({"id": "case", "question": "발표에 대해 쓰시오", "draft": text},
        analyzer=analysis, diagnoser=scorer, generator=llm, judge_llm=llm,
        config={"loop": {"max_steps": max_steps, "allow_document_rewrite": False},
                "anonymization_pattern": r"#@[^#\r\n]+#"},
        prompts={key: key for key in ("plan_loop", "revise_loop", "rv_loop")},
        on_step=saved.append, on_event=on_event)
    return result, saved, analysis, scorer


def test_rejection_keeps_state_then_different_plan_adopts_then_stops():
    old = "학생들이가 발표했다. 다른 내용이다."
    rejected = "학생들은 발표했다. 다른 내용이다."
    accepted = "학생들이 발표했다. 다른 내용이다."
    llm = ScriptedLLM(plans=[selection(),
        selection(scope="morpheme", action="DELETE", first="m:1.1:4"), None],
        patches=["학생들은 발표했다."], judgments=[verdict("fail"), verdict()])
    result, rows, analysis, scorer = run(llm)
    assert result["returned_text"] == accepted and result["stop_reason"] == "no_clear_revision"
    assert [r["decision"] for r in rows] == ["REJECT", "ACCEPT", "STOP"]
    assert [r["current_text"] for r in rows] == [old, old, accepted]
    assert rows[0]["scores_candidate"][RUBRICS[0]] == 9
    assert rows[0]["scores_after"][RUBRICS[0]] == 3  # High candidate scores do not override a rejection.
    assert scorer.calls == [old, rejected, accepted]
    assert analysis.calls == [old, rejected, accepted]
    planners = [p for role, p in llm.calls if role == "planner"]
    assert planners[1]["previous_attempts"][0]["rv"]["preservation"] == "fail"
    assert planners[2]["draft"] == accepted
    for role, payload in llm.calls:
        if role == "rv":
            assert payload["question"] == "발표에 대해 쓰시오"
            assert set(payload) == {"question", "before", "after", "plan", "diff", "korean_changes"}
            assert "profile" not in payload and "scores" not in payload and "previous_attempts" not in payload
    assert rows == [{"sample_id": "case", **r} for r in result["trajectory"]]


def test_scoring_happens_after_rv_and_adoption_is_logged_before_scoring_failure():
    candidate = "학생들이 발표했다. 다른 내용이다."
    llm = ScriptedLLM([selection()], ["학생들이 발표했다."], [verdict()])
    events = []
    result, rows, _, _ = run(llm, ScorerFixture(fail_on=candidate), on_event=events.append)
    assert result["returned_text"] == candidate
    assert result["stop_reason"] == "accepted_state_scoring_error" and result["status"] == "error"
    assert rows[0]["decision"] == "ACCEPT" and rows[0]["scores_after"] is None
    assert rows[0]["errors"][0]["stage"] == "kanana_candidate"
    assert next(e for e in events if e["stage"] == "decision")["current_text_after"] == candidate


@pytest.mark.parametrize("failure", [RuntimeError("offline"), CallBudgetExceeded("budget")])
def test_later_failure_never_resets_an_accepted_edit(failure):
    llm = ScriptedLLM([selection(), failure], ["학생들이 발표했다."], [verdict()])
    result, rows, _, _ = run(llm, max_steps=2)
    assert result["returned_text"] == "학생들이 발표했다. 다른 내용이다."
    assert result["accepted_steps"] == 1 and len(rows) == 2


def test_unknown_is_not_adopted_and_repeated_plan_stops():
    llm = ScriptedLLM([selection(), selection()], ["학생들이 발표했다."], [verdict("unknown")])
    result, rows, _, _ = run(llm)
    assert result["returned_text"] == result["original_text"]
    assert result["stop_reason"] == "repeated_plan" and len(rows) == 2
    assert len([r for r, _ in llm.calls if r == "revise"]) == 1


def test_changed_preserve_contract_is_a_new_plan_after_rejection():
    retry = selection()
    retry.preserve_ids = ["s:1.1", "s:1.2"]
    llm = ScriptedLLM([selection(), retry, None],
                      ["학생들은 발표했다.", "학생들이 발표했다."], [verdict("fail"), verdict()])
    result, rows, _, _ = run(llm)
    assert [row["decision"] for row in rows] == ["REJECT", "ACCEPT", "STOP"]
    assert result["returned_text"] == "학생들이 발표했다. 다른 내용이다."


def test_no_change_skips_rv_and_reuses_original_score():
    llm = ScriptedLLM([selection(), None], ["학생들이가 발표했다."])
    result, rows, _, scorer = run(llm)
    assert rows[0]["rv"]["hard_reasons"] == ["no_change"]
    assert result["accepted_steps"] == 0 and len(scorer.calls) == 1
    assert not any(role == "rv" for role, _ in llm.calls)


def test_markers_are_protected_even_inside_target_and_no_bad_candidate_reaches_rv():
    llm = ScriptedLLM([selection(), None], ["민수는 발표했다."])
    result, rows, _, _ = run(llm, text="#@이름#은 발표했다. 다른 내용이다.")
    assert rows[0]["rv"]["hard_reasons"] == ["anonymization_marker_changed"]
    assert result["returned_text"] == result["original_text"]
    assert not any(role == "rv" for role, _ in llm.calls)


def test_cycle_back_to_original_is_rejected_before_rv():
    llm = ScriptedLLM([selection(), selection(goal="이전 문장으로 바꾼다"), None],
                      ["학생들이 발표했다.", "학생들이가 발표했다."], [verdict()])
    result, rows, _, _ = run(llm)
    assert rows[1]["rv"]["hard_reasons"] == ["previously_accepted_state"]
    assert result["returned_text"] == "학생들이 발표했다. 다른 내용이다."


def test_new_four_criteria_and_summary_do_not_invent_p1_acceptance_results():
    llm = ScriptedLLM([selection(), None], ["학생들이 발표했다."], [verdict()])
    result, _, _, _ = run(llm)
    rendered = render_loop_summary([result], {"trajectory_path": "trajectory.jsonl"})
    assert "ACCEPT" in rendered and "최종 채택 글" in rendered and "후보 점수" in rendered
    assert "criteria_only" not in rendered
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg["mode"] == "loop" and cfg["loop"]["max_steps"] == 5


def test_rv_quote_validation_does_not_accept_fabricated_evidence():
    from verak.src.schemas import ScopeIssue
    before, after = "학생들이가 발표했다. 뒤다.", "학생들이 발표했다. 뒤다."
    bad = verdict()
    bad.issues = [ScopeIssue(requirement="preservation", location="문장", before_quote="없는 인용",
                            after_quote="", reason="검증되지 않은 근거")]
    llm = ScriptedLLM(judgments=[bad])
    with pytest.raises(ValueError, match="quote"):
        judge_scoped("문항", before, after, resolve(before), surface_diff(before, after), {},
                     llm=llm, prompt="rv")


def test_rv_api_failure_is_unknown_and_does_not_adopt():
    from verak.src.llm import JSONFailure
    llm = ScriptedLLM([selection(), None], ["학생들이 발표했다."], [JSONFailure("request failed")])
    result, rows, _, _ = run(llm)
    assert result["returned_text"] == result["original_text"]
    assert rows[0]["rv"]["errors"][0]["error_type"] == "JSONFailure"
    assert rows[0]["rv"]["korean_consistency"] == "unknown"


def test_cli_writes_readable_summary_and_self_contained_trajectory(tmp_path, monkeypatch):
    import verak.src.run_single as cli
    old, accepted = "학생들이가 발표했다. 다른 내용이다.", "학생들이 발표했다. 다른 내용이다."
    llm = ScriptedLLM([selection(), None], ["학생들이 발표했다."], [verdict()])
    monkeypatch.setattr(cli, "LLM", lambda *args: llm)
    monkeypatch.setattr(cli, "BareunBackend", lambda **kwargs: None)
    monkeypatch.setattr(cli, "Analyzer", lambda backend: AnalysisFixture())
    monkeypatch.setattr(cli, "LocalKanana", lambda config: None)
    monkeypatch.setattr(cli, "Diagnoser", lambda *args: ScorerFixture())
    source = tmp_path / "input.jsonl"
    source.write_text(json.dumps({"id": "case", "question": "발표에 대해 쓰시오", "draft": old}, ensure_ascii=False) + "\n")
    output = tmp_path / "run"
    assert cli.main(["--input", str(source), "--output-dir", str(output), "--limit", "1"]) == 0
    result = json.loads((output / "results.jsonl").read_text())
    assert result["returned_text"] == accepted
    rows = [json.loads(line) for line in (output / "trajectory.jsonl").read_text().splitlines()]
    assert [row["decision"] for row in rows] == ["ACCEPT", "STOP"]
    assert all(row["question"] == "발표에 대해 쓰시오" for row in rows)
    report = json.loads((output / "report.json").read_text())
    assert report["accepted_steps"] == 1 and report["step_errors"] == 0 and report["rv_errors"] == 0
    assert "accept_by_condition" not in report
    assert "최종 채택 글" in (output / "summary.md").read_text()
