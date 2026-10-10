"""Score-driven planning and exact patch application; these are not quality scores."""

import json

import pytest

from feak_tc.agent.editing import (apply_patch, score_priority, text_units, validate_plan,
                                  validate_source_review)
from feak_tc.agent.roles import Roles
from feak_tc.agent.schemas import (EditScope, PatchResponse, PlanResponse, ReplacementPatch,
                                  RevisionPlan, RubricAssessment, RubricCheck, SourceIssue, SourceReview)
from feak_tc.diagnose.constants import RUBRIC_KEYS
from feak_tc.runtime.openai import ResponseError


TEXT = "  비용은 1.5배다.\r\n규칙은 중요하고 중요한 규칙을 정하는 것이 중요하다.  \n그래도 예외는 허용해야 한다.\n"
SCORES = {key: 8.2 for key in RUBRIC_KEYS}
SCORES.update(content_2=6.1, expression_1=6.2)


def plan():
    quoted = text_units(TEXT)[1].text
    return RevisionPlan(target_rubric="expression_1", target_span=quoted,
                        problem="중요하다는 의미가 반복된다", goal="규칙의 중요성을 한 번만 표현한다",
                        must_preserve=["규칙의 중요성", "예외를 허용한다는 입장"],
                        evidence_spans=[quoted], reader_impact="반복으로 핵심을 읽기 어렵다",
                        context_check="앞뒤 조건은 유지하고 이 문장의 반복만 고친다", operation="replace",
                        edit_scope=EditScope(start_unit="U0002", end_unit="U0002"))


def decision(outcome="edit"):
    priority = score_priority(SCORES)
    checks = [RubricCheck(rubric=row["rubric"], finding="no_actionable_issue", reason="원문에 필요한 설명이 있다")
              for row in priority]
    request = plan() if outcome == "edit" else None
    if request:
        checks[1].finding = "actionable"
        checks[1].reason = "중요하다는 의미의 반복이 있다"
    elif outcome == "needs_information":
        checks[0].finding = "needs_information"
        checks[0].reason = "실제 비용 자료가 없어 수치를 보충할 수 없다"
    return PlanResponse(outcome=outcome, priority_checks=checks, plan=request, reason="점수 순서와 원문을 확인했다")


def patch():
    return ReplacementPatch(start_unit="U0002", end_unit="U0002", expected_text=text_units(TEXT)[1].text,
                            replacement="중요한 규칙을 정해야 한다.")


def source_review(outcome="edit"):
    response = decision(outcome)
    by_rubric = {item.rubric: item for item in response.priority_checks}
    issue = SourceIssue(**{key: getattr(plan(), key) for key in SourceIssue.model_fields})
    return SourceReview(assessments=[RubricAssessment(**by_rubric[key].model_dump(),
        issue=issue if by_rubric[key].finding == "actionable" else None) for key in RUBRIC_KEYS])


def test_native_score_priority_is_not_rounded_or_clipped():
    scores = dict(SCORES, content_2=9.125, expression_1=6.20001, content_3=6.20002)
    priority = score_priority(scores)
    assert [r["rubric"] for r in priority[:2]] == ["expression_1", "content_3"]
    assert priority[-1]["score"] == 9.125
    assert scores["expression_1"] == 6.20001


@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), True, "6.1"])
def test_invalid_score_never_triggers_paid_api(invalid):
    calls = []
    with pytest.raises(ValueError):
        Roles(lambda **kw: calls.append(kw)).plan("문항", TEXT, dict(SCORES, content_2=invalid))
    assert not calls


def test_missing_rubric_never_gets_silently_invented():
    scores = dict(SCORES)
    del scores["content_2"]
    with pytest.raises(ValueError):
        score_priority(scores)


def test_source_review_is_blind_to_scores_then_kanana_selects_lowest_actionable_rubric():
    calls = []
    def client(**kwargs):
        calls.append(json.loads(kwargs["user"]))
        return (source_review() if kwargs["schema"] is SourceReview else plan()).model_dump()
    response = Roles(client).plan("규칙에 관한 의견을 쓰시오", TEXT, SCORES)
    assert response.plan.target_rubric == "expression_1"
    assert [c.rubric for c in response.priority_checks[:2]] == ["content_2", "expression_1"]
    assert len(response.priority_checks) == 8
    assert set(calls[0]) == {"writing_prompt", "current_draft", "rubric_definitions", "text_units"}
    assert calls[1]["rubric_priority"][0]["score"] == 6.1
    assert calls[1]["rubric_scores"] == SCORES
    assert calls[1]["selected_issue"]["rubric"] == "expression_1"
    assert calls[0]["text_units"][1]["text"] == plan().target_span
    assert response.source_review == source_review()


def test_same_source_findings_with_different_kanana_scores_change_selected_target():
    review = source_review()
    second = next(item for item in review.assessments if item.rubric == "content_2")
    second.finding, second.issue = "actionable", review.assessments[6].issue
    source_calls = []
    def client(**kwargs):
        payload = json.loads(kwargs["user"])
        if kwargs["schema"] is SourceReview:
            source_calls.append(kwargs)
            return review.model_dump()
        selected = payload["selected_issue"]
        return plan().model_copy(update={"target_rubric": selected["rubric"]}).model_dump()
    roles = Roles(client)
    assert roles.plan("문항", TEXT, SCORES).plan.target_rubric == "content_2"
    assert roles.plan("문항", TEXT, dict(SCORES, expression_1=5.9)).plan.target_rubric == "expression_1"
    assert source_calls[0] == source_calls[1]


@pytest.mark.parametrize("outcome", ["no_actionable_issue", "needs_information"])
def test_no_source_issue_skips_planning_call_even_if_score_is_low(outcome):
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        return source_review(outcome).model_dump()
    result = Roles(client).plan("문항", TEXT, dict(SCORES, content_2=1.1))
    assert result.outcome == outcome and result.plan is None
    assert len(calls) == 1


@pytest.mark.parametrize("failure", ["duplicate_rubric", "fabricated_evidence"])
def test_source_review_cannot_fabricate_quotes_or_duplicate_rubrics(failure):
    review = source_review()
    if failure == "duplicate_rubric":
        review.assessments[0].rubric = "content_1"
    else:
        review.assessments[6].issue.evidence_spans = ["원문 밖의 근거"]
    with pytest.raises(ValueError):
        validate_source_review(review, TEXT)


def test_planning_cannot_replace_the_frozen_issue_after_seeing_scores():
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        if kwargs["schema"] is SourceReview:
            return source_review().model_dump()
        return plan().model_copy(update={"problem": "점수를 올리려고 만든 새 문제"}).model_dump()
    with pytest.raises(ResponseError):
        Roles(client).plan("문항", TEXT, SCORES)
    assert len(calls) == 3 and calls[1] == calls[2]


@pytest.mark.parametrize("failure", ["skipped_lower", "ignored_actionable", "wrong_target", "fabricated_evidence", "outside_scope"])
def test_grounded_plan_contract_rejects_untraceable_plan(failure):
    response = decision()
    if failure == "skipped_lower": response.priority_checks.pop(0)
    if failure == "ignored_actionable": response.priority_checks[0].finding = "actionable"
    if failure == "wrong_target": response.plan.target_rubric = "task_1"
    if failure == "fabricated_evidence": response.plan.evidence_spans = ["원문에 없는 표현"]
    if failure == "outside_scope": response.plan.edit_scope.start_unit = response.plan.edit_scope.end_unit = "U0003"
    with pytest.raises(ValueError):
        validate_plan(response, TEXT, score_priority(SCORES))


@pytest.mark.parametrize("outcome", ["no_actionable_issue", "needs_information"])
def test_stop_requires_all_rubrics_checked_and_distinguishes_missing_information(outcome):
    response = decision(outcome)
    validate_plan(response, TEXT, score_priority(SCORES))
    response.priority_checks.pop()
    with pytest.raises(ValueError):
        validate_plan(response, TEXT, score_priority(SCORES))
    wrong = decision("needs_information")
    wrong.outcome = "no_actionable_issue"
    with pytest.raises(ValueError):
        validate_plan(wrong, TEXT, score_priority(SCORES))


def test_exact_patch_preserves_unallocated_unicode_whitespace_and_decimal():
    units = text_units(TEXT)
    assert len(units) == 3
    updated = apply_patch(TEXT, plan(), patch())
    assert updated == TEXT[:units[1].start] + patch().replacement + TEXT[units[1].end:]
    assert updated.startswith("  비용은 1.5배다.\r\n")
    assert updated.endswith("  \n그래도 예외는 허용해야 한다.\n")


def test_repeated_text_addresses_only_the_allocated_occurrence():
    text = "반복이다. 반복이다. 조건은 유지한다."
    request = plan()
    changed = ReplacementPatch(start_unit="U0002", end_unit="U0002", expected_text="반복이다.", replacement="다른 설명이다.")
    assert apply_patch(text, request, changed) == "반복이다. 다른 설명이다. 조건은 유지한다."


@pytest.mark.parametrize("failure", ["wider_scope", "wrong_source", "reversed", "unknown_unit"])
def test_invalid_patch_cannot_change_source(failure):
    request, edit = plan(), patch()
    if failure == "wider_scope": edit.end_unit = "U0003"
    if failure == "wrong_source": edit.expected_text = "없는 원문"
    if failure == "reversed":
        request.edit_scope.start_unit = edit.start_unit = "U0003"
    if failure == "unknown_unit":
        request.edit_scope.end_unit = edit.end_unit = "U9999"
    with pytest.raises(ValueError):
        apply_patch(TEXT, request, edit)


def test_local_deletion_is_allowed_but_deleting_entire_essay_is_not():
    edit = patch(); edit.replacement = ""
    assert apply_patch(TEXT, plan(), edit) == "  비용은 1.5배다.\r\n  \n그래도 예외는 허용해야 한다.\n"
    request = plan()
    request.edit_scope = EditScope(start_unit="U0001", end_unit="U0003")
    units = text_units(TEXT)
    edit = ReplacementPatch(**request.edit_scope.model_dump(), expected_text=TEXT[units[0].start:units[-1].end], replacement="")
    with pytest.raises(ValueError):
        apply_patch(TEXT, request, edit)


def test_reviser_assembles_patch_and_receives_full_context():
    calls = []
    def client(**kwargs):
        calls.append(json.loads(kwargs["user"]))
        assert kwargs["schema"] is PatchResponse
        return PatchResponse(outcome="revised", patch=patch(), summary_of_change="반복 표현을 줄임").model_dump()
    result = Roles(client).revise("문항", TEXT, plan())
    assert result.revised_text == apply_patch(TEXT, plan(), patch())
    assert result.patch == patch()
    assert calls[0]["current_draft"] == TEXT
    assert calls[0]["allocated_scope"]["expected_text"] == patch().expected_text


def test_invalid_patch_api_retry_is_bounded_and_cannot_escape_scope():
    edit = patch(); edit.end_unit = "U0003"
    calls = []
    def client(**kwargs):
        calls.append(kwargs)
        return PatchResponse(outcome="revised", patch=edit, summary_of_change="범위 밖 변경 시도").model_dump()
    with pytest.raises(ResponseError):
        Roles(client).revise("문항", TEXT, plan())
    assert len(calls) == 2 and calls[0] == calls[1]


def test_reviser_may_decline_instead_of_inventing_content():
    def client(**kwargs):
        return PatchResponse(outcome="cannot_revise", patch=None,
                             summary_of_change="작성자의 의도가 없어 임의로 정할 수 없다").model_dump()
    result = Roles(client).revise("문항", TEXT, plan())
    assert result.outcome == "cannot_revise" and result.patch is None
    assert result.revised_text == TEXT
