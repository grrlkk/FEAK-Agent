"""Score ordering and exact source-span editing, without semantic acceptance rules."""

from dataclasses import dataclass
import math
import re

from feak_tc.diagnose.constants import RUBRIC_KEYS


@dataclass(frozen=True)
class TextUnit:
    unit_id: str
    start: int
    end: int
    text: str

    def public(self):
        return {"unit_id": self.unit_id, "text": self.text}


def text_units(text):
    """Deterministic addressable spans, not a claim to linguistic sentence parsing.

    Whitespace stays in the source. Decimal points and punctuation without a
    following separator are not split. A plan may allocate several adjacent units.
    """
    units, start = [], 0
    boundaries = [(m.start(), m.end()) for m in re.finditer(r"(?<=[.!?。！？])\s+|\n+", text)]
    for end, following in [*boundaries, (len(text), len(text))]:
        left, right = start, end
        while left < right and text[left].isspace():
            left += 1
        while right > left and text[right - 1].isspace():
            right -= 1
        if left < right:
            units.append(TextUnit(f"U{len(units) + 1:04d}", left, right, text[left:right]))
        start = following
    if not units:
        raise ValueError("Cannot plan edits for a blank draft")
    return units


def score_priority(scores):
    if set(scores) != set(RUBRIC_KEYS):
        raise ValueError("Planner requires all eight native rubric scores")
    if any(isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v)
           for v in scores.values()):
        raise ValueError("Planner scores must be finite numbers")
    # Stable rubric order resolves exact ties; values are not rounded or rescaled.
    ordered = sorted(RUBRIC_KEYS, key=lambda key: scores[key])
    return [{"rubric": key, "score": scores[key], "priority": index + 1}
            for index, key in enumerate(ordered)]


def scope_bounds(text, scope):
    units = {unit.unit_id: unit for unit in text_units(text)}
    if scope.start_unit not in units or scope.end_unit not in units:
        raise ValueError("Edit scope must reference current source units")
    start, end = units[scope.start_unit].start, units[scope.end_unit].end
    if start >= end:
        raise ValueError("Edit scope must be in source order")
    return start, end


def validate_plan(response, text, priority):
    checked = [check.rubric for check in response.priority_checks]
    expected = [item["rubric"] for item in priority]
    if checked != expected:
        raise ValueError("Review rubrics in Kanana score order without skipping any")
    findings = [check.finding for check in response.priority_checks]
    if response.plan is None:
        if checked != expected or "actionable" in findings:
            raise ValueError("Stopping requires checking all eight rubrics without an actionable issue")
        expected_outcome = "needs_information" if "needs_information" in findings else "no_actionable_issue"
        if response.outcome != expected_outcome:
            raise ValueError("Distinguish missing information from no actionable issue")
        return
    if "actionable" not in findings:
        raise ValueError("An edit needs an actionable issue")
    plan = response.plan
    if checked[findings.index("actionable")] != plan.target_rubric:
        raise ValueError("Plan must target the first actionable rubric")
    start, end = scope_bounds(text, plan.edit_scope)
    if plan.target_span not in text[start:end]:
        raise ValueError("target_span must quote text inside the allocated edit scope")
    if any(quote not in text for quote in plan.evidence_spans):
        raise ValueError("Evidence must quote the current draft exactly")


def validate_source_review(review, text):
    if [item.rubric for item in review.assessments] != list(RUBRIC_KEYS):
        raise ValueError("Source review must assess each rubric once in canonical order")
    for item in review.assessments:
        if item.issue and any(quote not in text for quote in item.issue.evidence_spans):
            raise ValueError("Source review evidence must quote the current draft exactly")


def validate_selected_issue(plan, selected):
    if plan.target_rubric != selected.rubric:
        raise ValueError("Planning cannot replace the score-selected rubric")
    for field in ("problem", "evidence_spans", "reader_impact", "context_check"):
        if getattr(plan, field) != getattr(selected.issue, field):
            raise ValueError("Planning must retain the source diagnosis without inventing another issue")


def apply_patch(text, plan, patch):
    if (patch.start_unit, patch.end_unit) != (plan.edit_scope.start_unit, plan.edit_scope.end_unit):
        raise ValueError("Reviser cannot expand the Planner's edit scope")
    start, end = scope_bounds(text, plan.edit_scope)
    if patch.expected_text != text[start:end]:
        raise ValueError("Patch source must match the allocated span exactly")
    result = text[:start] + patch.replacement + text[end:]
    if not result.strip():
        raise ValueError("A revision cannot delete the entire essay")
    return result
