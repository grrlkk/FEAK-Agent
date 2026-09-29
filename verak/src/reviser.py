"""A single replacement, assembled without regenerating unallocated text."""

from collections import Counter

from .schemas import Goal, Replacement


def validate_scope(text, goal: Goal):
    start, end = goal.span_before
    if not (0 <= start < end <= len(text)):
        raise ValueError("Invalid source character span")
    return start, end


def replace_span(text, goal, replacement):
    start, end = validate_scope(text, goal)
    return text[:start] + replacement + text[end:]


def hard_checks(before, after, goal, anonymization_pattern=None, constraints=None):
    """Only exact constraints, never inferred linguistic requirements."""
    import re
    start, end = validate_scope(before, goal)
    prefix, suffix = before[:start], before[end:]
    reasons = []
    if len(after) < len(prefix) + len(suffix) or not after.startswith(prefix) or not after.endswith(suffix):
        reasons.append("outside_allocated_scope")
    if before == after:
        reasons.append("no_change")
    if not after.strip():
        reasons.append("empty_candidate")
    if anonymization_pattern:
        original = Counter(m.group(0) for m in re.finditer(anonymization_pattern, before))
        revised = Counter(m.group(0) for m in re.finditer(anonymization_pattern, after))
        if original != revised:
            reasons.append("anonymization_marker_changed")
    # Explicit machine-readable task constraints only. No LLM-derived rules.
    for item in constraints or []:
        kind, value = item["kind"], item["value"]
        if kind == "max_chars" and len(after) > value:
            reasons.append("max_chars")
        elif kind == "min_chars" and len(after) < value:
            reasons.append("min_chars")
        elif kind == "required_literal" and value not in after:
            reasons.append("required_literal")
        elif kind not in {"max_chars", "min_chars", "required_literal"}:
            raise ValueError("Unsupported explicit task constraint")
    return not reasons, reasons


def revise(question, text, goal, profile, *, llm, prompt, anonymization_pattern=None):
    start, end = validate_scope(text, goal)
    response = llm.request(Replacement, prompt, {
        "question": question, "draft": text, "goal": goal.model_dump(),
        "profile": profile.to_dict(), "allocated_text": text[start:end],
        "anonymization_pattern": anonymization_pattern,
    }, role="revise", retries=0)
    return replace_span(text, goal, response.replacement)
