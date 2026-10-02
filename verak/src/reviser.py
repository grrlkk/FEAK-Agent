"""A single replacement, assembled without regenerating unallocated text."""

from collections import Counter

from .schemas import Goal, Replacement, ReorderResponse, ScopePlan


def validate_scope(text, goal: Goal):
    start, end = goal.span_before
    insertion = isinstance(goal, ScopePlan) and goal.action == "ADD" and start == end
    if not (0 <= start <= end <= len(text)) or (start == end and not insertion):
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


def apply_scoped_edit(text, plan, edit, *, allow_document_rewrite=False):
    target = plan.target
    start, end = target.start, target.end
    if not 0 <= start < end <= len(text) or text[start:end] != target.text:
        raise ValueError("Stale target: plan is not anchored to the current text")
    if plan.action == "ADD":
        if target.insertion_at not in {start, end}:
            raise ValueError("ADD must use its planned insertion anchor")
        content = edit["replacement"]
        return text[:target.insertion_at] + content + text[target.insertion_at:]
    if plan.action == "DELETE":
        if edit.get("replacement", "") != "":
            raise ValueError("DELETE cannot add replacement text")
        return text[:start] + text[end:]
    if plan.action == "REWRITE":
        if not allow_document_rewrite and not (text[:start] + text[end:]).strip():
            raise ValueError("Whole-document REWRITE is disabled")
        return text[:start] + edit["replacement"] + text[end:]
    if plan.action != "REORDER":
        raise ValueError("Unknown edit action")
    pieces = target.pieces
    ids, order = [p.id for p in pieces], edit["order"]
    if len(ids) < 2 or len(set(ids)) != len(ids) or sorted(order) != sorted(ids):
        raise ValueError("REORDER must be a permutation of all source IDs exactly once")
    cursor, slots = start, []
    for piece in pieces:
        if not cursor <= piece.start < piece.end <= end or text[piece.start:piece.end] != piece.text:
            raise ValueError("Invalid REORDER source pieces")
        gap = text[cursor:piece.start]
        if gap.strip():
            raise ValueError("REORDER cannot omit source content between pieces")
        slots.append(gap)
        cursor = piece.end
    if text[cursor:end].strip():
        raise ValueError("REORDER cannot omit the end of the target")
    by_id = {p.id: p.text for p in pieces}
    replacement = "".join(gap + by_id[key] for gap, key in zip(slots, order)) + text[cursor:end]
    return text[:start] + replacement + text[end:]


def revise_scoped(question, text, plan, *, llm, prompt, allow_document_rewrite=False,
                  anonymization_pattern=None):
    # The model never returns a full edited document or selects another target/action.
    if plan.action == "DELETE":
        edit = {"replacement": ""}
    else:
        schema = ReorderResponse if plan.action == "REORDER" else Replacement
        response = llm.request(schema, prompt, {
            "question": question, "draft": text, "plan": plan.model_dump(),
            "anonymization_pattern": anonymization_pattern,
        }, role="revise", retries=0,
            validate=lambda result: apply_scoped_edit(text, plan, result.model_dump(),
                                        allow_document_rewrite=allow_document_rewrite))
        edit = response.model_dump()
    return apply_scoped_edit(text, plan, edit, allow_document_rewrite=allow_document_rewrite), edit
