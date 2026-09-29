"""Identical judge calls with only the supplied change information varied."""

import hashlib
import json

from feak_tc.runtime.openai import CallBudgetExceeded

from .change_info import surface_diff
from .llm import JSONFailure
from .reviser import hard_checks
from .schemas import CONDITIONS, GlobalJudgment, UnitJudgment

REQUIREMENTS = {
    "goal": "목표의 정당성과 개선",
    "selective": "모든 변경의 목표 기여 또는 수행상 필요성",
    "meaning": "허용 범위 밖 의미 보존과 허용 변경의 타당한 수행",
}


def candidate_side(pair_id, before, after, seed):
    # Stable under restart and condition iteration order, randomized across pairs.
    raw = json.dumps([seed, pair_id, before, after], ensure_ascii=False).encode()
    return "A" if hashlib.sha256(raw).digest()[0] % 2 == 0 else "B"


def acceptance(judgment):
    return (judgment["hard_ok"] and all(judgment[key] == "pass" for key in
            ("goal_valid", "goal_improved", "selective", "meaning"))
            and judgment["global"] in {"better", "equal"})


def condition_info(condition, before, after, units, spelling):
    if condition not in CONDITIONS:
        raise ValueError("Unknown judge information condition")
    result = {}
    if condition != "criteria_only":
        result["surface_diff"] = surface_diff(before, after)
    if condition == "korean":
        result["korean_changes"] = [unit.to_dict() for unit in units]
        # Include complete pair evidence too: any unmatched issue remains visible.
        result["spelling"] = {key: spelling[key] for key in ("status", "fixed", "introduced")}
    return result


def _ab(value, before_side, after_side):
    """Map location metadata together with the documents for the A/B call."""
    if isinstance(value, list):
        return [_ab(item, before_side, after_side) for item in value]
    if isinstance(value, dict):
        return {key.replace("before", before_side).replace("after", after_side):
                ({"before": before_side, "after": after_side}.get(item, item)
                 if key == "side" and isinstance(item, str) else _ab(item, before_side, after_side))
                for key, item in value.items()}
    return value


def _validate_quotes(issues, before, after, fields):
    for issue in issues:
        for field, text in zip(fields, (before, after)):
            quote = getattr(issue, field)
            if quote and quote not in text:
                raise ValueError("Judge evidence must quote the supplied document exactly")


def judge_pair(pair_id, question, before, after, goal, units, spelling, *, llm,
               unit_prompt, global_prompt, seed=42, conditions=CONDITIONS,
               anonymization_pattern=None, constraints=None):
    if tuple(conditions) != CONDITIONS:
        raise ValueError("P1 requires all three information conditions in the fixed order")
    hard_ok, hard_reasons = hard_checks(before, after, goal, anonymization_pattern, constraints)
    side = candidate_side(pair_id, before, after, seed)
    before_side = "B" if side == "A" else "A"
    documents = {before_side: before, side: after}
    results = {}
    for condition in conditions:
        info = condition_info(condition, before, after, units, spelling)
        local_payload = {"question": question, "before": before, "after": after,
                         "goal": goal.model_dump(), "requirements": REQUIREMENTS, **info}
        global_payload = {"question": question, "A": documents["A"], "B": documents["B"],
                          "goal": _ab(goal.model_dump(), before_side, side), "requirements": REQUIREMENTS,
                          **_ab(info, before_side, side)}
        errors = []
        try:
            local = llm.request(UnitJudgment, unit_prompt, local_payload, role="j_units", retries=2,
                condition=condition, validate=lambda response: _validate_quotes(response.issues, before, after,
                                                                              ("before_quote", "after_quote")))
            result = local.model_dump()
        except (JSONFailure, CallBudgetExceeded) as exc:
            result = {key: "unknown" for key in ("goal_valid", "goal_improved", "selective", "meaning")}
            result["issues"] = []
            errors.append({"stage": "j_units", "error_type": type(exc).__name__})
        try:
            global_result = llm.request(GlobalJudgment, global_prompt, global_payload, role="j_global", retries=2,
                condition=condition, validate=lambda response: _validate_quotes(response.issues,
                    documents["A"], documents["B"], ("a_quote", "b_quote")))
            preference = global_result.preference
            result["global"] = ("better" if preference == side else "worse") if preference in {"A", "B"} else preference
            for issue in global_result.issues:
                quotes = {"A": issue.a_quote, "B": issue.b_quote}
                result["issues"].append({"requirement": "global", "location": issue.location,
                    "before_quote": quotes[before_side], "after_quote": quotes[side], "reason": issue.reason})
            result["global_raw"] = global_result.model_dump()
        except (JSONFailure, CallBudgetExceeded) as exc:
            result["global"], result["global_raw"] = "unknown", None
            errors.append({"stage": "j_global", "error_type": type(exc).__name__})
        result.update(hard_ok=hard_ok, hard_reasons=hard_reasons, candidate_side=side, errors=errors)
        result["accept"] = acceptance(result)
        results[condition] = result
    return results
