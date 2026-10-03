"""Identical judge calls with only the supplied change information varied."""

import hashlib
import json

from feak_tc.runtime.openai import CallBudgetExceeded

from .change_info import surface_diff, edit_verification_info
from .llm import JSONFailure
from .reviser import hard_checks, apply_scoped_edit
from .schemas import CONDITIONS, GlobalJudgment, UnitJudgment, EditJudgments

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


LOOP_REQUIREMENTS = ("goal", "selectivity", "preservation", "korean_consistency")
EDIT_REQUIREMENTS = ("necessity", "preservation", "groundedness", "meaning", "korean_consistency")


def _aggregate(labels):
    values = list(labels)
    return 'fail' if 'fail' in values else 'unknown' if not values or 'unknown' in values else 'pass'


def _validate_edit_judgments(response, edits):
    expected = {edit['id']: edit for edit in edits}
    ids = [item.edit_id for item in response.edits]
    if len(ids) != len(set(ids)) or set(ids) != set(expected):
        raise ValueError('RV must judge every edit exactly once, without invented IDs')
    for item in response.edits:
        edit = expected[item.edit_id]
        texts = [edit['context'][side]['text'] for side in ('before', 'after')]
        _validate_quotes(item.issues, *texts, ('before_quote', 'after_quote'))
        covered = {issue.requirement for issue in item.issues}
        for requirement in EDIT_REQUIREMENTS:
            if getattr(item, requirement) != 'pass' and requirement not in covered:
                raise ValueError('Every FAIL/UNKNOWN needs edit-specific evidence and a reason')
        if any(getattr(item, issue.requirement) == 'pass' for issue in item.issues):
            raise ValueError('Issue contradicts its PASS label')
        if not item.reason.strip() or any(not issue.reason.strip() for issue in item.issues):
            raise ValueError('Edit verdict needs a nonblank reason')


def judge_scoped(question, before, after, plan, diff, verification, *, llm, prompt,
                 hard_ok=True, hard_reasons=None, planned_edit=None, allow_document_rewrite=False):
    """Batch independent edit verdicts; aggregate them exclusively in code."""
    reasons = list(hard_reasons or [])
    _, source_reasons = hard_checks(before, after, plan)
    reasons.extend(source_reasons)
    if plan.action == 'DELETE' and after != before[:plan.target.start] + before[plan.target.end:]:
        reasons.append('delete_added_or_replaced_content')
    if planned_edit is not None:
        try:
            if apply_scoped_edit(before, plan, planned_edit,
                                 allow_document_rewrite=allow_document_rewrite) != after:
                reasons.append('candidate_does_not_match_planned_edit')
        except (KeyError, ValueError):
            reasons.append('invalid_action_edit')
    if 'edits' not in verification:
        verification = edit_verification_info(before, after, plan, None, None)
    if diff != surface_diff(before, after):
        reasons.append('diff_does_not_match_texts')
    reasons.extend(verification['hard_reasons'])
    reasons = list(dict.fromkeys(reasons))
    hard_ok = hard_ok and not reasons
    result = {key: "unknown" for key in LOOP_REQUIREMENTS}
    result.update(issues=[], errors=[], hard_ok=hard_ok, hard_reasons=reasons, accept=False,
                  verifier='edit_level_v1', verdict='unknown', edits=[],
                  edit_count=verification['edit_count'], mechanical_checks={
                      key: verification[key] for key in ('counts', 'budget', 'style_checks')})
    if not hard_ok:
        result.update(verdict='fail', skipped='mechanical_constraint')
        result['selectivity'] = 'fail'
        if 'speech_level_changed' in reasons:
            result['korean_consistency'] = 'fail'
        return result
    if not verification['edits']:
        result.update(verdict='fail', hard_ok=False, hard_reasons=['no_detected_edits'], skipped='mechanical_constraint')
        return result
    try:
        # Neither complete before/after essays nor a flattened morphological
        # fragment list is sent. Only bounded contexts accompany each edit.
        contract = plan.model_dump()
        contract['target'].pop('text')
        contract['target'].pop('pieces')
        response = llm.request(EditJudgments, prompt, {
            'question': question, 'plan': contract, 'edits': verification['edits'],
            'edit_count': verification['edit_count'], 'raw_diff_count': verification['raw_diff_count'],
            'budget': verification['budget'],
        }, role="rv", retries=2,
            validate=lambda output: _validate_edit_judgments(output, verification['edits']))
        # Keep the prior four keys as derived summaries for the unchanged
        # Planner/history interface, never as an LLM essay-level judgment.
        judgments = {item.edit_id: item for item in response.edits}
        for edit in verification['edits']:
            item = judgments[edit['id']]
            row = item.model_dump()
            row['verdict'] = _aggregate(getattr(item, key) for key in EDIT_REQUIREMENTS)
            result['edits'].append(row)
            for issue in item.issues:
                requirement = {'necessity': 'selectivity', 'groundedness': 'preservation',
                               'meaning': 'preservation'}.get(issue.requirement, issue.requirement)
                result['issues'].append({**issue.model_dump(), 'requirement': requirement,
                                         'edit_requirement': issue.requirement, 'location': item.edit_id})
        result['goal'] = result['selectivity'] = _aggregate(item.necessity for item in response.edits)
        result['preservation'] = _aggregate(getattr(item, key) for item in response.edits
                                            for key in ('preservation', 'groundedness', 'meaning'))
        result['korean_consistency'] = _aggregate(item.korean_consistency for item in response.edits)
        result['verdict'] = _aggregate(item['verdict'] for item in result['edits'])
        result['accept'] = result['verdict'] == 'pass'
    except JSONFailure as exc:
        result["errors"].append({"stage": "rv", "error_type": type(exc).__name__})
    # Budget exhaustion is handled by the loop, which retains the last accepted state.
    return result
