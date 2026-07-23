"""Corruption chain generation: x0 → x1 → … with one operator per step."""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json

from .operators import OPERATOR_SPECS, build_prompt, parse_and_apply

_SYSTEM_PROMPT = (
    "You degrade Korean essays in controlled ways to build research training data. "
    "Follow the task exactly and return only a JSON object."
)


def generate_chain(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """Generate one corruption chain for a source essay.

    Returns a serializable dict with states, per-step operators/edits and the
    reverse-direction action labels. status is "ok" when every step applied,
    otherwise "partial" (some steps) or "failed" (none).
    """

    rng = rng or random.Random(f"{cfg.get('seed', 0)}:{record['record_id']}")
    depth = int(cfg.get("depth", 3))
    llm_cfg = dict(cfg.get("llm", {}))
    validity_cfg = dict(cfg.get("validity", {}))
    operators = rng.sample(sorted(OPERATOR_SPECS), min(depth, len(OPERATOR_SPECS)))

    source_text = " ".join(str(record["text"]).split())
    question = str(record.get("question") or "다음 글을 평가하세요.")
    states = [source_text]
    steps: list[dict[str, Any]] = []

    failure: Optional[list[str]] = None
    for operator in operators:
        step, errors = _generate_step(operator, states[-1], source_text, question, llm_cfg, validity_cfg)
        if step is None:
            failure = errors
            break
        steps.append(step)
        states.append(step.pop("new_text"))

    status = "ok" if len(steps) == len(operators) else ("partial" if steps else "failed")
    return {
        "record_id": record["record_id"],
        "question": question,
        "grader_avg": record.get("grader_avg"),
        "planned_operators": operators,
        "status": status,
        "failure_errors": failure,
        "states": states,
        "steps": steps,
    }


def _generate_step(
    operator: str,
    text: str,
    source_text: str,
    question: str,
    llm_cfg: Mapping[str, Any],
    validity_cfg: Mapping[str, Any],
) -> tuple[Optional[dict[str, Any]], list[str]]:
    max_attempts = int(llm_cfg.get("max_attempts", 3))
    errors: list[str] = []
    for attempt in range(1, max_attempts + 1):
        try:
            payload = request_json(
                system=_SYSTEM_PROMPT,
                user=build_prompt(operator, text, question),
                model=str(llm_cfg.get("model", "gpt-4o-mini")),
                temperature=float(llm_cfg.get("temperature", 0.5)),
                timeout=float(llm_cfg["timeout"]) if llm_cfg.get("timeout") is not None else None,
            )
            new_text, edits = parse_and_apply(operator, payload, text, source_text, validity_cfg)
        except (LLMUnavailable, LLMResponseError, ValueError, RuntimeError) as exc:
            errors.append(f"{operator} attempt{attempt}: {exc}")
            continue
        step = {
            "operator": operator,
            "reverse_action": OPERATOR_SPECS[operator]["reverse_action"],
            "intended_rubrics": OPERATOR_SPECS[operator]["intended_rubrics"],
            "edits": edits,
            "attempts": attempt,
            "errors": errors,
            "new_text": new_text,
        }
        return step, errors
    return None, errors
