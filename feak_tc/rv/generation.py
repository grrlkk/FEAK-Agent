"""LLM generation for the RV candidate types unavailable in trajectories."""

from __future__ import annotations

import json
from typing import Any, Callable, Mapping

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json


LLM_CANDIDATE_TYPES = ("wrong_target", "over_edit")

_SYSTEM_PROMPT = (
    "You create controlled Korean essay revisions for Revision Verifier research. "
    "Follow the requested candidate semantics exactly, preserve the original writing "
    "task, and return only the schema-conforming JSON object."
)


def llm_candidate_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            name: {"type": "string", "minLength": 1}
            for name in LLM_CANDIDATE_TYPES
        },
        "required": list(LLM_CANDIDATE_TYPES),
        "additionalProperties": False,
    }


def generate_llm_candidate_texts(
    anchor: Any,
    llm_cfg: Mapping[str, Any],
    *,
    request_fn: Callable[..., dict[str, Any]] = request_json,
) -> dict[str, str]:
    """Generate wrong-target and over-edit candidates in one call."""

    model = str(llm_cfg["model"])
    attempts = int(llm_cfg.get("max_attempts", 3))
    if attempts < 1:
        raise ValueError("RV LLM max_attempts must be positive")
    reasoning = llm_cfg.get("reasoning", {})
    text_cfg = llm_cfg.get("text", {})
    validation_cfg = llm_cfg.get("validation", {})
    errors: list[str] = []
    texts: dict[str, str] | None = None
    for attempt in range(1, attempts + 1):
        prompt = _build_prompt(anchor)
        if errors:
            prompt += (
                "\n\nThe previous response failed validation. Correct all of these "
                f"problems without explaining them: {errors[-1]}"
            )
        try:
            response = request_fn(
                system=_SYSTEM_PROMPT,
                user=prompt,
                model=model,
                temperature=None,
                reasoning_effort=_optional_string(reasoning.get("effort")),
                verbosity=_optional_string(text_cfg.get("verbosity")),
                json_schema=llm_candidate_schema(),
                response_format_name="rv_candidate_pair",
                timeout=(
                    float(llm_cfg["timeout"])
                    if llm_cfg.get("timeout") is not None
                    else None
                ),
            )
            texts = {name: str(response[name]).strip() for name in LLM_CANDIDATE_TYPES}
            break
        except (KeyError, LLMResponseError, LLMUnavailable, ValueError) as exc:
            errors.append(f"attempt {attempt}: {exc}")
    if texts is None:
        raise RuntimeError("RV candidate generation failed: " + " | ".join(errors))

    repair_attempts = int(llm_cfg.get("candidate_repair_attempts", 3))
    for repair_round in range(1, repair_attempts + 1):
        candidate_errors = _candidate_validation_errors(anchor, texts, validation_cfg)
        if not candidate_errors:
            return texts
        for candidate_type, candidate_error in candidate_errors.items():
            try:
                response = request_fn(
                    system=_SYSTEM_PROMPT,
                    user=_build_candidate_repair_prompt(
                        anchor,
                        candidate_type,
                        texts,
                        candidate_error,
                        validation_cfg,
                    ),
                    model=model,
                    temperature=None,
                    reasoning_effort=_optional_string(reasoning.get("effort")),
                    verbosity=_optional_string(text_cfg.get("verbosity")),
                    json_schema=_single_candidate_schema(candidate_type),
                    response_format_name=f"rv_{candidate_type}_repair",
                    timeout=(
                        float(llm_cfg["timeout"])
                        if llm_cfg.get("timeout") is not None
                        else None
                    ),
                )
                texts[candidate_type] = str(response[candidate_type]).strip()
            except (KeyError, LLMResponseError, LLMUnavailable, ValueError) as exc:
                errors.append(
                    f"repair round {repair_round} {candidate_type}: {exc}"
                )
    try:
        validate_llm_candidate_texts(anchor, texts, validation_cfg)
    except ValueError as exc:
        errors.append(f"final validation: {exc}")
        raise RuntimeError("RV candidate generation failed: " + " | ".join(errors))
    return texts


def validate_llm_candidate_texts(
    anchor: Any,
    texts: Mapping[str, str],
    validation_cfg: Mapping[str, Any],
) -> None:
    errors = _candidate_validation_errors(anchor, texts, validation_cfg)
    if errors:
        details = "; ".join(
            f"{candidate_type}: {message}"
            for candidate_type, message in errors.items()
        )
        raise ValueError(details)


def _candidate_validation_errors(
    anchor: Any,
    texts: Mapping[str, str],
    validation_cfg: Mapping[str, Any],
) -> dict[str, str]:
    current = str(anchor.current_text)
    reserved = {
        current,
        str(anchor.previous_text),
        str(anchor.partial_text),
        str(anchor.next_text),
    }
    min_ratio = float(validation_cfg.get("min_length_ratio", 0.60))
    max_ratio = float(validation_cfg.get("max_length_ratio", 1.80))
    min_ratio_by_candidate = validation_cfg.get("min_length_ratio_by_candidate", {})
    max_ratio_by_candidate = validation_cfg.get("max_length_ratio_by_candidate", {})
    errors: dict[str, str] = {}
    seen: dict[str, str] = {}
    for candidate_type in LLM_CANDIDATE_TYPES:
        if candidate_type not in texts:
            errors[candidate_type] = "candidate is missing"
            continue
        value = str(texts[candidate_type]).strip()
        if not value:
            errors[candidate_type] = "candidate is empty"
            continue
        if value in reserved:
            errors[candidate_type] = "candidate copies a trajectory state"
            continue
        if value in seen:
            errors[candidate_type] = f"candidate duplicates {seen[value]}"
            continue
        seen[value] = candidate_type
        ratio = len(value) / max(1, len(current))
        candidate_min_ratio = float(
            min_ratio_by_candidate.get(candidate_type, min_ratio)
        )
        candidate_max_ratio = float(
            max_ratio_by_candidate.get(candidate_type, max_ratio)
        )
        if ratio < candidate_min_ratio or ratio > candidate_max_ratio:
            errors[candidate_type] = (
                f"length ratio {ratio:.3f} is outside "
                f"[{candidate_min_ratio:.3f}, {candidate_max_ratio:.3f}]"
            )
    return errors


def _build_prompt(anchor: Any) -> str:
    row = anchor.row
    payload = {
        "question": anchor.question,
        "target_rubric": row["target_rubric"],
        "intended_action": row["reverse_action"],
        "intent": row.get("intent", ""),
        "corruption_type": row["corruption_op"],
        "known_corruption_edits": row.get("edits", []),
        "current_corrupted_state": anchor.current_text,
        "exact_target_repair_reference": anchor.previous_text,
        "forbidden_partial_repair_state": anchor.partial_text,
        "wrong_target_rubric_for_candidate": wrong_target_rubric(
            str(row["target_rubric"])
        ),
    }
    return (
        "Create exactly two full revised texts from the current corrupted state.\n"
        "- wrong_target: leave the known target defect unresolved and make one plausible, "
        "limited edit only to wrong_target_rubric_for_candidate. Do not introduce a second "
        "severe defect.\n"
        "- over_edit: fully repair the target defect, but also rewrite, add, move, or delete "
        "substantial non-target material unnecessarily. Keep it recognizable as the same essay.\n"
        "Each value must be the complete Korean essay after revision. Do not copy the exact "
        "repair reference for any candidate and do not return commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def _build_candidate_repair_prompt(
    anchor: Any,
    candidate_type: str,
    texts: Mapping[str, str],
    error: str,
    validation_cfg: Mapping[str, Any],
) -> str:
    current_length = len(str(anchor.current_text))
    default_min = float(validation_cfg.get("min_length_ratio", 0.60))
    default_max = float(validation_cfg.get("max_length_ratio", 1.80))
    minimum = float(
        validation_cfg.get("min_length_ratio_by_candidate", {}).get(
            candidate_type, default_min
        )
    )
    maximum = float(
        validation_cfg.get("max_length_ratio_by_candidate", {}).get(
            candidate_type, default_max
        )
    )
    forbidden = [
        str(anchor.current_text),
        str(anchor.previous_text),
        str(anchor.partial_text),
        str(anchor.next_text),
        *(str(value) for name, value in texts.items() if name != candidate_type),
    ]
    return (
        f"Regenerate only `{candidate_type}` because the prior value failed: {error}.\n"
        f"Semantic requirement: {_candidate_instruction(candidate_type)}\n"
        f"Return the complete Korean essay, between {int(current_length * minimum)} and "
        f"{int(current_length * maximum)} characters. It must differ from every forbidden "
        "text below. Return only the requested JSON field.\n\n"
        + json.dumps(
            {
                "question": anchor.question,
                "target_rubric": anchor.row["target_rubric"],
                "wrong_target_rubric_for_candidate": wrong_target_rubric(
                    str(anchor.row["target_rubric"])
                ),
                "intended_action": anchor.row["reverse_action"],
                "corruption_type": anchor.row["corruption_op"],
                "known_corruption_edits": anchor.row.get("edits", []),
                "current_corrupted_state": anchor.current_text,
                "exact_target_repair_reference": anchor.previous_text,
                "forbidden_partial_repair_state": anchor.partial_text,
                "forbidden_exact_texts": forbidden,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _candidate_instruction(candidate_type: str) -> str:
    return {
        "wrong_target": (
            "Leave the target defect unresolved and make one limited, plausible edit to a "
            "different rubric; this cannot be NO_EDIT."
        ),
        "over_edit": (
            "Fully repair the target defect but also change substantial non-target material "
            "unnecessarily while keeping the same essay recognizable."
        ),
    }[candidate_type]


def wrong_target_rubric(target_rubric: str) -> str:
    # A concrete non-target axis makes the negative candidate reproducible.
    return "organization_1" if target_rubric == "expression_2" else "expression_2"


def _single_candidate_schema(candidate_type: str) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {candidate_type: {"type": "string", "minLength": 1}},
        "required": [candidate_type],
        "additionalProperties": False,
    }


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None
