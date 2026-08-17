"""FEAK-TC v2 corruption-chain generation.

Generation is intentionally diagnosis-free: no rubric score or feature value
is exposed to an operator. G1 measures every generated state afterward and
keeps only steps whose predeclared target rubric actually decreases.
"""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json

from .normalize import normalize_text
from .operators import (
    MAIN_CHAIN_OPERATORS,
    OPERATOR_SPECS,
    build_payload_schema,
    build_prompt,
    generate_rule_payload,
    parse_and_apply,
    validate_normalized_corruption,
    validate_operator_preservation,
)

_SYSTEM_PROMPT = (
    "You apply one controlled degradation to Korean essays for research data. "
    "Preserve every dimension not named in the concrete edit instructions. "
    "Return only a JSON object."
)


def generate_chain(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """Generate x0→...→xK with operator metadata and uniform normalization."""

    rng = rng or random.Random(f"{cfg.get('seed', 0)}:{record['record_id']}")
    depth = int(cfg.get("depth", 3))
    llm_cfg = dict(cfg.get("llm", {}))
    normalization_cfg = dict(cfg.get("normalization", {}))
    generation_modes = list(cfg.get("generation", {}).get("modes", ["rule"]))
    validity_cfg = dict(cfg.get("validity", {}))
    configured_specs = dict(cfg.get("operators", {}))

    operator_names = sorted(set(MAIN_CHAIN_OPERATORS) & set(configured_specs))
    if not operator_names:
        return {
            "record_id": record["record_id"],
            "question": str(record.get("question") or "다음 글을 평가하세요."),
            "grader_avg": record.get("grader_avg"),
            "planned_operators": [],
            "status": "failed",
            "failure_errors": ["no main-chain corruption operators are configured"],
            "states": [],
            "normalizations": [],
            "steps": [],
        }
    operators = rng.sample(operator_names, min(depth, len(operator_names)))
    source_raw = _normalize_spaces(str(record["text"]))
    question = str(record.get("question") or "다음 글을 평가하세요.")

    try:
        source_text, source_norm = normalize_text(source_raw, normalization_cfg)
    except RuntimeError as exc:
        return {
            "record_id": record["record_id"],
            "question": question,
            "grader_avg": record.get("grader_avg"),
            "planned_operators": operators,
            "status": "failed",
            "failure_errors": [str(exc)],
            "states": [],
            "normalizations": [],
            "steps": [],
        }

    states = [source_text]
    normalizations = [source_norm]
    steps: list[dict[str, Any]] = []
    failure: Optional[list[str]] = None

    for step_idx, operator in enumerate(operators):
        spec = _merged_spec(operator, configured_specs[operator])
        operator_modes = list(spec.get("generation_modes", generation_modes))
        generator = _select_generator(
            operator_modes,
            record_id=str(record["record_id"]),
            operator=operator,
            step_idx=step_idx,
        )
        step, errors = _generate_step(
            operator=operator,
            generator=generator,
            text=states[-1],
            source_text=source_text,
            question=question,
            llm_cfg=llm_cfg,
            normalization_cfg=normalization_cfg,
            validity_cfg=validity_cfg,
            spec=spec,
            rng=rng,
        )
        if step is None:
            failure = errors
            break
        states.append(step.pop("new_text"))
        normalizations.append(step["normalization"])
        steps.append(step)

    status = "ok" if len(steps) == len(operators) else ("partial" if steps else "failed")
    return {
        "record_id": record["record_id"],
        "question": question,
        "grader_avg": record.get("grader_avg"),
        "planned_operators": operators,
        "status": status,
        "failure_errors": failure,
        "states": states,
        "normalizations": normalizations,
        "steps": steps,
    }


def _generate_step(
    *,
    operator: str,
    generator: str,
    text: str,
    source_text: str,
    question: str,
    llm_cfg: Mapping[str, Any],
    normalization_cfg: Mapping[str, Any],
    validity_cfg: Mapping[str, Any],
    spec: Mapping[str, Any],
    rng: random.Random,
) -> tuple[Optional[dict[str, Any]], list[str]]:
    attempts = 1 if generator == "rule" else int(llm_cfg.get("max_attempts", 3))
    model = str(llm_cfg.get("model", "gpt-4o-mini"))
    reasoning_effort = _nested_string(llm_cfg, "reasoning", "effort")
    verbosity = _nested_string(llm_cfg, "text", "verbosity")
    generation_uses_llm = generator != "rule"
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        try:
            if generator == "rule":
                payload = generate_rule_payload(operator, text, spec, rng)
            else:
                variant = generator.split(":", 1)[1]
                payload = request_json(
                    system=_SYSTEM_PROMPT,
                    user=build_prompt(operator, text, question, spec, variant),
                    model=model,
                    temperature=(
                        float(llm_cfg["temperature"])
                        if llm_cfg.get("temperature") is not None
                        else None
                    ),
                    reasoning_effort=reasoning_effort,
                    verbosity=verbosity,
                    json_schema=build_payload_schema(operator, spec),
                    response_format_name=f"corruption_{operator.lower()}",
                    timeout=float(llm_cfg["timeout"]) if llm_cfg.get("timeout") is not None else None,
                )
            raw_new_text, edits = parse_and_apply(
                operator,
                payload,
                text,
                source_text,
                validity_cfg,
                spec,
            )
            new_text, normalization = normalize_text(
                raw_new_text,
                normalization_cfg,
                post_validate=lambda candidate: _validate_candidate(
                    operator=operator,
                    before_text=text,
                    candidate=candidate,
                    edits=edits,
                ),
            )
        except (LLMUnavailable, LLMResponseError, ValueError, RuntimeError) as exc:
            errors.append(f"{operator}/{generator} attempt{attempt}: {exc}")
            continue

        preservation_check = validate_operator_preservation(
            operator,
            text,
            new_text,
            edits,
        )
        return {
            "operator": operator,
            "corruption_op": operator,
            "reverse_action": spec["reverse_action"],
            "intent": spec["intent"],
            "target_rubric": spec["target_rubric"],
            "target_features": [],
            "preserve_constraints": list(spec["preserve_constraints"]),
            "generator": generator,
            "requested_generator": generator,
            "fallback": False,
            "model": model if generation_uses_llm else None,
            "reasoning_effort": reasoning_effort if generation_uses_llm else None,
            "verbosity": verbosity if generation_uses_llm else None,
            "edits": edits,
            "attempts": attempt,
            "errors": errors,
            "raw_new_text": raw_new_text,
            "normalization": normalization,
            "normalized": bool(normalization.get("normalized")),
            "preservation_check": preservation_check,
            "new_text": new_text,
        }, errors

    if generator != "rule" and bool(llm_cfg.get("fallback_to_rule", True)):
        try:
            payload = generate_rule_payload(operator, text, spec, rng)
            raw_new_text, edits = parse_and_apply(
                operator,
                payload,
                text,
                source_text,
                validity_cfg,
                spec,
            )
            new_text, normalization = normalize_text(
                raw_new_text,
                normalization_cfg,
                post_validate=lambda candidate: _validate_candidate(
                    operator=operator,
                    before_text=text,
                    candidate=candidate,
                    edits=edits,
                ),
            )
        except (LLMUnavailable, LLMResponseError, ValueError, RuntimeError) as exc:
            errors.append(f"{operator}/rule_fallback: {exc}")
        else:
            preservation_check = validate_operator_preservation(
                operator,
                text,
                new_text,
                edits,
            )
            return {
                "operator": operator,
                "corruption_op": operator,
                "reverse_action": spec["reverse_action"],
                "intent": spec["intent"],
                "target_rubric": spec["target_rubric"],
                "target_features": [],
                "preserve_constraints": list(spec["preserve_constraints"]),
                "generator": "rule_fallback",
                "requested_generator": generator,
                "fallback": True,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "verbosity": verbosity,
                "edits": edits,
                "attempts": attempts + 1,
                "errors": errors,
                "raw_new_text": raw_new_text,
                "normalization": normalization,
                "normalized": bool(normalization.get("normalized")),
                "preservation_check": preservation_check,
                "new_text": new_text,
            }, errors
    return None, errors


def _merged_spec(operator: str, configured: Mapping[str, Any]) -> dict[str, Any]:
    base = dict(OPERATOR_SPECS[operator])
    merged = {**base, **dict(configured)}
    if merged["target_rubric"] != base["target_rubric"]:
        raise ValueError(
            f"{operator} target_rubric config {merged['target_rubric']} "
            f"does not match v2 mapping {base['target_rubric']}"
        )
    if tuple(merged["preserve_constraints"]) != tuple(base["preserve_constraints"]):
        raise ValueError(f"{operator} preserve_constraints do not match the v2 mapping")
    return merged


def _validate_candidate(
    *,
    operator: str,
    before_text: str,
    candidate: str,
    edits: list[Mapping[str, Any]],
) -> None:
    validate_normalized_corruption(operator, edits, candidate)
    validate_operator_preservation(operator, before_text, candidate, edits)


def _select_generator(
    modes: list[str],
    *,
    record_id: str,
    operator: str,
    step_idx: int,
) -> str:
    if not modes:
        raise ValueError("generation.modes must not be empty")
    index = sum(ord(char) for char in f"{record_id}:{operator}") + step_idx
    selected = str(modes[index % len(modes)])
    if selected != "rule" and not selected.startswith("llm:"):
        raise ValueError(f"Unknown generator mode: {selected}")
    return selected


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _nested_string(
    cfg: Mapping[str, Any],
    section: str,
    key: str,
) -> Optional[str]:
    nested = cfg.get(section)
    if not isinstance(nested, Mapping):
        return None
    value = nested.get(key)
    return str(value) if value is not None else None
