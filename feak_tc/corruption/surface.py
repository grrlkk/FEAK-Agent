"""Chain-external mechanical-error samples for surface-correction validation."""

from __future__ import annotations

import random
from typing import Any, Mapping, Optional

from .operators import (
    OPERATOR_SPECS,
    SURFACE_VALIDATION_OPERATOR,
    generate_rule_payload,
    parse_and_apply,
    validate_operator_preservation,
)


def generate_surface_sample(
    record: Mapping[str, Any],
    cfg: Mapping[str, Any],
    rng: Optional[random.Random] = None,
) -> dict[str, Any]:
    """Generate one rule-based D_raw input outside the Planner/TVM chain."""

    surface_cfg = dict(cfg.get("surface_validation", {}))
    operator = str(surface_cfg.get("operator", SURFACE_VALIDATION_OPERATOR))
    if operator != SURFACE_VALIDATION_OPERATOR:
        raise ValueError(f"surface validation operator must be {SURFACE_VALIDATION_OPERATOR}")
    if str(surface_cfg.get("generation_mode", "rule")) != "rule":
        raise ValueError("surface validation currently supports rule generation only")

    rng = rng or random.Random(f"{cfg.get('seed', 0)}:surface:{record['record_id']}")
    clean_text = _normalize_spaces(str(record["text"]))
    spec = {
        **OPERATOR_SPECS[operator],
        "edits_per_step": int(surface_cfg.get("edits_per_sample", 3)),
    }
    payload = generate_rule_payload(operator, clean_text, spec, rng)
    surface_input, edits = parse_and_apply(
        operator,
        payload,
        clean_text,
        clean_text,
        dict(cfg.get("validity", {})),
        spec,
    )
    preservation = validate_operator_preservation(
        operator,
        clean_text,
        surface_input,
        edits,
    )
    return {
        "record_id": str(record["record_id"]),
        "question": str(record.get("question") or "다음 글을 평가하세요."),
        "corruption_op": operator,
        "usage": "surface_correction_validation_only",
        "clean_text": clean_text,
        "surface_input_text": surface_input,
        "expected_surface_output": clean_text,
        "edits": edits,
        "preservation_check": preservation,
    }


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())
