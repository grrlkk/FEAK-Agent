"""One-step FEAK-TC MVP loop."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from feak_tc.diagnose import Diagnosis, Diagnoser, get_diagnoser

from .heuristic import build_result, select
from .patch import apply_patch
from .propose import propose
from .surface import normalize_surface_text
from .transition import compute_transition
from .validity import patch_validity_violations


def one_step(
    text: str,
    diagnoser: Optional[Diagnoser] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    diagnoser = diagnoser or get_diagnoser("stub", weak_top_n=int(cfg.get("weak_rubric_top_n", 3)))
    before = diagnoser.diagnose(text)
    surface = normalize_surface_text(text, cfg=cfg)
    working_text = surface.normalized_text if surface is not None else text
    action_before = before
    if surface is not None:
        before.metadata["surface_normalizer"] = surface.summary_dict()
        if working_text != text:
            action_before = diagnoser.diagnose(working_text)
            action_before.weak_rubrics = list(before.weak_rubrics)
            action_before.metadata["surface_normalizer"] = surface.summary_dict()
            action_before.metadata["weak_rubrics_source"] = "pre_surface_diagnosis"
    candidates = propose(action_before, n_per_action=int(cfg.get("n_per_action", 1)), cfg=cfg)

    results = []
    for candidate in candidates:
        patched = apply_patch(working_text, candidate, cfg=cfg)
        after_text = patched.new_text if patched.new_text is not None else working_text
        violations = patch_validity_violations(working_text, patched, cfg=cfg)
        if violations:
            # Structurally invalid patch: reject without spending a re-diagnosis.
            after = action_before
        else:
            after = action_before if after_text == working_text else diagnoser.diagnose(after_text)
        transition = compute_transition(action_before, after, patched)
        results.append(build_result(patched, transition, cfg, extra_reject_reasons=violations))

    decision = select(results, cfg)
    output = {
        "before": before,
        "results": results,
        "decision": decision,
    }
    if action_before is not before:
        output["action_before"] = action_before
    if surface is not None:
        output["surface_normalization"] = surface
    return output


def serializable_one_step(
    text: str,
    diagnoser: Optional[Diagnoser] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    output = one_step(text=text, diagnoser=diagnoser, cfg=cfg)
    serialized = {
        "before": _serialize_diagnosis(output["before"]),
        "results": [result.to_dict() for result in output["results"]],
        "decision": output["decision"].to_dict(),
    }
    if "action_before" in output:
        serialized["action_before"] = _serialize_diagnosis(output["action_before"])
    surface = output.get("surface_normalization")
    if surface is not None:
        serialized["surface_normalization"] = surface.to_dict()
    return serialized


def _serialize_diagnosis(diag: Diagnosis) -> dict[str, Any]:
    return {
        "text": diag.text,
        "rubrics": diag.rubrics,
        "features": diag.features,
        "weak_rubrics": diag.weak_rubrics,
        "metadata": diag.metadata,
    }
