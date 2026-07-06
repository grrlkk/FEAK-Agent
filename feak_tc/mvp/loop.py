"""One-step FEAK-TC MVP loop."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from feak_tc.diagnose import Diagnoser, get_diagnoser

from .heuristic import build_result, select
from .patch import apply_patch
from .propose import propose
from .transition import compute_transition


def one_step(
    text: str,
    diagnoser: Optional[Diagnoser] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    cfg = cfg or {}
    diagnoser = diagnoser or get_diagnoser("stub", weak_top_n=int(cfg.get("weak_rubric_top_n", 3)))
    before = diagnoser.diagnose(text)
    candidates = propose(before, n_per_action=int(cfg.get("n_per_action", 1)), cfg=cfg)

    results = []
    for candidate in candidates:
        patched = apply_patch(text, candidate, cfg=cfg)
        after_text = patched.new_text if patched.new_text is not None else text
        after = before if after_text == text else diagnoser.diagnose(after_text)
        transition = compute_transition(before, after, patched)
        results.append(build_result(patched, transition, cfg))

    decision = select(results, cfg)
    return {
        "before": before,
        "results": results,
        "decision": decision,
    }


def serializable_one_step(
    text: str,
    diagnoser: Optional[Diagnoser] = None,
    cfg: Optional[Mapping[str, Any]] = None,
) -> dict[str, Any]:
    output = one_step(text=text, diagnoser=diagnoser, cfg=cfg)
    return {
        "before": {
            "text": output["before"].text,
            "rubrics": output["before"].rubrics,
            "features": output["before"].features,
            "weak_rubrics": output["before"].weak_rubrics,
            "metadata": output["before"].metadata,
        },
        "results": [result.to_dict() for result in output["results"]],
        "decision": output["decision"].to_dict(),
    }
