"""Heuristic transition selector for MVP."""

from __future__ import annotations

from typing import Any, Mapping, Optional

from feak_tc.diagnose.constants import MAX_SCORE, MIN_SCORE

from .schemas import Candidate, CandidateResult, Decision, Transition


DEFAULT_HEURISTIC_CFG = {
    "accept_threshold": 0.0,
    # Kanana rubric deltas are integer steps on a 1-9 scale; normalize them to
    # [0, 1] so they are commensurable with the other 0-1 signals.
    "rubric_score_range": float(MAX_SCORE - MIN_SCORE),
    "hard_constraints": {
        "target_gain_min": None,
        "non_target_drop_max": 1.0,
        "edit_ratio_max": 0.5,
        "goal_preservation_min": 0.7,
        "evidence_match_min": 0.3,
    },
    "weights": {
        "target_gain": 2.0,
        "target_gap_reduction": 1.0,
        "evidence_match": 0.5,
        "goal_preservation": 0.5,
        "non_target_drop": -2.0,
        "edit_ratio": -0.5,
    },
}


def heuristic_score(transition: Transition, cfg: Optional[Mapping[str, Any]] = None) -> float:
    cfg = _merged_cfg(cfg)
    weights = cfg["weights"]
    rubric_range = max(1.0, float(cfg["rubric_score_range"]))
    return float(
        weights["target_gain"] * (transition.target_gain / rubric_range)
        + weights["target_gap_reduction"] * transition.target_gap_reduction
        + weights["evidence_match"] * transition.evidence_match
        + weights["goal_preservation"] * transition.goal_preservation
        + weights["non_target_drop"] * (transition.non_target_drop / rubric_range)
        + weights["edit_ratio"] * transition.edit_ratio
    )


def build_result(
    candidate: Candidate,
    transition: Transition,
    cfg: Optional[Mapping[str, Any]] = None,
    extra_reject_reasons: Optional[list[str]] = None,
) -> CandidateResult:
    cfg = _merged_cfg(cfg)
    score = heuristic_score(transition, cfg)
    reasons = list(extra_reject_reasons or [])
    reasons += _hard_constraint_violations(transition, cfg)
    if candidate.action_type == "STOP":
        reasons = []
    elif not extra_reject_reasons and _has_no_effect(transition):
        reasons.append("no_effect")
    return CandidateResult(
        candidate=candidate,
        transition=transition,
        heuristic_score=score,
        rejected=bool(reasons),
        reject_reasons=reasons,
    )


def select(results: list[CandidateResult], cfg: Optional[Mapping[str, Any]] = None) -> Decision:
    cfg = _merged_cfg(cfg)
    if not results:
        return Decision("stop", None, "No candidates were produced.", [])

    scores = [result.heuristic_score for result in results]
    viable = [
        (idx, result)
        for idx, result in enumerate(results)
        if not result.rejected and result.candidate.action_type != "STOP"
    ]
    if not viable:
        stop_idx = next(
            (
                idx
                for idx, result in enumerate(results)
                if not result.rejected and result.candidate.action_type == "STOP"
            ),
            None,
        )
        if stop_idx is not None:
            return Decision(
                "stop",
                stop_idx,
                "No viable non-STOP candidates remain; selecting STOP.",
                scores,
            )
        return Decision("reject_all", None, "All non-STOP candidates violated hard constraints.", scores)

    chosen_idx, chosen = max(viable, key=lambda item: item[1].heuristic_score)
    if chosen.heuristic_score < float(cfg["accept_threshold"]):
        return Decision(
            "stop",
            None,
            f"Best score {chosen.heuristic_score:.3f} is below accept_threshold.",
            scores,
        )
    return Decision(
        "accept",
        chosen_idx,
        f"Candidate {chosen_idx} has the highest viable heuristic score.",
        scores,
    )


def _hard_constraint_violations(transition: Transition, cfg: Mapping[str, Any]) -> list[str]:
    hard = cfg["hard_constraints"]
    reasons = []
    gain_min = hard.get("target_gain_min")
    if gain_min is not None and transition.target_gain < float(gain_min):
        reasons.append("target_gain")
    if transition.non_target_drop > float(hard["non_target_drop_max"]):
        reasons.append("non_target_drop")
    if transition.edit_ratio > float(hard["edit_ratio_max"]):
        reasons.append("edit_ratio")
    if transition.goal_preservation < float(hard["goal_preservation_min"]):
        reasons.append("goal_preservation")
    if transition.evidence_match < float(hard["evidence_match_min"]):
        reasons.append("evidence_match")
    return reasons


def _has_no_effect(transition: Transition) -> bool:
    eps = 1e-9
    return (
        abs(transition.edit_ratio) <= eps
        and abs(transition.target_gain) <= eps
        and abs(transition.target_gap_reduction) <= eps
    )


def _merged_cfg(cfg: Optional[Mapping[str, Any]]) -> dict[str, Any]:
    merged = {
        "accept_threshold": DEFAULT_HEURISTIC_CFG["accept_threshold"],
        "rubric_score_range": DEFAULT_HEURISTIC_CFG["rubric_score_range"],
        "hard_constraints": dict(DEFAULT_HEURISTIC_CFG["hard_constraints"]),
        "weights": dict(DEFAULT_HEURISTIC_CFG["weights"]),
    }
    if not cfg:
        return merged
    if "accept_threshold" in cfg:
        merged["accept_threshold"] = cfg["accept_threshold"]
    if "rubric_score_range" in cfg:
        merged["rubric_score_range"] = cfg["rubric_score_range"]
    if "hard_constraints" in cfg and isinstance(cfg["hard_constraints"], Mapping):
        merged["hard_constraints"].update(cfg["hard_constraints"])
    if "weights" in cfg and isinstance(cfg["weights"], Mapping):
        merged["weights"].update(cfg["weights"])
    return merged
