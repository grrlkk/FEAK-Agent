"""Dataclasses for the one-step FEAK-TC MVP loop."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Optional

from feak_tc.diagnose.constants import ACTION_TYPES, RUBRIC_KEYS, validate_action_type


@dataclass
class Patch:
    operation: str
    target_span: str
    before: str
    after: str
    reason: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class Candidate:
    action_type: str
    target_rubric: str
    target_span: str
    instruction: str
    new_text: Optional[str] = None
    patch: Optional[Patch] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.action_type = validate_action_type(self.action_type)
        if self.target_rubric not in RUBRIC_KEYS:
            raise ValueError(f"Unknown target_rubric: {self.target_rubric}")

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        if self.patch is not None:
            data["patch"] = self.patch.to_dict()
        return data


@dataclass
class Transition:
    action_type: str
    target_rubric: str
    target_gain: float
    non_target_drop: float
    target_gap_reduction: float
    evidence_match: float
    edit_ratio: float
    goal_preservation: float
    emb_sim: float

    def __post_init__(self) -> None:
        self.action_type = validate_action_type(self.action_type)
        if self.target_rubric not in RUBRIC_KEYS:
            raise ValueError(f"Unknown target_rubric: {self.target_rubric}")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class CandidateResult:
    candidate: Candidate
    transition: Transition
    heuristic_score: float
    rejected: bool = False
    reject_reasons: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "candidate": self.candidate.to_dict(),
            "transition": self.transition.to_dict(),
            "heuristic_score": self.heuristic_score,
            "rejected": self.rejected,
            "reject_reasons": list(self.reject_reasons),
        }


@dataclass
class Decision:
    decision: str
    chosen_index: Optional[int]
    reason: str
    scores: list[float]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
