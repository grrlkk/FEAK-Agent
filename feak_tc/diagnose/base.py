"""Common diagnoser interface for FEAK-TC."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Protocol

from .constants import RUBRIC_KEYS, require_feak_features, require_rubric_scores


@dataclass
class Diagnosis:
    """Canonical observation used by the MVP controller."""

    text: str
    rubrics: dict[str, float]
    features: dict[str, float]
    weak_rubrics: list[str]
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.rubrics = require_rubric_scores(self.rubrics)
        self.features = require_feak_features(self.features)
        self.weak_rubrics = [key for key in self.weak_rubrics if key in RUBRIC_KEYS]


class Diagnoser(Protocol):
    def diagnose(self, text: str) -> Diagnosis:
        """Return rubric scores, FEAK features, and weak-rubric selection."""


def select_weak_rubrics(
    rubrics: Mapping[str, float],
    top_n: int = 3,
    threshold: Optional[float] = None,
) -> list[str]:
    """Select low-scoring rubrics using a simple deterministic MVP rule."""

    normalized = require_rubric_scores(rubrics)
    ranked = sorted(normalized.items(), key=lambda item: (item[1], RUBRIC_KEYS.index(item[0])))
    if threshold is not None:
        weak = [key for key, score in ranked if score <= threshold]
        if weak:
            return weak[:top_n]
    return [key for key, _ in ranked[:top_n]]
