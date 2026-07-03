"""Legacy KoBERT/UKTA diagnoser adapter.

This wraps the existing src/apps code without modifying it. The mapping from
the 11 legacy rubric outputs to the Kanana 8-rubric schema is approximate and
intended only for baseline plumbing.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Optional

from .base import Diagnosis, select_weak_rubrics
from .constants import FEAK_FEATURE_NAMES


_LEGACY_TO_KANANA = {
    "task_1": ["prompt_comprehension"],
    "content_1": ["topic_clarity"],
    "content_2": ["narrative"],
    "content_3": ["originality"],
    "organization_1": ["intra_paragraph_structure"],
    "organization_2": ["inter_paragraph_structure", "structural_consistency"],
    "expression_1": ["vocabulary", "sentence_expression"],
    "expression_2": ["grammar"],
}


class FeakKobertDiagnoser:
    def __init__(self, weak_top_n: int = 3) -> None:
        self.weak_top_n = weak_top_n
        self._process_module: Optional[Any] = None

    def diagnose(self, text: str) -> Diagnosis:
        process_module = self._ensure_loaded()
        result = process_module.process(text)
        legacy_scores = result.get("essay_score", {})
        if not isinstance(legacy_scores, dict):
            legacy_scores = {}
        rubrics = {
            key: _scale_legacy_score(_mean(legacy_scores.get(name, 0.0) for name in names))
            for key, names in _LEGACY_TO_KANANA.items()
        }
        features = _flatten_legacy_features(result)
        weak = select_weak_rubrics(rubrics, top_n=self.weak_top_n)
        return Diagnosis(
            text=text,
            rubrics=rubrics,
            features=features,
            weak_rubrics=weak,
            metadata={"diagnoser": "feak_kobert", "legacy_raw": legacy_scores},
        )

    def _ensure_loaded(self):
        if self._process_module is not None:
            return self._process_module
        project_root = Path(__file__).resolve().parents[2]
        src_path = project_root / "src"
        if str(src_path) not in sys.path:
            sys.path.insert(0, str(src_path))
        from apps.cohesion import process as process_module

        process_module.initialize_models()
        self._process_module = process_module
        return self._process_module


def _scale_legacy_score(score_0_to_3: float) -> float:
    return max(1.0, min(9.0, 1.0 + float(score_0_to_3) * (8.0 / 3.0)))


def _mean(values) -> float:
    values = [float(value) for value in values]
    return sum(values) / len(values) if values else 0.0


def _flatten_legacy_features(result: dict[str, Any]) -> dict[str, float]:
    flattened: dict[str, float] = {}
    for value in result.values():
        if isinstance(value, dict):
            for inner_key, inner_value in value.items():
                if isinstance(inner_value, (int, float)):
                    flattened[inner_key] = float(inner_value)
    return {name: float(flattened.get(name, 0.0)) for name in FEAK_FEATURE_NAMES}
