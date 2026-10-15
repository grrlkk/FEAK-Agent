"""Expose the existing scorer's native output without rescaling or retraining."""

from dataclasses import dataclass
import math

from feak_tc.diagnose.constants import RUBRIC_KEYS, scores_to_rubric_dict


@dataclass
class ScoreState:
    scores: dict
    metadata: dict


class StateScorer:
    def __init__(self, diagnoser, source="rf_corrected_score"):
        if source not in {"rf_corrected_score", "soft_mean", "final"}:
            raise ValueError("Unknown score source")
        self.diagnoser = diagnoser
        self.source = source
        self.cache = {}

    def score(self, text):
        if text not in self.cache:
            diagnosis = self.diagnoser.diagnose(text)
            if self.source == "final":
                scores = dict(diagnosis.rubrics)
            else:
                scores = scores_to_rubric_dict(diagnosis.metadata[self.source])
            if set(scores) != set(RUBRIC_KEYS) or not all(math.isfinite(x) for x in scores.values()):
                raise ValueError("Scorer must return eight finite rubric values")
            self.cache[text] = ScoreState(scores, {
                "source": self.source, "native_rating_range": [1, 9], "rescaled": False,
                "note": "RF continuous correction can exceed the native grade endpoints; no clipping/rounding.",
                "final_grades": dict(diagnosis.rubrics),
                "soft_mean": diagnosis.metadata.get("soft_mean"),
                "soft_std": diagnosis.metadata.get("soft_std"),
                "rf_corrected_score": diagnosis.metadata.get("rf_corrected_score"),
            })
        return self.cache[text]
