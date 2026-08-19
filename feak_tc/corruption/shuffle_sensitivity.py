"""Sentence-order perturbations and paired scorer sensitivity summaries."""

from __future__ import annotations

import random
import statistics
from collections import defaultdict
from typing import Any, Mapping, Sequence

from feak_tc.diagnose.constants import RUBRIC_KEYS, scores_to_rubric_dict
from feak_tc.diagnose.stub import split_sentences


SHUFFLE_LEVELS = ("single", "partial", "full")


def build_shuffle_sensitivity_chains(
    records: Sequence[Mapping[str, Any]],
    *,
    levels: Sequence[str] = SHUFFLE_LEVELS,
    replicates: int = 3,
    seed: int = 20260812,
) -> list[dict[str, Any]]:
    """Build paired original/shuffled states without using corruption operators."""

    unknown = sorted(set(levels) - set(SHUFFLE_LEVELS))
    if unknown:
        raise ValueError(f"unknown shuffle sensitivity levels: {unknown}")
    if replicates < 1:
        raise ValueError("replicates must be positive")

    chains = []
    for record in records:
        source_id = str(record["record_id"])
        original = _source_text(record)
        sentences = split_sentences(original)
        if len(sentences) < 4:
            continue
        for level in levels:
            for replicate in range(replicates):
                rng = random.Random(f"{seed}:{source_id}:{level}:{replicate}")
                shuffled = perturb_sentence_order(sentences, level=level, rng=rng)
                chains.append(
                    {
                        "record_id": f"{source_id}:shuffle:{level}:r{replicate + 1}",
                        "source_record_id": source_id,
                        "question": str(
                            record.get("question") or "다음 글을 평가하세요."
                        ),
                        "shuffle_level": level,
                        "replicate": replicate + 1,
                        "seed": seed,
                        "status": "ok",
                        "states": [original, shuffled],
                        "steps": [],
                    }
                )
    return chains


def perturb_sentence_order(
    sentences: Sequence[str],
    *,
    level: str,
    rng: random.Random,
) -> str:
    """Return a sentence-multiset-preserving order perturbation."""

    ordered = list(sentences)
    if len(ordered) < 4:
        raise ValueError("shuffle sensitivity requires at least four sentences")
    if level == "single":
        moved_index = rng.randrange(1, len(ordered) - 1)
        moved = ordered.pop(moved_index)
        destinations = [
            index
            for index in range(len(ordered) + 1)
            if abs(index - moved_index) >= 2
        ]
        destination = rng.choice(destinations)
        ordered.insert(destination, moved)
    elif level == "partial":
        positions = list(range(1, len(ordered) - 1))
        count = max(2, round(len(positions) * 0.5))
        selected = sorted(rng.sample(positions, min(count, len(positions))))
        values = [ordered[index] for index in selected]
        _shuffle_until_changed(values, rng)
        for index, value in zip(selected, values):
            ordered[index] = value
    elif level == "full":
        _shuffle_until_changed(ordered, rng)
    else:
        raise ValueError(f"unknown shuffle sensitivity level: {level}")
    if ordered == list(sentences):
        raise RuntimeError("shuffle sensitivity perturbation produced no change")
    return " ".join(ordered)


def summarize_shuffle_measurements(
    chains: Sequence[Mapping[str, Any]],
    measurements: Mapping[tuple[str, int], Mapping[str, Any]],
    *,
    score_basis: str = "rf_corrected",
    target_rubric: str = "organization_1",
    thresholds: Sequence[float] = (0.225, 0.3, 0.5),
) -> dict[str, Any]:
    """Summarize paired original-minus-shuffled rubric score changes."""

    if target_rubric not in RUBRIC_KEYS:
        raise ValueError(f"unknown target rubric: {target_rubric}")
    rows = []
    missing = []
    for chain in chains:
        record_id = str(chain["record_id"])
        before = measurements.get((record_id, 0))
        after = measurements.get((record_id, 1))
        if before is None or after is None:
            missing.append(record_id)
            continue
        before_scores = _scores(before, score_basis)
        after_scores = _scores(after, score_basis)
        drops = {
            rubric: before_scores[rubric] - after_scores[rubric]
            for rubric in RUBRIC_KEYS
        }
        non_target = [value for key, value in drops.items() if key != target_rubric]
        rows.append(
            {
                "record_id": record_id,
                "source_record_id": str(chain["source_record_id"]),
                "level": str(chain["shuffle_level"]),
                "replicate": int(chain["replicate"]),
                "target_drop": drops[target_rubric],
                "max_non_target_drop": max(non_target),
                "target_specificity": drops[target_rubric] - max(non_target),
                "rubric_drops": drops,
            }
        )

    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[row["level"]].append(row)
    return {
        "score_basis": score_basis,
        "target_rubric": target_rubric,
        "pairs": len(rows),
        "source_essays": len({row["source_record_id"] for row in rows}),
        "missing_pairs": missing,
        "thresholds": list(thresholds),
        "by_level": {
            level: _level_summary(group, thresholds)
            for level, group in sorted(grouped.items())
        },
        "rows": rows,
    }


def _level_summary(
    rows: Sequence[Mapping[str, Any]],
    thresholds: Sequence[float],
) -> dict[str, Any]:
    drops = [float(row["target_drop"]) for row in rows]
    specificity = [float(row["target_specificity"]) for row in rows]
    return {
        "pairs": len(rows),
        "source_essays": len({str(row["source_record_id"]) for row in rows}),
        "target_drop": {
            "mean": statistics.mean(drops),
            "median": statistics.median(drops),
            "min": min(drops),
            "max": max(drops),
        },
        "positive_fraction": sum(value > 0 for value in drops) / len(drops),
        "above_threshold": {
            str(threshold): sum(value > threshold for value in drops)
            for threshold in thresholds
        },
        "target_specificity_median": statistics.median(specificity),
    }


def _scores(row: Mapping[str, Any], score_basis: str) -> dict[str, float]:
    if score_basis == "rf_corrected":
        values = row.get("rf_corrected")
        if not isinstance(values, list):
            raise ValueError("rf_corrected score basis requested but values are missing")
        return scores_to_rubric_dict(values)
    if score_basis == "integer":
        return {str(key): float(value) for key, value in row["rubrics"].items()}
    raise ValueError(f"unknown score basis: {score_basis}")


def _source_text(record: Mapping[str, Any]) -> str:
    if "text" in record:
        return " ".join(str(record["text"]).split())
    states = record.get("states")
    if isinstance(states, list) and states:
        return " ".join(str(states[0]).split())
    raise ValueError("source record requires text or non-empty states")


def _shuffle_until_changed(values: list[str], rng: random.Random) -> None:
    original = list(values)
    for _ in range(8):
        rng.shuffle(values)
        if values != original:
            return
    values.reverse()
