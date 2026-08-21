"""Paired held-out comparisons for TVM and fixed-split baselines."""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np


def paired_accuracy_comparison(
    left: Sequence[Mapping[str, Any]],
    right: Sequence[Mapping[str, Any]],
    *,
    left_name: str,
    right_name: str,
    seed: int = 20260821,
    bootstrap_samples: int = 10000,
) -> dict[str, Any]:
    """Compare correctness on identical pairs with paired bootstrap and McNemar."""

    left_by_id = {str(row["pair_id"]): bool(row["correct"]) for row in left}
    right_by_id = {str(row["pair_id"]): bool(row["correct"]) for row in right}
    if len(left_by_id) != len(left) or len(right_by_id) != len(right):
        raise ValueError("paired comparison contains duplicate pair IDs")
    if set(left_by_id) != set(right_by_id):
        raise ValueError("paired comparison requires identical pair IDs")
    identifiers = sorted(left_by_id)
    if not identifiers:
        raise ValueError("paired comparison requires at least one pair")
    left_values = np.asarray([left_by_id[key] for key in identifiers], dtype=np.float32)
    right_values = np.asarray([right_by_id[key] for key in identifiers], dtype=np.float32)
    differences = left_values - right_values
    rng = np.random.default_rng(seed)
    samples = rng.integers(
        0,
        len(identifiers),
        size=(int(bootstrap_samples), len(identifiers)),
    )
    bootstrap = differences[samples].mean(axis=1)
    left_only = int(np.sum((left_values == 1) & (right_values == 0)))
    right_only = int(np.sum((left_values == 0) & (right_values == 1)))
    return {
        "left": left_name,
        "right": right_name,
        "pairs": len(identifiers),
        "left_accuracy": float(left_values.mean()),
        "right_accuracy": float(right_values.mean()),
        "accuracy_difference": float(differences.mean()),
        "paired_bootstrap_95": {
            "low": float(np.percentile(bootstrap, 2.5)),
            "high": float(np.percentile(bootstrap, 97.5)),
            "samples": int(bootstrap_samples),
            "seed": int(seed),
        },
        "discordant": {"left_only_correct": left_only, "right_only_correct": right_only},
        "mcnemar_exact_two_sided_p": _exact_binomial_two_sided(left_only, right_only),
    }


def _exact_binomial_two_sided(left_only: int, right_only: int) -> float:
    discordant = int(left_only) + int(right_only)
    if discordant == 0:
        return 1.0
    tail = min(int(left_only), int(right_only))
    probability = sum(math.comb(discordant, value) for value in range(tail + 1))
    return min(1.0, 2.0 * probability / (2.0**discordant))
