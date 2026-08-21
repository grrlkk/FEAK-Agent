"""Validation-only model selection for TVM Stage-1 runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence


def select_best_runs(
    reports: Sequence[Mapping[str, Any]],
    *,
    expected_learning_rates: Sequence[float] | None = None,
) -> list[dict[str, Any]]:
    """Select one LR per model/feature condition without reading test metrics."""

    if not reports:
        raise ValueError("TVM selection requires at least one validation report")
    grouped: dict[tuple[str, str], list[Mapping[str, Any]]] = defaultdict(list)
    for report in reports:
        if report.get("gate") != "tvm_stage1_validation":
            raise ValueError("selection accepts TVM validation reports only")
        if bool(report.get("test_split_evaluated", report.get("test_split_read"))):
            raise ValueError("validation selection cannot consume a test-evaluated report")
        grouped[(str(report["model_key"]), str(report["feature_variant"]))].append(
            report
        )

    expected = (
        {float(value) for value in expected_learning_rates}
        if expected_learning_rates is not None
        else None
    )
    selected = []
    for (model_key, feature_variant), candidates in sorted(grouped.items()):
        rates = [float(candidate["learning_rate"]) for candidate in candidates]
        if len(rates) != len(set(rates)):
            raise ValueError(f"duplicate learning rate for {model_key}/{feature_variant}")
        if expected is not None and set(rates) != expected:
            raise ValueError(
                f"incomplete LR sweep for {model_key}/{feature_variant}: {sorted(rates)}"
            )
        signatures = {
            (
                str(candidate["data"]["sha256"]),
                str(candidate["data"]["prompt_sha256"]),
                int(candidate["seed"]),
                str(candidate["split"]["digest"]),
            )
            for candidate in candidates
        }
        if len(signatures) != 1:
            raise ValueError(
                f"incomparable data or split for {model_key}/{feature_variant}"
            )
        winner = min(
            candidates,
            key=lambda candidate: (
                -float(candidate["validation"]["pairwise_accuracy"]),
                float(candidate["validation"]["mean_loss"]),
                float(candidate["learning_rate"]),
            ),
        )
        selected.append(
            {
                "model_key": model_key,
                "model_role": str(winner["model_role"]),
                "feature_variant": feature_variant,
                "learning_rate": float(winner["learning_rate"]),
                "validation": dict(winner["validation"]),
                "run_dir": str(winner["run_dir"]),
                "data_sha256": str(winner["data"]["sha256"]),
                "prompt_sha256": str(winner["data"]["prompt_sha256"]),
                "split_digest": str(winner["split"]["digest"]),
                "selection_rule": (
                    "max validation pairwise_accuracy; then min validation mean_loss; "
                    "then min learning_rate"
                ),
            }
        )
    return selected
