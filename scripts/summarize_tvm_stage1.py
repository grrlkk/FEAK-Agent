#!/usr/bin/env python
"""Aggregate selected TVM runs, baselines, and paired synthetic-test comparisons."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.tvm.comparison import paired_accuracy_comparison


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--baseline-report", required=True)
    parser.add_argument("--baseline-predictions", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    selection = _read_json(Path(args.selection_manifest))
    baseline_report = _read_json(Path(args.baseline_report))
    baseline_rows = _read_jsonl(Path(args.baseline_predictions))
    if selection.get("gate") != "tvm_stage1_validation_selection":
        raise SystemExit("invalid TVM selection manifest")
    if baseline_report.get("gate") != "tvm_stage1_fixed_split_baselines":
        raise SystemExit("invalid fixed-split baseline report")

    predictions: dict[str, list[dict[str, Any]]] = defaultdict(list)
    results = {}
    for selected in selection["selected"]:
        name = f"tvm_{selected['model_key']}_{selected['feature_variant']}"
        run_dir = Path(selected["run_dir"])
        test_report = _read_json(run_dir / "test_report.json")
        test_predictions = _read_jsonl(run_dir / "test_predictions.jsonl")
        if test_report.get("gate") != "tvm_stage1_synthetic_test":
            raise SystemExit(f"missing synthetic test report for {name}")
        predictions[name] = test_predictions
        results[name] = {
            "model_role": selected["model_role"],
            "feature_variant": selected["feature_variant"],
            "learning_rate": selected["learning_rate"],
            "validation": test_report["validation"],
            "test": test_report["test"],
            "run_dir": str(run_dir.resolve()),
        }
    for row in baseline_rows:
        if row.get("split") == "test":
            predictions[str(row["model"])].append(row)
    results.update(baseline_report["evaluation"])

    comparisons = []
    for variant in ("full", "scorer_free"):
        qwen = f"tvm_qwen_{variant}"
        kanana = f"tvm_kanana_{variant}"
        comparisons.append(_compare(predictions, qwen, kanana))
        for tvm in (qwen, kanana):
            for baseline in (
                f"feature_lightgbm_{variant}",
                f"heuristic_{variant}",
                "bge_m3_action_linear",
                "immediate_target_gain",
            ):
                comparisons.append(_compare(predictions, tvm, baseline))
    comparisons.extend(
        [
            _compare(predictions, "tvm_qwen_full", "tvm_qwen_scorer_free"),
            _compare(predictions, "tvm_kanana_full", "tvm_kanana_scorer_free"),
        ]
    )
    payload = {
        "gate": "tvm_stage1_synthetic_summary",
        "main_model": "tvm_qwen_scorer_free",
        "independence_control": "tvm_kanana_scorer_free",
        "results": results,
        "paired_test_comparisons": comparisons,
        "limitations": [
            "all metrics in this report use synthetic corruption preferences",
            "human blind preference remains the decisive external validation",
            "scorer_free removes Kanana score outputs from TVM inputs but labels were still filtered with the corruption pipeline",
        ],
    }
    Path(args.output).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _compare(
    predictions: Mapping[str, Sequence[Mapping[str, Any]]],
    left: str,
    right: str,
) -> dict[str, Any]:
    if left not in predictions or right not in predictions:
        raise SystemExit(f"missing predictions for paired comparison: {left}, {right}")
    return paired_accuracy_comparison(
        predictions[left],
        predictions[right],
        left_name=left,
        right_name=right,
    )


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
