#!/usr/bin/env python
"""Select one Stage-1 learning rate per condition using validation only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.tvm.selection import select_best_runs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--config", default="configs/tvm_stage1.yaml")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    report_paths = sorted(Path(args.runs_root).glob("**/validation_report.json"))
    if not report_paths:
        raise SystemExit(f"no validation reports under {args.runs_root}")
    reports = []
    for path in report_paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        report["run_dir"] = str(path.parent.resolve())
        reports.append(report)
    selected = select_best_runs(
        reports,
        expected_learning_rates=config["training"]["learning_rates"],
    )
    expected_conditions = {
        (model_key, variant)
        for model_key in config["models"]
        for variant in config["feature_variants"]
    }
    actual_conditions = {
        (row["model_key"], row["feature_variant"]) for row in selected
    }
    if actual_conditions != expected_conditions:
        raise SystemExit(
            f"condition mismatch: expected={sorted(expected_conditions)}, "
            f"actual={sorted(actual_conditions)}"
        )
    payload = {
        "gate": "tvm_stage1_validation_selection",
        "test_metrics_read": False,
        "runs_root": str(Path(args.runs_root).resolve()),
        "selected": selected,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
