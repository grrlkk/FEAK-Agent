#!/usr/bin/env python
"""Run or resume held-out evaluation for all validation-selected TVMs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--config", default="configs/tvm_stage1.yaml")
    args = parser.parse_args()

    manifest = Path(args.selection_manifest)
    selection = json.loads(manifest.read_text(encoding="utf-8"))
    if selection.get("gate") != "tvm_stage1_validation_selection" or bool(
        selection.get("test_metrics_read")
    ):
        raise SystemExit("invalid validation-only selection manifest")
    selected = list(selection.get("selected") or [])
    if not selected:
        raise SystemExit("selection manifest has no runs")
    for position, row in enumerate(selected, 1):
        run_dir = Path(row["run_dir"])
        report = run_dir / "test_report.json"
        predictions = run_dir / "test_predictions.jsonl"
        if report.exists() and predictions.exists():
            print(f"[{position}/{len(selected)}] skip completed {run_dir}", flush=True)
            continue
        if report.exists() or predictions.exists():
            raise SystemExit(f"partial test output exists: {run_dir}")
        print(f"[{position}/{len(selected)}] evaluate {run_dir}", flush=True)
        subprocess.run(
            [
                sys.executable,
                str(PROJECT_ROOT / "scripts" / "evaluate_tvm_stage1.py"),
                "--run-dir",
                str(run_dir),
                "--selection-manifest",
                str(manifest),
                "--config",
                args.config,
            ],
            cwd=PROJECT_ROOT,
            check=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
