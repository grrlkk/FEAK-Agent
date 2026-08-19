#!/usr/bin/env python
"""Run STEP 3 essay-grouped LightGBM sanity on accepted adjacent transitions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.g2 import (
    TRANSITION_FEATURES,
    build_gbm_pairs,
    run_grouped_lightgbm_ranker,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--elite-stats", default="configs/elite_features.yaml")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--exclude-feature",
        action="append",
        choices=TRANSITION_FEATURES,
        default=[],
        help="Transition feature to exclude; may be repeated for ablations.",
    )
    args = parser.parse_args()

    audit_rows = _read_jsonl(Path(args.audit))
    with open(args.elite_stats, encoding="utf-8") as file:
        elite_payload = yaml.safe_load(file) or {}
    elite_stats = elite_payload.get("features", elite_payload)

    pairs = build_gbm_pairs(audit_rows, elite_stats)
    selected_features = tuple(
        feature
        for feature in TRANSITION_FEATURES
        if feature not in set(args.exclude_feature)
    )
    report, predictions = run_grouped_lightgbm_ranker(
        pairs,
        seed=args.seed,
        transition_features=selected_features,
    )
    _write_jsonl(Path(args.predictions_out), predictions)
    Path(args.report_out).write_text(
        json.dumps({"gate": "STEP3_GBM_SANITY", "gbm": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
