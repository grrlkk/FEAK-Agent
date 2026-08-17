#!/usr/bin/env python
"""Evaluate two completed blind human reviews and unresolved disagreements."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.g2 import evaluate_two_human_reviews


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rater-one", required=True)
    parser.add_argument("--rater-two", required=True)
    parser.add_argument("--key", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--disagreements-out", required=True)
    parser.add_argument("--adjudication")
    parser.add_argument("--threshold", type=float, default=0.70)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    outputs = [Path(args.report_out), Path(args.disagreements_out)]
    existing = [str(path) for path in outputs if path.exists()]
    if existing and not args.overwrite:
        raise SystemExit(f"refusing to overwrite existing outputs: {existing}")

    adjudication = (
        _read_jsonl(Path(args.adjudication)) if args.adjudication else []
    )
    report, disagreements = evaluate_two_human_reviews(
        _read_jsonl(Path(args.rater_one)),
        _read_jsonl(Path(args.rater_two)),
        _read_jsonl(Path(args.key)),
        adjudication_rows=adjudication,
        threshold=args.threshold,
    )
    outputs[0].write_text(
        json.dumps(
            {"gate": "G2_HUMAN_BLIND_REVIEW", **report},
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    _write_jsonl(outputs[1], disagreements)
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
