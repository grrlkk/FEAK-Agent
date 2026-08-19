#!/usr/bin/env python
"""Prepare blinded corruption A/B review rows without evaluating them."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.g2 import build_human_review_pairs


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--review-out", required=True)
    parser.add_argument("--key-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    parser.add_argument(
        "--operator",
        action="append",
        default=[],
        help="limit valid transitions to these corruption operators (repeatable)",
    )
    args = parser.parse_args()

    outputs = [Path(args.review_out), Path(args.key_out), Path(args.summary_out)]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing review outputs: {existing}")

    audit_rows = _read_jsonl(Path(args.audit))
    audit_rows = _filter_accepted_operators(audit_rows, set(args.operator))
    review, key, summary = build_human_review_pairs(
        _read_jsonl(Path(args.chains)),
        audit_rows,
        count=args.count,
        seed=args.seed,
    )
    _write_jsonl(outputs[0], review)
    _write_jsonl(outputs[1], key)
    outputs[2].write_text(
        json.dumps(
            {
                **summary,
                "status": "pending_human_review",
                "required_raters": 2,
                "agreement_threshold": 0.70,
                "operator_filter": sorted(args.operator),
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _filter_accepted_operators(
    rows: list[dict],
    operators: set[str],
) -> list[dict]:
    if not operators:
        return rows
    return [
        {
            **row,
            "accepted": bool(row.get("accepted"))
            and str(row.get("corruption_op")) in operators,
        }
        for row in rows
    ]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
