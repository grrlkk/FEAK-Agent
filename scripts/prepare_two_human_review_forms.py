#!/usr/bin/env python
"""Create independently ordered blind review files for two human raters."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path


HIDDEN_FIELDS = {
    "expected_preference",
    "cleaner_stage",
    "corrupted_stage",
    "operators",
    "target_rubrics",
    "target_drops",
    "corruption_op",
    "target_rubric",
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review", required=True)
    parser.add_argument("--rater-one-out", required=True)
    parser.add_argument("--rater-two-out", required=True)
    parser.add_argument("--expected-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260729)
    args = parser.parse_args()

    outputs = [Path(args.rater_one_out), Path(args.rater_two_out)]
    existing = [str(path) for path in outputs if path.exists()]
    if existing:
        raise SystemExit(f"refusing to overwrite existing rater files: {existing}")

    source = _read_jsonl(Path(args.review))
    if len(source) != args.expected_count:
        raise SystemExit(
            f"expected {args.expected_count} review rows, found {len(source)}"
        )
    pair_ids = [str(row.get("pair_id", "")) for row in source]
    if len(set(pair_ids)) != len(pair_ids) or any(not pair_id for pair_id in pair_ids):
        raise SystemExit("review rows require unique non-empty pair_id values")
    leaked = sorted(
        field
        for row in source
        for field in HIDDEN_FIELDS
        if field in row
    )
    if leaked:
        raise SystemExit(f"review source leaks hidden fields: {sorted(set(leaked))}")

    for index, output in enumerate(outputs, 1):
        rows = [
            {
                **dict(row),
                "rater_id": f"R{index}",
                "preference": "",
                "notes": "",
            }
            for row in source
        ]
        random.Random(args.seed + index).shuffle(rows)
        _write_jsonl(output, rows)
    print(
        json.dumps(
            {
                "pairs_per_rater": len(source),
                "rater_one_out": str(outputs[0]),
                "rater_two_out": str(outputs[1]),
                "answer_fields": ["preference", "notes"],
                "allowed_preferences": ["A", "B", "TIE"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
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
