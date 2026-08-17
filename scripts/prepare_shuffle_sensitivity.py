#!/usr/bin/env python
"""Create paired sentence-order perturbations for scorer sensitivity tests."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.shuffle_sensitivity import (
    SHUFFLE_LEVELS,
    build_shuffle_sensitivity_chains,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument(
        "--level",
        action="append",
        choices=SHUFFLE_LEVELS,
        help="May be repeated; defaults to all levels.",
    )
    parser.add_argument("--seed", type=int, default=20260812)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    records = _read_jsonl(Path(args.input))[: args.limit]
    chains = build_shuffle_sensitivity_chains(
        records,
        levels=args.level or SHUFFLE_LEVELS,
        replicates=args.replicates,
        seed=args.seed,
    )
    with output.open("w", encoding="utf-8") as file:
        for chain in chains:
            file.write(json.dumps(chain, ensure_ascii=False) + "\n")
    print(f"prepared {len(chains)} shuffle sensitivity pairs -> {output}")
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
