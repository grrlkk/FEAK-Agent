#!/usr/bin/env python
"""Seed shard outputs with verified source-state measurements from an earlier run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--reference-chains", required=True)
    parser.add_argument("--reference-measured", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument(
        "--all-states",
        action="store_true",
        help="Reuse every available state when both chain files describe the same states.",
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must satisfy 0 <= shard < num-shards")
    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")

    reference_chains = {
        str(row["record_id"]): row
        for row in _read_jsonl(Path(args.reference_chains))
    }
    reference_rows = {}
    for path in args.reference_measured:
        for row in _read_jsonl(Path(path)):
            state_index = int(row["state_index"])
            if args.all_states or state_index == 0:
                reference_rows[(str(row["record_id"]), state_index)] = row

    selected = [
        row
        for index, row in enumerate(_read_jsonl(Path(args.chains)))
        if index % args.num_shards == args.shard
    ]
    reused = []
    for chain in selected:
        record_id = str(chain["record_id"])
        reference = reference_chains.get(record_id)
        if reference is None or reference["states"][0] != chain["states"][0]:
            raise SystemExit(f"source text mismatch for {record_id}")
        if (record_id, 0) not in reference_rows:
            raise SystemExit(f"source measurement missing for {record_id}")
        if args.all_states:
            if list(reference["states"]) != list(chain["states"]):
                raise SystemExit(f"state text mismatch for {record_id}")
            reused.extend(
                reference_rows[(record_id, state_index)]
                for state_index in range(len(chain["states"]))
                if (record_id, state_index) in reference_rows
            )
        else:
            reused.append(reference_rows[(record_id, 0)])

    with output.open("w", encoding="utf-8") as file:
        for row in reused:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"seeded {len(reused)} verified source measurements -> {output}")
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
