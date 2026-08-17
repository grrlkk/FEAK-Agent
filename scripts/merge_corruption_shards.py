#!/usr/bin/env python
"""Merge disjoint corruption JSONL shards and restore source-record order."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--shards", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite.")

    source_order = {}
    with open(args.source, encoding="utf-8") as file:
        for index, line in enumerate(file):
            source_order[str(json.loads(line)["record_id"])] = index

    rows = []
    for shard in args.shards:
        with open(shard, encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    record_ids = [str(row["record_id"]) for row in rows]
    if len(record_ids) != len(set(record_ids)):
        raise SystemExit("duplicate record_id across shards")
    missing = [record_id for record_id in record_ids if record_id not in source_order]
    if missing:
        raise SystemExit(f"shard record_ids missing from source: {missing}")
    rows.sort(key=lambda row: source_order[str(row["record_id"])])

    with output.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"merged {len(rows)} records -> {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
