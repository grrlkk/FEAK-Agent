#!/usr/bin/env python
"""Generate corruption chains (LLM phase, no GPU) from a source pool."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption import generate_chain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="source pool jsonl (record_id, question, text)")
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--append", action="store_true", help="resume: skip record_ids already in output")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not (args.append or args.overwrite):
        raise SystemExit(f"{output} exists; pass --append or --overwrite.")
    done = set()
    if args.append and output.exists():
        with output.open() as f:
            done = {json.loads(line)["record_id"] for line in f}

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    records = []
    with open(args.input, encoding="utf-8") as f:
        for line in f:
            row = _source_record(json.loads(line))
            if row["record_id"] not in done:
                records.append(row)
    if args.limit:
        records = records[: args.limit]
    if args.num_shards < 1 or not 0 <= args.shard < args.num_shards:
        raise SystemExit("--shard must satisfy 0 <= shard < num-shards")
    records = [
        record
        for index, record in enumerate(records)
        if index % args.num_shards == args.shard
    ]

    statuses: Counter = Counter()
    mode = "a" if args.append and output.exists() else "w"
    with output.open(mode, encoding="utf-8") as out:
        for i, record in enumerate(records, 1):
            chain = generate_chain(record, cfg)
            statuses[chain["status"]] += 1
            out.write(json.dumps(chain, ensure_ascii=False) + "\n")
            out.flush()
            print(f"[shard {args.shard} {i}/{len(records)}] "
                  f"{record['record_id']}: {chain['status']} "
                  f"({', '.join(s['operator'] for s in chain['steps'])})",
                  flush=True)

    print(f"\ndone: {dict(statuses)} -> {output}", flush=True)
    return 0


def _source_record(row: dict) -> dict:
    """Reuse the exact source essays from a prior chain file when requested."""

    if "text" in row:
        return row
    states = row.get("states")
    if isinstance(states, list) and states:
        return {
            "record_id": row["record_id"],
            "question": row.get("question"),
            "grader_avg": row.get("grader_avg"),
            "text": states[0],
        }
    raise ValueError("source row requires text or non-empty states")


if __name__ == "__main__":
    raise SystemExit(main())
