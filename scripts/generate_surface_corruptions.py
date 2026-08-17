#!/usr/bin/env python
"""Generate small chain-external grammar samples for surface API validation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption import generate_surface_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite.")

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    limit = args.limit
    if limit is None:
        limit = int(cfg.get("surface_validation", {}).get("sample_count", 10))

    samples = []
    with open(args.input, encoding="utf-8") as file:
        for line in file:
            source = _source_record(json.loads(line))
            try:
                samples.append(generate_surface_sample(source, cfg))
            except ValueError as exc:
                print(f"skip {source['record_id']}: {exc}")
                continue
            if len(samples) >= limit:
                break

    with output.open("w", encoding="utf-8") as file:
        for sample in samples:
            file.write(json.dumps(sample, ensure_ascii=False) + "\n")
    print(f"generated {len(samples)} surface samples -> {output}")
    return 0


def _source_record(row: dict) -> dict:
    if "text" in row:
        return row
    states = row.get("states")
    if isinstance(states, list) and states:
        return {
            "record_id": row["record_id"],
            "question": row.get("question"),
            "text": states[0],
        }
    raise ValueError("source row requires text or non-empty states")


if __name__ == "__main__":
    raise SystemExit(main())
