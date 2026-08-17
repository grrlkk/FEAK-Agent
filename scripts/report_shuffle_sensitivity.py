#!/usr/bin/env python
"""Report paired organization_1 sensitivity to sentence-order perturbations."""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.shuffle_sensitivity import summarize_shuffle_measurements


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--measured", required=True, nargs="+")
    parser.add_argument("--output", required=True)
    parser.add_argument("--score-basis", default="rf_corrected")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    chains = _read_jsonl(Path(args.chains))
    measurements = {}
    for pattern in args.measured:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            for row in _read_jsonl(Path(path)):
                measurements[(str(row["record_id"]), int(row["state_index"]))] = row
    report = summarize_shuffle_measurements(
        chains,
        measurements,
        score_basis=args.score_basis,
    )
    output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps({key: value for key, value in report.items() if key != "rows"}, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


if __name__ == "__main__":
    raise SystemExit(main())
