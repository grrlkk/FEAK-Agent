#!/usr/bin/env python
"""Measure the Kanana scorer noise floor.

For a few essays, scores each (1) twice verbatim (decoding determinism),
(2) with collapsed whitespace, and (3) with a meaning-preserving synonym swap.
The max per-rubric score change under these perturbations is the noise floor
that hard constraints like non_target_drop_max must stay above.

  python scripts/measure_scorer_noise.py --device-id 3 --n-essays 3
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.diagnose import get_diagnoser
from feak_tc.diagnose.constants import RUBRIC_KEYS
from feak_tc.mvp.batch import iter_text_records

_SYNONYM_SWAPS = [
    ("그러나", "하지만"),
    ("예를 들어", "예컨대"),
    ("매우", "아주"),
    ("가장", "제일"),
    ("때문에", "탓에"),
]


def _variant_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _variant_synonym(text: str) -> tuple[str, str]:
    for before, after in _SYNONYM_SWAPS:
        if before in text:
            return text.replace(before, after, 1), f"{before}->{after}"
    return text, "none"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/data_jsonl/train.jsonl")
    parser.add_argument("--n-essays", type=int, default=3)
    parser.add_argument("--min-chars", type=int, default=300)
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--kanana-m", type=int, default=1)
    parser.add_argument("--output", default="experiments/results/scorer_noise.json")
    args = parser.parse_args()

    diagnoser = get_diagnoser(
        "kanana",
        config_kwargs={
            "m": args.kanana_m,
            "generate_feedback": False,
            "device_id": args.device_id,
            "load_in_4bit": True,
        },
    )

    input_path = PROJECT_ROOT / args.input
    records = []
    for record in iter_text_records(input_path, limit=200, min_chars=args.min_chars):
        records.append(record)
        if len(records) >= args.n_essays:
            break

    report = []
    for record in records:
        question = record.metadata.get("question")
        if question:
            diagnoser.question = question
        base = record.text

        def score(text: str) -> dict[str, float]:
            return dict(diagnoser.diagnose(text).rubrics)

        run1 = score(base)
        run2 = score(base)
        ws_scores = score(_variant_whitespace(base))
        syn_text, swap = _variant_synonym(base)
        syn_scores = score(syn_text) if swap != "none" else None

        def diffs(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
            return {key: b[key] - a[key] for key in RUBRIC_KEYS if abs(b[key] - a[key]) > 1e-9}

        entry = {
            "record_id": record.record_id,
            "run1": run1,
            "rerun_diffs": diffs(run1, run2),
            "whitespace_diffs": diffs(run1, ws_scores),
            "synonym_swap": swap,
            "synonym_diffs": diffs(run1, syn_scores) if syn_scores else None,
        }
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    max_abs = {"rerun": 0.0, "whitespace": 0.0, "synonym": 0.0}
    for entry in report:
        for key, field in (("rerun", "rerun_diffs"), ("whitespace", "whitespace_diffs"), ("synonym", "synonym_diffs")):
            values = entry.get(field) or {}
            for delta in values.values():
                max_abs[key] = max(max_abs[key], abs(delta))
    summary = {"n_essays": len(report), "max_abs_diff": max_abs}
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "essays": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
