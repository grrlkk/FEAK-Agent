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
from feak_tc.diagnose.constants import RUBRIC_KEYS, scores_to_rubric_dict
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

        def score(text: str) -> dict[str, dict[str, float]]:
            diag = diagnoser.diagnose(text)
            return {
                "integer": dict(diag.rubrics),
                "rf_corrected": _metadata_scores(diag.metadata.get("rf_corrected_score")),
                "soft_mean": _metadata_scores(diag.metadata.get("soft_mean")),
            }

        run1 = score(base)
        run2 = score(base)
        ws_scores = score(_variant_whitespace(base))
        syn_text, swap = _variant_synonym(base)
        syn_scores = score(syn_text) if swap != "none" else None

        def diffs(a: dict[str, float], b: dict[str, float]) -> dict[str, float]:
            return {key: b[key] - a[key] for key in RUBRIC_KEYS if abs(b[key] - a[key]) > 1e-9}

        def basis_diffs(
            a: dict[str, dict[str, float]],
            b: dict[str, dict[str, float]],
            basis: str,
        ) -> dict[str, float]:
            if not a[basis] or not b[basis]:
                return {}
            return diffs(a[basis], b[basis])

        entry = {
            "record_id": record.record_id,
            "run1": run1["integer"],
            "run1_rf_corrected": run1["rf_corrected"],
            "run1_soft_mean": run1["soft_mean"],
            "rerun_diffs": basis_diffs(run1, run2, "integer"),
            "rerun_diffs_rf_corrected": basis_diffs(run1, run2, "rf_corrected"),
            "rerun_diffs_soft_mean": basis_diffs(run1, run2, "soft_mean"),
            "whitespace_diffs": basis_diffs(run1, ws_scores, "integer"),
            "whitespace_diffs_rf_corrected": basis_diffs(run1, ws_scores, "rf_corrected"),
            "whitespace_diffs_soft_mean": basis_diffs(run1, ws_scores, "soft_mean"),
            "synonym_swap": swap,
            "synonym_diffs": basis_diffs(run1, syn_scores, "integer") if syn_scores else None,
            "synonym_diffs_rf_corrected": basis_diffs(run1, syn_scores, "rf_corrected") if syn_scores else None,
            "synonym_diffs_soft_mean": basis_diffs(run1, syn_scores, "soft_mean") if syn_scores else None,
        }
        report.append(entry)
        print(json.dumps(entry, ensure_ascii=False))

    max_abs_integer = {"rerun": 0.0, "whitespace": 0.0, "synonym": 0.0}
    max_abs_continuous = {"rerun": 0.0, "whitespace": 0.0, "synonym": 0.0}
    max_abs_soft_mean = {"rerun": 0.0, "whitespace": 0.0, "synonym": 0.0}
    totals_integer = _diff_totals()
    totals_continuous = _diff_totals()
    totals_soft_mean = _diff_totals()
    for entry in report:
        for key, field in (("rerun", "rerun_diffs"), ("whitespace", "whitespace_diffs"), ("synonym", "synonym_diffs")):
            _accumulate_if_present(entry.get(field), key, max_abs_integer, totals_integer)
        for key, field in (
            ("rerun", "rerun_diffs_rf_corrected"),
            ("whitespace", "whitespace_diffs_rf_corrected"),
            ("synonym", "synonym_diffs_rf_corrected"),
        ):
            _accumulate_if_present(entry.get(field), key, max_abs_continuous, totals_continuous)
        for key, field in (
            ("rerun", "rerun_diffs_soft_mean"),
            ("whitespace", "whitespace_diffs_soft_mean"),
            ("synonym", "synonym_diffs_soft_mean"),
        ):
            _accumulate_if_present(entry.get(field), key, max_abs_soft_mean, totals_soft_mean)
    summary = {
        "n_essays": len(report),
        "continuous_basis": "rf_corrected_score",
        "max_abs_diff": max_abs_integer,
        "max_abs_diff_integer": max_abs_integer,
        "mean_abs_diff_integer": _mean_abs(totals_integer),
        "max_abs_diff_continuous": max_abs_continuous,
        "mean_abs_diff_continuous": _mean_abs(totals_continuous),
        "max_abs_diff_soft_mean": max_abs_soft_mean,
        "mean_abs_diff_soft_mean": _mean_abs(totals_soft_mean),
    }
    print("SUMMARY " + json.dumps(summary, ensure_ascii=False))

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps({"summary": summary, "essays": report}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"wrote {output_path}")
    return 0

def _metadata_scores(raw: object) -> dict[str, float]:
    if raw is None:
        return {}
    try:
        return scores_to_rubric_dict(raw)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return {}


def _diff_totals() -> dict[str, dict[str, float]]:
    return {
        "rerun": {"sum": 0.0, "count": 0.0},
        "whitespace": {"sum": 0.0, "count": 0.0},
        "synonym": {"sum": 0.0, "count": 0.0},
    }


def _accumulate(
    values: dict[str, float],
    key: str,
    max_abs: dict[str, float],
    totals: dict[str, dict[str, float]],
) -> None:
    for rubric in RUBRIC_KEYS:
        delta = values.get(rubric, 0.0)
        abs_delta = abs(delta)
        max_abs[key] = max(max_abs[key], abs_delta)
        totals[key]["sum"] += abs_delta
        totals[key]["count"] += 1.0


def _accumulate_if_present(
    values: object,
    key: str,
    max_abs: dict[str, float],
    totals: dict[str, dict[str, float]],
) -> None:
    if values is None:
        return
    _accumulate(dict(values), key, max_abs, totals)  # type: ignore[arg-type]


def _mean_abs(totals: dict[str, dict[str, float]]) -> dict[str, float]:
    means = {}
    for key, values in totals.items():
        count = values["count"]
        means[key] = values["sum"] / count if count else 0.0
    return means


if __name__ == "__main__":
    raise SystemExit(main())
