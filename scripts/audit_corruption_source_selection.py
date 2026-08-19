#!/usr/bin/env python
"""Audit source-selection bias introduced by keeping complete corruption chains."""

from __future__ import annotations

import argparse
import glob
import json
import math
import re
import statistics
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping


_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+|\n+")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, help="ordered source-pool JSONL")
    parser.add_argument(
        "--attempts",
        required=True,
        nargs="+",
        help="attempt JSONL paths or glob patterns",
    )
    parser.add_argument("--selected", required=True, help="selected complete-chain JSONL")
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    source_rows = _read_jsonl_paths([args.source])
    attempt_paths = _expand_paths(args.attempts)
    attempt_rows = _read_jsonl_paths(attempt_paths)
    selected_rows = _read_jsonl_paths([args.selected])
    report = build_selection_bias_report(source_rows, attempt_rows, selected_rows)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def build_selection_bias_report(
    source_rows: list[Mapping[str, Any]],
    attempt_rows: list[Mapping[str, Any]],
    selected_rows: list[Mapping[str, Any]],
) -> dict[str, Any]:
    source_by_id = _unique_by_record_id(source_rows, "source")
    attempts_by_id = _unique_by_record_id(attempt_rows, "attempts")
    selected_by_id = _unique_by_record_id(selected_rows, "selected")

    unknown_attempts = sorted(set(attempts_by_id) - set(source_by_id))
    unknown_selected = sorted(set(selected_by_id) - set(attempts_by_id))
    if unknown_attempts:
        raise ValueError(f"attempt record_ids missing from source: {unknown_attempts}")
    if unknown_selected:
        raise ValueError(f"selected record_ids missing from attempts: {unknown_selected}")
    non_ok_selected = sorted(
        record_id
        for record_id in selected_by_id
        if attempts_by_id[record_id].get("status") != "ok"
    )
    if non_ok_selected:
        raise ValueError(f"selected records are not complete chains: {non_ok_selected}")

    attempted_ids = list(attempts_by_id)
    selected_ids = set(selected_by_id)
    ok_ids = {
        record_id
        for record_id, row in attempts_by_id.items()
        if row.get("status") == "ok"
    }
    cohorts = {
        "attempted": attempted_ids,
        "selected": [record_id for record_id in attempted_ids if record_id in selected_ids],
        "not_selected": [record_id for record_id in attempted_ids if record_id not in selected_ids],
        "generation_ok": [record_id for record_id in attempted_ids if record_id in ok_ids],
        "generation_incomplete": [record_id for record_id in attempted_ids if record_id not in ok_ids],
    }
    summaries = {
        name: _cohort_summary(ids, source_by_id, attempts_by_id)
        for name, ids in cohorts.items()
    }
    comparisons = {
        "selected_vs_not_selected": _compare_cohorts(
            cohorts["selected"], cohorts["not_selected"], source_by_id, attempts_by_id
        ),
        "generation_ok_vs_incomplete": _compare_cohorts(
            cohorts["generation_ok"],
            cohorts["generation_incomplete"],
            source_by_id,
            attempts_by_id,
        ),
    }

    numeric_effects = [
        abs(float(metric["standardized_mean_difference"]))
        for comparison in comparisons.values()
        for metric in comparison["numeric"].values()
        if metric["standardized_mean_difference"] is not None
    ]
    max_effect = max(numeric_effects, default=0.0)
    if max_effect >= 0.5:
        severity = "notable"
        recommendation = (
            "Do not treat the selected pool as source-representative; add a stratified "
            "sampling or partial-chain analysis before final-scale production."
        )
    elif max_effect >= 0.2:
        severity = "moderate"
        recommendation = (
            "Proceed with the staged expansion without relaxing guards, retain every "
            "attempt, and repeat this audit on the cumulative 200-essay pool."
        )
    else:
        severity = "small"
        recommendation = "Proceed and repeat the audit at the next staged checkpoint."

    failed_operator_counts: Counter[str] = Counter()
    failed_stage_counts: Counter[int] = Counter()
    for row in attempt_rows:
        planned = list(row.get("planned_operators") or [])
        completed = len(row.get("steps") or [])
        if row.get("status") == "ok":
            continue
        failed_stage_counts[completed + 1] += 1
        if completed < len(planned):
            failed_operator_counts[str(planned[completed])] += 1

    all_questions_unique = (
        len({str(source_by_id[record_id].get("question") or "") for record_id in attempted_ids})
        == len(attempted_ids)
    )
    return {
        "audit": "corruption_source_selection_bias",
        "counts": {
            "source_pool": len(source_rows),
            "attempted": len(attempt_rows),
            "selected": len(selected_rows),
            "generation_ok": len(ok_ids),
            "generation_incomplete": len(attempt_rows) - len(ok_ids),
            "attempt_status": dict(
                sorted(Counter(str(row.get("status")) for row in attempt_rows).items())
            ),
        },
        "cohorts": summaries,
        "comparisons": comparisons,
        "failure_profile": {
            "failed_operator": dict(sorted(failed_operator_counts.items())),
            "failed_stage_1_based": {
                str(key): value for key, value in sorted(failed_stage_counts.items())
            },
        },
        "question_distribution_note": {
            "all_attempted_questions_unique": all_questions_unique,
            "exact_question_frequency_comparison_informative": not all_questions_unique,
            "proxy_used": "question_char_count",
        },
        "assessment": {
            "heuristic_not_a_preregistered_gate": True,
            "smd_interpretation": {"small_below": 0.2, "notable_at_or_above": 0.5},
            "max_absolute_numeric_smd": max_effect,
            "severity": severity,
            "recommendation": recommendation,
        },
    }


def _cohort_summary(
    record_ids: Iterable[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
    attempts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    ids = list(record_ids)
    metrics = _numeric_metrics(ids, source_by_id)
    first_operators = [
        str((attempts_by_id[record_id].get("planned_operators") or ["NONE"])[0])
        for record_id in ids
    ]
    questions = [str(source_by_id[record_id].get("question") or "") for record_id in ids]
    return {
        "n": len(ids),
        "numeric": {name: _numeric_summary(values) for name, values in metrics.items()},
        "unique_questions": len(set(questions)),
        "planned_first_operator": dict(sorted(Counter(first_operators).items())),
    }


def _compare_cohorts(
    left_ids: list[str],
    right_ids: list[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
    attempts_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    left_metrics = _numeric_metrics(left_ids, source_by_id)
    right_metrics = _numeric_metrics(right_ids, source_by_id)
    left_operators = Counter(
        str((attempts_by_id[record_id].get("planned_operators") or ["NONE"])[0])
        for record_id in left_ids
    )
    right_operators = Counter(
        str((attempts_by_id[record_id].get("planned_operators") or ["NONE"])[0])
        for record_id in right_ids
    )
    left_questions = Counter(
        str(source_by_id[record_id].get("question") or "")
        for record_id in left_ids
    )
    right_questions = Counter(
        str(source_by_id[record_id].get("question") or "")
        for record_id in right_ids
    )
    return {
        "left_n": len(left_ids),
        "right_n": len(right_ids),
        "numeric": {
            name: {
                "left_mean": statistics.mean(left_metrics[name]) if left_metrics[name] else None,
                "right_mean": statistics.mean(right_metrics[name]) if right_metrics[name] else None,
                "standardized_mean_difference": _standardized_mean_difference(
                    left_metrics[name], right_metrics[name]
                ),
            }
            for name in left_metrics
        },
        "planned_first_operator_total_variation": _total_variation(
            left_operators, right_operators
        ),
        "exact_question_distribution": {
            "left_unique": len(left_questions),
            "right_unique": len(right_questions),
            "shared_questions": len(set(left_questions) & set(right_questions)),
            "total_variation": _total_variation(left_questions, right_questions),
            "sparse_category_caution": True,
        },
    }


def _numeric_metrics(
    record_ids: Iterable[str],
    source_by_id: Mapping[str, Mapping[str, Any]],
) -> dict[str, list[float]]:
    rows = [source_by_id[record_id] for record_id in record_ids]
    return {
        "text_char_count": [float(len(str(row.get("text") or ""))) for row in rows],
        "sentence_count": [float(_sentence_count(str(row.get("text") or ""))) for row in rows],
        "grader_avg": [float(row["grader_avg"]) for row in rows],
        "question_char_count": [
            float(len(str(row.get("question") or ""))) for row in rows
        ],
    }


def _sentence_count(text: str) -> int:
    parts = [part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()]
    content_parts = [part for part in parts if not part.startswith("핵심 키워드:")]
    return max(1, len(content_parts)) if text.strip() else 0


def _numeric_summary(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"mean": None, "median": None, "stddev": None, "min": None, "max": None}
    return {
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stddev": statistics.stdev(values) if len(values) >= 2 else 0.0,
        "min": min(values),
        "max": max(values),
    }


def _standardized_mean_difference(left: list[float], right: list[float]) -> float | None:
    if not left or not right:
        return None
    left_variance = statistics.variance(left) if len(left) >= 2 else 0.0
    right_variance = statistics.variance(right) if len(right) >= 2 else 0.0
    denominator = len(left) + len(right) - 2
    pooled_variance = (
        ((len(left) - 1) * left_variance + (len(right) - 1) * right_variance)
        / denominator
        if denominator > 0
        else 0.0
    )
    if pooled_variance == 0.0:
        return 0.0 if statistics.mean(left) == statistics.mean(right) else None
    return (statistics.mean(left) - statistics.mean(right)) / math.sqrt(pooled_variance)


def _total_variation(left: Counter[str], right: Counter[str]) -> float | None:
    left_total = sum(left.values())
    right_total = sum(right.values())
    if not left_total or not right_total:
        return None
    categories = set(left) | set(right)
    return 0.5 * sum(
        abs(left[key] / left_total - right[key] / right_total)
        for key in categories
    )


def _unique_by_record_id(
    rows: Iterable[Mapping[str, Any]], label: str
) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for row in rows:
        record_id = str(row["record_id"])
        if record_id in result:
            raise ValueError(f"duplicate record_id in {label}: {record_id}")
        result[record_id] = row
    return result


def _expand_paths(patterns: Iterable[str]) -> list[str]:
    paths: list[str] = []
    for pattern in patterns:
        matches = sorted(glob.glob(pattern))
        paths.extend(matches or [pattern])
    return paths


def _read_jsonl_paths(paths: Iterable[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        with open(path, encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    return rows


if __name__ == "__main__":
    raise SystemExit(main())
