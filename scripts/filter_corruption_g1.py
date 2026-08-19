#!/usr/bin/env python
"""Join G1 chains with diagnoses and keep strict target-rubric drops."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from itertools import product
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.measure import evaluate_chain
from feak_tc.corruption.quality import (
    audit_edit_artifacts,
    audit_operator_balance,
    audit_semantic_edit_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--measured", required=True, nargs="+")
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--accepted-out", required=True)
    parser.add_argument("--summary-out", required=True)
    parser.add_argument(
        "--quality-report-out",
        help="Defaults to <summary-out stem>_quality.json.",
    )
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    with open(cfg["schema"], encoding="utf-8") as file:
        schema = yaml.safe_load(file)

    measurements: dict[tuple[str, int], dict] = {}
    for pattern in args.measured:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as file:
                for line in file:
                    row = json.loads(line)
                    measurements[(row["record_id"], int(row["state_index"]))] = row

    chains = []
    with open(args.chains, encoding="utf-8") as file:
        chains = [json.loads(line) for line in file]

    audit_rows: list[dict] = []
    for chain in chains:
        audit_rows.extend(
            evaluate_chain(chain, measurements, schema, cfg["measurement"])
        )
    accepted_rows, artifact_report, artifact_quarantine = _quarantine_artifacts(
        audit_rows,
        cfg.get("artifact_audit", {}),
    )
    semantic_artifact_report = audit_semantic_edit_artifacts(
        accepted_rows,
        cfg.get("semantic_artifact_audit", {}),
    )
    balance_report = audit_operator_balance(
        accepted_rows,
        cfg.get("balance", {}),
    )

    _write_jsonl(Path(args.audit_out), audit_rows)
    _write_jsonl(Path(args.accepted_out), accepted_rows)

    by_operator: dict[str, Counter] = defaultdict(Counter)
    by_generator: dict[str, Counter] = defaultdict(Counter)
    by_requested_generator: dict[str, Counter] = defaultdict(Counter)
    for row in audit_rows:
        outcome = "accepted" if row["accepted"] else row["acceptance_reason"]
        by_operator[row["corruption_op"]][outcome] += 1
        by_generator[row["generator"]][outcome] += 1
        by_requested_generator[row["requested_generator"]][outcome] += 1
    summary = {
        "essays": len(chains),
        "chain_statuses": dict(Counter(chain["status"] for chain in chains)),
        "states": sum(len(chain["states"]) for chain in chains),
        "normalized_states": sum(
            bool(item.get("normalized"))
            for chain in chains
            for item in chain["normalizations"]
        ),
        "generated_steps": len(audit_rows),
        "accepted_steps": len(accepted_rows),
        "acceptance_rate": len(accepted_rows) / len(audit_rows) if audit_rows else 0.0,
        "accepted_essays": len({row["essay_id"] for row in accepted_rows}),
        "accepted_steps_per_essay": dict(
            sorted(
                Counter(
                    sum(row["accepted"] for row in audit_rows if row["essay_id"] == chain["record_id"])
                    for chain in chains
                ).items()
            )
        ),
        "score_basis": cfg["measurement"]["score_basis"],
        "target_drop_min": cfg["measurement"]["target_drop_min"],
        "target_drop_min_by_operator": cfg["measurement"].get(
            "target_drop_min_by_operator",
            {},
        ),
        "acceptance_gate_verified": all(
            row["accepted"]
            == (
                row["target_drop"] is not None
                and row["target_drop"] > row["target_drop_min"]
                and not any(
                    bool(check.get("flagged"))
                    for check in row.get("quality_checks", {}).values()
                )
            )
            for row in audit_rows
        ),
        "feature_usage": schema["features"]["usage"],
        "artifact_audit": artifact_report,
        "artifact_quarantine": artifact_quarantine,
        "semantic_artifact_audit": semantic_artifact_report,
        "operator_balance": balance_report,
        "fallback_steps": sum(row["fallback"] for row in audit_rows),
        "by_operator": {key: dict(value) for key, value in sorted(by_operator.items())},
        "by_generator": {key: dict(value) for key, value in sorted(by_generator.items())},
        "by_requested_generator": {
            key: dict(value) for key, value in sorted(by_requested_generator.items())
        },
        "operator_stats": _group_stats(audit_rows, "corruption_op"),
        "generator_stats": _group_stats(audit_rows, "generator"),
        "operator_generator_coverage": _operator_generator_coverage(audit_rows),
        "target_drop_sensitivity": _target_drop_sensitivity(
            audit_rows,
            thresholds=_sensitivity_thresholds(cfg["measurement"]),
        ),
        "operator_threshold_calibration": _calibrate_operator_thresholds(
            audit_rows,
            thresholds=tuple(
                threshold
                for threshold in _sensitivity_thresholds(cfg["measurement"])
                if threshold >= 0.225
            ),
            max_operator_fraction=float(
                cfg.get("balance", {}).get("max_operator_fraction", 0.4)
            ),
        ),
    }
    Path(args.summary_out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    quality_path = (
        Path(args.quality_report_out)
        if args.quality_report_out
        else Path(args.summary_out).with_name(
            Path(args.summary_out).stem + "_quality.json"
        )
    )
    quality_path.write_text(
        json.dumps(
            {
                "artifact_audit": artifact_report,
                "artifact_quarantine": artifact_quarantine,
                "semantic_artifact_audit": semantic_artifact_report,
                "operator_balance": balance_report,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    failures = []
    if (
        bool(cfg.get("artifact_audit", {}).get("fail_pipeline", True))
        and not artifact_report["passed"]
    ):
        failures.append("artifact_audit")
    if (
        bool(cfg.get("semantic_artifact_audit", {}).get("fail_pipeline", True))
        and not semantic_artifact_report["passed"]
    ):
        failures.append("semantic_artifact_audit")
    if (
        bool(cfg.get("balance", {}).get("fail_pipeline", False))
        and not balance_report["passed"]
    ):
        failures.append("operator_balance")
    if failures:
        raise SystemExit(
            "corruption quality gate failed: "
            + ", ".join(failures)
            + f"; see {quality_path}"
        )
    return 0


def _quarantine_artifacts(
    audit_rows: list[dict],
    artifact_cfg: dict,
) -> tuple[list[dict], dict, dict]:
    """Reject every accepted transition participating in a corpus artifact."""

    accepted_rows = [row for row in audit_rows if row["accepted"]]
    initial_report = audit_edit_artifacts(accepted_rows, artifact_cfg)
    report = initial_report
    rejected_pair_ids: set[str] = set()
    iterations = 0
    while not report["passed"]:
        iterations += 1
        affected = {
            str(pair_id)
            for violation in report.get("violations", [])
            for pair_id in violation.get("affected_pair_ids", [])
        }
        changed = 0
        for row in audit_rows:
            pair_id = f"{row['essay_id']}:stage{row['stage_k']}"
            if not row["accepted"] or pair_id not in affected:
                continue
            row["accepted"] = False
            row["acceptance_reason"] = "corpus_artifact"
            row.setdefault("quality_checks", {})["corpus_artifact"] = {
                "flagged": True,
                "pair_id": pair_id,
            }
            rejected_pair_ids.add(pair_id)
            changed += 1
        if changed == 0:
            break
        accepted_rows = [row for row in audit_rows if row["accepted"]]
        report = audit_edit_artifacts(accepted_rows, artifact_cfg)

    return accepted_rows, report, {
        "applied": bool(rejected_pair_ids),
        "iterations": iterations,
        "rejected_steps": len(rejected_pair_ids),
        "rejected_pair_ids": sorted(rejected_pair_ids),
        "initial_artifact_audit": initial_report,
        "final_passed": report["passed"],
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _group_stats(rows: list[dict], key: str) -> dict[str, dict]:
    grouped: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    result = {}
    for name, group in sorted(grouped.items()):
        accepted = [row for row in group if row["accepted"]]
        drops = [float(row["target_drop"]) for row in accepted]
        result[name] = {
            "generated": len(group),
            "accepted": len(accepted),
            "rejected": len(group) - len(accepted),
            "acceptance_rate": len(accepted) / len(group),
            "accepted_target_drop": {
                "mean": statistics.mean(drops) if drops else None,
                "median": statistics.median(drops) if drops else None,
                "min": min(drops) if drops else None,
                "max": max(drops) if drops else None,
            },
        }
    return result


def _operator_generator_coverage(rows: list[dict]) -> dict[str, dict]:
    coverage: dict[str, dict[str, Counter]] = defaultdict(
        lambda: {"requested": Counter(), "applied": Counter()}
    )
    for row in rows:
        operator = str(row["corruption_op"])
        coverage[operator]["requested"][row["requested_generator"]] += 1
        coverage[operator]["applied"][row["generator"]] += 1
    return {
        operator: {
            category: dict(sorted(counts.items()))
            for category, counts in categories.items()
        }
        for operator, categories in sorted(coverage.items())
    }


def _target_drop_sensitivity(
    rows: list[dict],
    thresholds: tuple[float, ...],
) -> dict[str, dict]:
    result = {}
    for threshold in thresholds:
        selected = [
            row
            for row in rows
            if row["target_drop"] is not None
            and row["target_drop"] > threshold
            and not any(
                bool(check.get("flagged"))
                for check in row.get("quality_checks", {}).values()
            )
        ]
        counts = Counter(row["corruption_op"] for row in selected)
        dominant = max(counts, key=counts.get) if counts else None
        result[str(threshold)] = {
            "steps": len(selected),
            "by_operator": dict(sorted(counts.items())),
            "dominant_operator": dominant,
            "dominant_fraction": (
                counts[dominant] / len(selected) if dominant is not None else 0.0
            ),
        }
    return result


def _sensitivity_thresholds(measurement_cfg: dict) -> tuple[float, ...]:
    configured = measurement_cfg.get(
        "target_drop_sensitivity",
        [0.3, 0.4, 0.5, 0.6],
    )
    values = {0.0, 0.225, float(measurement_cfg["target_drop_min"])}
    values.update(float(value) for value in configured)
    return tuple(sorted(values))


def _calibrate_operator_thresholds(
    rows: list[dict],
    *,
    thresholds: tuple[float, ...],
    max_operator_fraction: float,
) -> dict:
    """Find the largest quality-clean pool satisfying the operator cap."""

    operators = sorted({str(row["corruption_op"]) for row in rows})
    quality_clean = [
        row
        for row in rows
        if row["target_drop"] is not None
        and not any(
            bool(check.get("flagged"))
            for check in row.get("quality_checks", {}).values()
        )
    ]
    best = None
    combinations_tested = 0
    for values in product(thresholds, repeat=len(operators)):
        combinations_tested += 1
        minimums = dict(zip(operators, values))
        selected = [
            row
            for row in quality_clean
            if float(row["target_drop"])
            > minimums[str(row["corruption_op"])]
        ]
        counts = Counter(str(row["corruption_op"]) for row in selected)
        if set(counts) != set(operators):
            continue
        dominant_fraction = max(counts.values()) / len(selected)
        if dominant_fraction > max_operator_fraction:
            continue
        candidate = {
            "steps": len(selected),
            "thresholds": minimums,
            "by_operator": dict(sorted(counts.items())),
            "dominant_fraction": dominant_fraction,
        }
        rank = (
            candidate["steps"],
            -sum(values),
            tuple(-value for value in values),
        )
        if best is None or rank > best[0]:
            best = (rank, candidate)
    return {
        "thresholds_tested": list(thresholds),
        "combinations_tested": combinations_tested,
        "max_operator_fraction": max_operator_fraction,
        "recommended": best[1] if best is not None else None,
    }


if __name__ == "__main__":
    raise SystemExit(main())
