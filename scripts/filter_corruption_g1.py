#!/usr/bin/env python
"""Join G1 chains with diagnoses and keep strict target-rubric drops."""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.measure import evaluate_chain


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--measured", required=True, nargs="+")
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--audit-out", required=True)
    parser.add_argument("--accepted-out", required=True)
    parser.add_argument("--summary-out", required=True)
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
    accepted_rows = [row for row in audit_rows if row["accepted"]]

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
        "acceptance_gate_verified": all(
            row["accepted"]
            == (
                row["target_drop"] is not None
                and row["target_drop"] > cfg["measurement"]["target_drop_min"]
            )
            for row in audit_rows
        ),
        "feature_usage": schema["features"]["usage"],
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
            audit_rows, thresholds=(0.0, 0.01, 0.05, 0.1, 0.225, 0.3)
        ),
    }
    Path(args.summary_out).write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


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
            if row["target_drop"] is not None and row["target_drop"] > threshold
        ]
        result[str(threshold)] = {
            "steps": len(selected),
            "by_operator": dict(
                sorted(Counter(row["corruption_op"] for row in selected).items())
            ),
        }
    return result


if __name__ == "__main__":
    raise SystemExit(main())
