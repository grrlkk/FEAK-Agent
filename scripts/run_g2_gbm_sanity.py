#!/usr/bin/env python
"""Run G2 grouped GBM sanity and prepare the blinded 50-pair human review."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.g2 import (
    build_gbm_pairs,
    build_human_review_pairs,
    evaluate_human_review,
    run_grouped_gbm,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--audit", required=True)
    parser.add_argument("--elite-stats", default="configs/elite_features.yaml")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--human-review-out", required=True)
    parser.add_argument("--human-key-out", required=True)
    parser.add_argument("--human-count", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260722)
    args = parser.parse_args()

    chains = _read_jsonl(Path(args.chains))
    audit_rows = _read_jsonl(Path(args.audit))
    with open(args.elite_stats, encoding="utf-8") as file:
        elite_payload = yaml.safe_load(file) or {}
    elite_stats = elite_payload.get("features", elite_payload)

    gbm_pairs = build_gbm_pairs(audit_rows, elite_stats)
    gbm_report, predictions = run_grouped_gbm(gbm_pairs, seed=args.seed)

    review_path = Path(args.human_review_out)
    key_path = Path(args.human_key_out)
    if review_path.exists() != key_path.exists():
        raise SystemExit("human review and key files must either both exist or both be absent")
    if review_path.exists():
        review_rows = _read_jsonl(review_path)
        key_rows = _read_jsonl(key_path)
        human_sample = {
            "sampled_pairs": len(review_rows),
            "source": "existing_review_files",
        }
    else:
        review_rows, key_rows, human_sample = build_human_review_pairs(
            chains,
            audit_rows,
            count=args.human_count,
            seed=args.seed,
        )
        _write_jsonl(review_path, review_rows)
        _write_jsonl(key_path, key_rows)

    human_gate = evaluate_human_review(
        review_rows,
        key_rows,
        required=args.human_count,
        threshold=0.70,
    )
    similarity_methods = {}
    for pair in gbm_pairs:
        method = str(pair["similarity"].get("method", "unknown"))
        similarity_methods[method] = similarity_methods.get(method, 0) + 1
    report = {
        "gate": "G2",
        "gbm": gbm_report,
        "similarity_methods": similarity_methods,
        "human_sample": human_sample,
        "human_gate": human_gate,
        "g2_status": (
            "passed"
            if gbm_report["significantly_above_random"]
            and human_gate["status"] == "passed"
            else "failed"
            if not gbm_report["significantly_above_random"]
            or human_gate["status"] == "failed"
            else "pending_human_review"
        ),
    }
    _write_jsonl(Path(args.predictions_out), predictions)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
