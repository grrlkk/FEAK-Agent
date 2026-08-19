#!/usr/bin/env python
"""Run pre-measurement quality gates on complete corruption chains."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.quality import (
    audit_edit_artifacts,
    audit_operator_balance,
    audit_semantic_edit_artifacts,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--report-out", required=True)
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="accept validated non-empty chain prefixes whose later step failed",
    )
    args = parser.parse_args()

    with open(args.chains, encoding="utf-8") as file:
        chains = [json.loads(line) for line in file if line.strip()]
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    report = audit_generated_chains(chains, cfg, allow_partial=args.allow_partial)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if not report["passed"]:
        raise SystemExit(f"generated corruption quality gate failed; see {args.report_out}")
    return 0


def audit_generated_chains(
    chains: list[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    allow_partial: bool = False,
) -> dict[str, Any]:
    record_ids = [str(chain["record_id"]) for chain in chains]
    duplicate_record_ids = sorted(
        record_id for record_id, count in Counter(record_ids).items() if count > 1
    )
    rows = []
    preservation_failures = 0
    fallback_steps = 0
    normalized_steps = 0
    malformed_chains = 0
    status_counts: Counter[str] = Counter()
    for chain in chains:
        steps = list(chain.get("steps") or [])
        states = list(chain.get("states") or [])
        status = str(chain.get("status") or "")
        status_counts[status] += 1
        allowed_statuses = {"ok", "partial"} if allow_partial else {"ok"}
        if (
            status not in allowed_statuses
            or not steps
            or len(states) != len(steps) + 1
        ):
            malformed_chains += 1
        for stage_k, step in enumerate(steps, 1):
            rows.append(
                {
                    "essay_id": str(chain["record_id"]),
                    "question": str(chain.get("question") or ""),
                    "text_before": states[stage_k - 1] if stage_k - 1 < len(states) else "",
                    "stage_k": stage_k,
                    "corruption_op": step["operator"],
                    "edits": step.get("edits", []),
                }
            )
            preservation_failures += not bool(
                step.get("preservation_check", {}).get("passed")
            )
            fallback_steps += bool(step.get("fallback"))
            normalized_steps += bool(step.get("normalized"))

    artifact_report = audit_edit_artifacts(rows, cfg.get("artifact_audit", {}))
    semantic_artifact_report = audit_semantic_edit_artifacts(
        rows,
        cfg.get("semantic_artifact_audit", {}),
    )
    balance_report = audit_operator_balance(rows, cfg.get("balance", {}))
    balance_blocks = bool(cfg.get("balance", {}).get("fail_pipeline", False))
    passed = (
        not duplicate_record_ids
        and malformed_chains == 0
        and preservation_failures == 0
        and fallback_steps == 0
        and normalized_steps == 0
        and artifact_report["passed"]
        and semantic_artifact_report["passed"]
        and (balance_report["passed"] or not balance_blocks)
    )
    return {
        "gate": "generated_corruption_quality",
        "passed": passed,
        "chains": len(chains),
        "allow_partial": allow_partial,
        "chain_statuses": dict(status_counts),
        "transitions": len(rows),
        "duplicate_record_ids": duplicate_record_ids,
        "malformed_chains": malformed_chains,
        "preservation_failures": preservation_failures,
        "fallback_steps": fallback_steps,
        "normalized_steps": normalized_steps,
        "artifact_audit": artifact_report,
        "semantic_artifact_audit": semantic_artifact_report,
        "generated_operator_balance": balance_report,
    }


if __name__ == "__main__":
    raise SystemExit(main())
