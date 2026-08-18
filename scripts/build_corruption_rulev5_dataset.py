#!/usr/bin/env python
"""Build a clean-prefix rule v5 pool with replacement OFFTOPIC transitions."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.quality import (
    audit_edit_artifacts,
    audit_operator_balance,
    audit_semantic_edit_artifacts,
)
from feak_tc.corruption.distractors import is_standalone_distractor_sentence


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original-chains", required=True)
    parser.add_argument("--original-accepted", required=True)
    parser.add_argument("--replacement-accepted", required=True)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()

    chains = _read_jsonl(Path(args.original_chains))
    original = _read_jsonl(Path(args.original_accepted))
    replacement = _read_jsonl(Path(args.replacement_accepted))
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    final, report = build_rulev5_pool(chains, original, replacement, cfg)
    _write_jsonl(Path(args.output), final)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def build_rulev5_pool(
    chains: list[dict],
    original: list[dict],
    replacement: list[dict],
    cfg: dict,
) -> tuple[list[dict], dict]:
    steps_by_essay = {
        str(chain["record_id"]): list(chain.get("steps") or []) for chain in chains
    }
    clean = []
    dropped_contaminated = []
    for row in original:
        history = steps_by_essay[str(row["essay_id"])][: int(row["stage_k"])]
        if any(step.get("operator") == "INSERT_OFFTOPIC" for step in history):
            dropped_contaminated.append(row)
        else:
            clean.append(row)
    accepted_offtopic = [
        row
        for row in replacement
        if row.get("accepted") and row.get("corruption_op") == "INSERT_OFFTOPIC"
    ]
    distractor_cfg = cfg.get("operators", {}).get("INSERT_OFFTOPIC", {})
    minimum = int(distractor_cfg.get("distractor_min_chars", 15))
    maximum = int(distractor_cfg.get("distractor_max_chars", 80))
    expected_insertions = int(distractor_cfg.get("edits_per_step", 1))
    new_offtopic = [
        row
        for row in accepted_offtopic
        if _has_standalone_insertions(
            row,
            expected=expected_insertions,
            minimum=minimum,
            maximum=maximum,
        )
    ]
    rejected_standalone = [
        row for row in accepted_offtopic if row not in new_offtopic
    ]

    balance_cfg = cfg.get("balance", {})
    maximum = float(balance_cfg.get("max_operator_fraction", 0.40))
    maximum_new = math.floor(maximum * len(clean) / max(1e-12, 1.0 - maximum))
    selected_offtopic = sorted(
        new_offtopic,
        key=lambda row: (-float(row["target_drop"]), str(row["essay_id"])),
    )[:maximum_new]
    final = clean + selected_offtopic
    final.sort(key=lambda row: (str(row["essay_id"]), int(row["stage_k"])))

    keys = [(str(row["chain_id"]), int(row["stage_k"])) for row in final]
    artifact = audit_edit_artifacts(final, cfg.get("artifact_audit", {}))
    semantic = audit_semantic_edit_artifacts(
        final,
        cfg.get("semantic_artifact_audit", {}),
    )
    balance = audit_operator_balance(final, balance_cfg)
    passed = (
        len(keys) == len(set(keys))
        and artifact["passed"]
        and semantic["passed"]
        and balance["passed"]
    )
    return final, {
        "gate": "corruption_rulev5_clean_prefix_pool",
        "passed": passed,
        "original_accepted": len(original),
        "clean_prefix_retained": len(clean),
        "old_current_or_downstream_offtopic_dropped": len(dropped_contaminated),
        "replacement_offtopic_accepted": len(accepted_offtopic),
        "replacement_offtopic_standalone_passed": len(new_offtopic),
        "replacement_offtopic_standalone_rejected": len(rejected_standalone),
        "replacement_offtopic_standalone_rejected_pair_ids": [
            f"{row['essay_id']}:stage{row['stage_k']}"
            for row in rejected_standalone
        ],
        "replacement_offtopic_selected": len(selected_offtopic),
        "replacement_offtopic_balance_cap": maximum_new,
        "final_rows": len(final),
        "final_essays": len({row["essay_id"] for row in final}),
        "duplicate_natural_keys": len(keys) - len(set(keys)),
        "operators": dict(Counter(row["corruption_op"] for row in final)),
        "artifact_audit": artifact,
        "semantic_artifact_audit": semantic,
        "operator_balance": balance,
    }


def _read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _has_standalone_insertions(
    row: dict,
    *,
    expected: int,
    minimum: int,
    maximum: int,
) -> bool:
    insertions = [
        str(edit.get("text") or "")
        for edit in row.get("edits", [])
        if edit.get("operation") == "insert_after"
    ]
    return len(insertions) == expected and all(
        is_standalone_distractor_sentence(
            text,
            minimum=minimum,
            maximum=maximum,
        )
        for text in insertions
    )


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
