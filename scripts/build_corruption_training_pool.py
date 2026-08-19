#!/usr/bin/env python
"""Merge accepted corruption transitions into a deduplicated training pool."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

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
    parser.add_argument("--base", required=True, nargs="+")
    parser.add_argument("--additional", required=True, nargs="+")
    parser.add_argument("--target-size", required=True, type=int)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--output", required=True)
    parser.add_argument("--report-out", required=True)
    args = parser.parse_args()
    if args.target_size < 1:
        raise SystemExit("--target-size must be positive")

    base = _read_many(args.base)
    additional = _read_many(args.additional)
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    rows, report = build_training_pool(
        base,
        additional,
        cfg,
        target_size=args.target_size,
    )
    _write_jsonl(Path(args.output), rows)
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if report["passed"] else 2


def build_training_pool(
    base_rows: Sequence[Mapping[str, Any]],
    additional_rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    target_size: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Keep the base pool, then select the strongest clean additional rows."""

    if target_size < 1:
        raise ValueError("target_size must be positive")
    base = [_with_transition_id(row) for row in base_rows if row.get("accepted", True)]
    additional = [
        _with_transition_id(row) for row in additional_rows if row.get("accepted", True)
    ]
    base_keys = [_natural_key(row) for row in base]
    base_transition_ids = [str(row["transition_id"]) for row in base]
    base_pairs = [_text_pair(row) for row in base]
    base_generated_values = [
        text for row in base for text in _generated_edit_texts(row)
    ]
    base_generated_texts = set(base_generated_values)
    base_valid = (
        len(base) <= target_size
        and len(base_keys) == len(set(base_keys))
        and len(base_transition_ids) == len(set(base_transition_ids))
        and len(base_pairs) == len(set(base_pairs))
        and len(base_generated_values) == len(base_generated_texts)
    )

    selected = list(base)
    natural_keys = set(base_keys)
    transition_ids = set(base_transition_ids)
    text_pairs = set(base_pairs)
    generated_texts = set(base_generated_texts)
    operator_counts = Counter(str(row["corruption_op"]) for row in selected)
    max_fraction = float(cfg.get("balance", {}).get("max_operator_fraction", 0.40))
    per_operator_cap = math.floor(max_fraction * target_size)
    skipped = Counter()

    candidates = sorted(
        additional,
        key=lambda row: (
            -_target_drop(row),
            str(row["transition_id"]),
        ),
    )
    for row in candidates:
        if len(selected) >= target_size:
            break
        natural_key = _natural_key(row)
        transition_id = str(row["transition_id"])
        text_pair = _text_pair(row)
        edit_values = _generated_edit_texts(row)
        edit_texts = set(edit_values)
        operator = str(row["corruption_op"])
        if natural_key in natural_keys or transition_id in transition_ids:
            skipped["duplicate_natural_key"] += 1
            continue
        if text_pair in text_pairs:
            skipped["duplicate_text_pair"] += 1
            continue
        if len(edit_values) != len(edit_texts) or edit_texts & generated_texts:
            skipped["duplicate_generated_edit"] += 1
            continue
        if _contains_reserved_context(row, base_generated_texts):
            skipped["base_edit_in_additional_context"] += 1
            continue
        if operator_counts[operator] >= per_operator_cap:
            skipped["operator_cap"] += 1
            continue
        selected.append(row)
        natural_keys.add(natural_key)
        transition_ids.add(transition_id)
        text_pairs.add(text_pair)
        generated_texts.update(edit_texts)
        operator_counts[operator] += 1

    selected.sort(
        key=lambda row: (
            str(row["essay_id"]),
            int(row["stage_k"]),
            str(row["transition_id"]),
        )
    )
    artifact = audit_edit_artifacts(selected, cfg.get("artifact_audit", {}))
    semantic = audit_semantic_edit_artifacts(
        selected,
        cfg.get("semantic_artifact_audit", {}),
    )
    balance = audit_operator_balance(selected, cfg.get("balance", {}))
    generated_edit_count = sum(len(_generated_edit_texts(row)) for row in selected)
    target_reached = len(selected) == target_size
    passed = (
        base_valid
        and target_reached
        and len(natural_keys) == len(selected)
        and len(transition_ids) == len(selected)
        and len(text_pairs) == len(selected)
        and generated_edit_count == len(generated_texts)
        and artifact["passed"]
        and semantic["passed"]
        and balance["passed"]
    )
    return selected, {
        "gate": "corruption_training_pool",
        "passed": passed,
        "target_size": target_size,
        "target_reached": target_reached,
        "base_input_rows": len(base_rows),
        "base_retained": len(base),
        "base_valid": base_valid,
        "additional_input_rows": len(additional_rows),
        "additional_eligible": len(additional),
        "additional_selected": len(selected) - len(base),
        "final_rows": len(selected),
        "final_essays": len({str(row["essay_id"]) for row in selected}),
        "unique_transition_ids": len(transition_ids),
        "unique_natural_keys": len(natural_keys),
        "unique_text_pairs": len(text_pairs),
        "unique_generated_edits": len(generated_texts),
        "generated_edit_count": generated_edit_count,
        "per_operator_cap": per_operator_cap,
        "operators": dict(sorted(operator_counts.items())),
        "skipped": dict(sorted(skipped.items())),
        "artifact_audit": artifact,
        "semantic_artifact_audit": semantic,
        "operator_balance": balance,
    }


def _with_transition_id(row: Mapping[str, Any]) -> dict[str, Any]:
    copied = dict(row)
    chain_id = str(copied["chain_id"])
    stage_k = int(copied["stage_k"])
    copied["transition_id"] = f"{chain_id}:stage{stage_k}"
    return copied


def _natural_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["chain_id"]), int(row["stage_k"])


def _text_pair(row: Mapping[str, Any]) -> tuple[str, str]:
    return str(row.get("text_before") or ""), str(row.get("text") or "")


def _generated_edit_texts(row: Mapping[str, Any]) -> list[str]:
    return [
        str(edit.get("text") or "").strip()
        for edit in row.get("edits", [])
        if str(edit.get("operation") or "") in {"insert_after", "replace"}
        and str(edit.get("text") or "").strip()
    ]


def _contains_reserved_context(
    row: Mapping[str, Any],
    reserved_texts: set[str],
) -> bool:
    before = str(row.get("text_before") or "")
    after = str(row.get("text") or "")
    return any(text in before or text in after for text in reserved_texts)


def _target_drop(row: Mapping[str, Any]) -> float:
    value = row.get("target_drop")
    return float(value) if isinstance(value, (int, float)) else float("-inf")


def _read_many(paths: Sequence[str]) -> list[dict[str, Any]]:
    rows = []
    for raw_path in paths:
        with open(raw_path, encoding="utf-8") as file:
            rows.extend(json.loads(line) for line in file if line.strip())
    return rows


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
