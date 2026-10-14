#!/usr/bin/env python
"""Strongly regenerate v2 candidates rejected by both non-generator judges."""

from __future__ import annotations

import argparse
import json
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.rv.judge import read_jsonl, write_json_atomic, write_jsonl_atomic
from feak_tc.rv.rebuild import (
    STRONG_RETRY_PROMPT_VERSION,
    apply_replacements,
    build_state_groups,
    generate_replacement,
    regeneration_input_digest,
    validate_replacement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rv_pilot_v2_retry.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--skip-pending",
        action="store_true",
        help="Finalize cached successes and report remaining candidates as failed.",
    )
    args = parser.parse_args()

    config = _read_yaml(Path(args.config))
    input_cfg = config["input"]
    output_cfg = config["output"]
    generation_cfg = config["regeneration"]
    staging_path = Path(input_cfg["staging_dataset"])
    prefix = str(input_cfg["candidate_judge_prefix"])
    cache_path = Path(output_cfg["retry_cache"])
    report_path = Path(output_cfg["retry_report"])
    workers = int(args.workers or generation_cfg.get("workers", 1))

    rows = read_jsonl(staging_path)
    groups = build_state_groups(rows)
    rejected_ids = _strictly_rejected_regenerated_ids(rows, prefix)
    selected = sorted(
        [row for row in rows if str(row["sample_id"]) in rejected_ids],
        key=lambda row: str(row["sample_id"]),
    )
    cached = _load_cache(cache_path, selected, groups, generation_cfg)
    pending = [row for row in selected if str(row["sample_id"]) not in cached]
    print(
        f"strong retry: rejected={len(selected)} cache_hits={len(cached)} "
        f"pending={len(pending)}",
        flush=True,
    )
    if args.skip_pending:
        generated = {}
        failures = {
            str(row["sample_id"]): "not regenerated after exhausting strong retries"
            for row in pending
        }
    else:
        generated, failures = _generate_pending(
            pending,
            groups,
            generation_cfg,
            cache_path=cache_path,
            workers=workers,
        )
    replacements = {**cached, **generated}
    updated = apply_replacements(
        rows,
        replacements,
        dataset_version=str(config["version"]),
        model=str(generation_cfg["model"]),
    )
    write_jsonl_atomic(staging_path, updated)
    report = {
        "version": str(config["version"]),
        "prompt_version": STRONG_RETRY_PROMPT_VERSION,
        "strictly_rejected_regenerated_candidates": len(selected),
        "by_candidate_type": _counts(selected, "candidate_type"),
        "cache_hits": len(cached),
        "new_api_generations": len(generated),
        "updated_candidates": len(replacements),
        "failed_candidates": len(failures),
        "failed_sample_ids": sorted(failures),
        "staging_dataset": str(staging_path),
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if not failures else 2


def _strictly_rejected_regenerated_ids(
    rows: Sequence[Mapping[str, Any]], prefix: str
) -> set[str]:
    key_rows = read_jsonl(f"{prefix}_hidden_key.jsonl")
    rater_paths = sorted(Path(f"{prefix}_{name}.jsonl") for name in ("gpt5", "gpt41"))
    raters = [
        {str(row["review_id"]): row for row in read_jsonl(path)}
        for path in rater_paths
    ]
    regenerated = {
        str(row["sample_id"]): row
        for row in rows
        if row.get("provenance", {}).get("candidate_method")
        == "llm_selective_regeneration_v2"
    }
    rejected = set()
    for key in key_rows:
        review_id = str(key["review_id"])
        for side in ("candidate_a", "candidate_b"):
            sample_id = str(key[side]["sample_id"])
            if sample_id not in regenerated:
                continue
            expected = str(key[side]["candidate_type"])
            if not all(
                rater[review_id][side]["inferred_candidate_type"] == expected
                for rater in raters
            ):
                rejected.add(sample_id)
    return rejected


def _generate_pending(
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    cache_path: Path,
    workers: int,
) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    generated: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    lock = threading.Lock()

    def call_one(row: Mapping[str, Any]) -> dict[str, Any]:
        group = groups[str(row["state_id"])]
        text, metrics = generate_replacement(
            row,
            group,
            config,
            generation_variant="strong",
        )
        return {
            "sample_id": str(row["sample_id"]),
            "input_sha256": regeneration_input_digest(
                row, group, generation_variant="strong"
            ),
            "prompt_version": STRONG_RETRY_PROMPT_VERSION,
            "model": str(config["model"]),
            "text": text,
            "metrics": metrics,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_one, row): row for row in rows}
        for future in as_completed(futures):
            row = futures[future]
            sample_id = str(row["sample_id"])
            try:
                entry = future.result()
            except Exception as exc:
                failures[sample_id] = str(exc)
                print(f"failed {sample_id}: {exc}", flush=True)
                continue
            with lock:
                generated[sample_id] = entry
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"generated {len(generated)}/{len(rows)} {sample_id}", flush=True)
    return generated, failures


def _load_cache(
    path: Path,
    selected: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
) -> dict[str, dict[str, Any]]:
    if not path.exists():
        return {}
    by_id = {str(row["sample_id"]): row for row in selected}
    cached = {}
    for entry in read_jsonl(path):
        sample_id = str(entry.get("sample_id") or "")
        row = by_id.get(sample_id)
        if row is None:
            continue
        group = groups[str(row["state_id"])]
        if (
            entry.get("input_sha256")
            != regeneration_input_digest(row, group, generation_variant="strong")
            or entry.get("prompt_version") != STRONG_RETRY_PROMPT_VERSION
            or entry.get("model") != config["model"]
        ):
            continue
        try:
            metrics = validate_replacement(
                row,
                group,
                str(entry["text"]),
                config.get("validation", {}),
                generation_variant="strong",
            )
        except (KeyError, ValueError):
            continue
        cached[sample_id] = {**dict(entry), "metrics": metrics}
    return cached


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    result: dict[str, int] = {}
    for row in rows:
        key = str(row[field])
        result[key] = result.get(key, 0) + 1
    return dict(sorted(result.items()))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
