#!/usr/bin/env python
"""Selectively regenerate failed RV candidates into a v2 staging dataset."""

from __future__ import annotations

import argparse
import hashlib
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
    REGENERATION_PROMPT_VERSION,
    apply_replacements,
    build_state_groups,
    generate_replacement,
    regeneration_input_digest,
    select_regeneration_rows,
    validate_replacement,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rv_pilot_v2.yaml")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--limit", type=int, help="Process only the first N targets for a smoke run.")
    args = parser.parse_args()

    config = _read_yaml(Path(args.config))
    input_path = Path(config["input"]["dataset"])
    output = config["output"]
    staging_path = Path(output["staging_dataset"])
    cache_path = Path(output["regeneration_cache"])
    report_path = Path(output["regeneration_report"])
    generation_cfg = config["regeneration"]
    workers = int(args.workers or generation_cfg.get("workers", 1))
    if workers < 1:
        raise SystemExit("--workers must be positive")

    rows = read_jsonl(input_path)
    groups = build_state_groups(rows)
    selected = select_regeneration_rows(
        rows,
        over_edit_actions=generation_cfg.get("over_edit_actions", ["ADD_DETAIL"]),
    )
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        selected = selected[: args.limit]
    cached = _load_cache(cache_path, selected, groups, generation_cfg)
    pending = [row for row in selected if str(row["sample_id"]) not in cached]
    print(
        f"selective regeneration: selected={len(selected)} cache_hits={len(cached)} "
        f"pending={len(pending)} model={generation_cfg['model']}",
        flush=True,
    )
    generated = _generate_pending(
        pending,
        groups,
        generation_cfg,
        cache_path=cache_path,
        workers=workers,
    )
    replacements = {**cached, **generated}
    staging_rows = apply_replacements(
        rows,
        replacements,
        dataset_version=str(config["version"]),
        model=str(generation_cfg["model"]),
    )
    write_jsonl_atomic(staging_path, staging_rows)
    report = {
        "version": str(config["version"]),
        "prompt_version": REGENERATION_PROMPT_VERSION,
        "source_dataset": str(input_path),
        "source_sha256": _sha256(input_path),
        "staging_dataset": str(staging_path),
        "staging_sha256": _sha256(staging_path),
        "source_rows": len(rows),
        "staging_rows": len(staging_rows),
        "selected_candidates": len(selected),
        "regenerated_candidates": len(replacements),
        "cache_hits": len(cached),
        "new_api_generations": len(generated),
        "by_candidate_type": _counts(selected, "candidate_type"),
        "by_action": _counts(selected, "intended_action"),
        "all_rows_pending_relabel": all(
            row["label_status"] == "pending_instance_relabel"
            and row["training_eligible"] is False
            for row in staging_rows
        ),
    }
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0


def _generate_pending(
    rows: Sequence[Mapping[str, Any]],
    groups: Mapping[str, Mapping[str, Mapping[str, Any]]],
    config: Mapping[str, Any],
    *,
    cache_path: Path,
    workers: int,
) -> dict[str, dict[str, Any]]:
    generated: dict[str, dict[str, Any]] = {}
    failures: list[str] = []
    lock = threading.Lock()

    def call_one(row: Mapping[str, Any]) -> dict[str, Any]:
        group = groups[str(row["state_id"])]
        text, metrics = generate_replacement(row, group, config)
        return {
            "sample_id": str(row["sample_id"]),
            "input_sha256": regeneration_input_digest(row, group),
            "prompt_version": REGENERATION_PROMPT_VERSION,
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
                failures.append(f"{sample_id}: {exc}")
                print(f"failed {sample_id}: {exc}", flush=True)
                continue
            with lock:
                generated[sample_id] = entry
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                with cache_path.open("a", encoding="utf-8") as file:
                    file.write(json.dumps(entry, ensure_ascii=False) + "\n")
                print(f"generated {len(generated)}/{len(rows)} {sample_id}", flush=True)
    if failures:
        raise RuntimeError(
            f"{len(failures)} selective regenerations failed; rerun to resume. "
            + " | ".join(failures[:5])
        )
    return generated


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
            entry.get("input_sha256") != regeneration_input_digest(row, group)
            or entry.get("prompt_version") != REGENERATION_PROMPT_VERSION
            or entry.get("model") != config["model"]
        ):
            continue
        try:
            metrics = validate_replacement(
                row,
                group,
                str(entry["text"]),
                config.get("validation", {}),
            )
        except (KeyError, ValueError):
            continue
        cached[sample_id] = {**dict(entry), "metrics": metrics}
    return cached


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for row in rows:
        value = str(row[field])
        counts[value] = counts.get(value, 0) + 1
    return dict(sorted(counts.items()))


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
