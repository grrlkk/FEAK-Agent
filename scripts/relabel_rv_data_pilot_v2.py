#!/usr/bin/env python
"""Relabel every RV v2 candidate with three non-generator LLM judges."""

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
from feak_tc.rv.relabel import (
    aggregate_relabels,
    build_relabel_packets,
    relabel_packet_digest,
    request_relabel,
    strict_candidate_quality,
    validate_relabel,
)
from feak_tc.rv.schema import validate_rv_sample


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rv_relabel_v2.yaml")
    parser.add_argument("--limit-states", type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    config = _read_yaml(Path(args.config))
    input_cfg = config["input"]
    output_cfg = config["output"]
    prefix = Path(args.output_prefix or output_cfg["prefix"])
    rows = read_jsonl(input_cfg["dataset"])
    public_rows, key_rows = build_relabel_packets(rows, seed=int(config["seed"]))
    if args.limit_states is not None:
        if args.limit_states < 1:
            raise SystemExit("--limit-states must be positive")
        public_rows = public_rows[: args.limit_states]
        selected = {str(row["review_id"]) for row in public_rows}
        key_rows = [row for row in key_rows if str(row["review_id"]) in selected]
        selected_states = {str(row["state_id"]) for row in key_rows}
        rows = [row for row in rows if str(row["state_id"]) in selected_states]

    public_path = Path(f"{prefix}_public.jsonl")
    key_path = Path(f"{prefix}_hidden_key.jsonl")
    write_jsonl_atomic(public_path, public_rows)
    write_jsonl_atomic(key_path, key_rows)
    print(f"packets: states={len(public_rows)} candidates={len(rows)}", flush=True)

    request_cfg = config["request"]
    results_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_cfg in config["models"]:
        name = str(model_cfg["name"])
        results_by_model[name] = _review_model(
            public_rows,
            Path(f"{prefix}_{name}.jsonl"),
            model_cfg,
            workers=int(request_cfg["workers"]),
            max_attempts=int(request_cfg["max_attempts"]),
            timeout=float(request_cfg["timeout"]),
        )

    quality_prefix = str(input_cfg["candidate_judge_prefix"])
    quality_key = read_jsonl(f"{quality_prefix}_hidden_key.jsonl")
    quality_results = {
        name: read_jsonl(f"{quality_prefix}_{name}.jsonl")
        for name in input_cfg["candidate_quality_judges"]
    }
    generated_quality = strict_candidate_quality(quality_key, quality_results)
    all_rows, train_rows, report = aggregate_relabels(
        rows,
        key_rows,
        results_by_model,
        generated_quality,
        dataset_version=str(config["version"]),
    )
    for row in train_rows:
        validate_rv_sample(row)
    report.update(
        {
            "input_dataset": str(input_cfg["dataset"]),
            "public_packet": str(public_path),
            "hidden_key": str(key_path),
            "model_ids": {
                str(item["name"]): str(item["id"]) for item in config["models"]
            },
        }
    )
    all_path = Path(f"{prefix}_all.jsonl")
    train_path = Path(f"{prefix}_train.jsonl")
    report_path = Path(f"{prefix}_report.json")
    write_jsonl_atomic(all_path, all_rows)
    write_jsonl_atomic(train_path, train_rows)
    write_json_atomic(report_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"all={all_path} train={train_path} report={report_path}", flush=True)
    return 0


def _review_model(
    public_rows: Sequence[Mapping[str, Any]],
    path: Path,
    model_cfg: Mapping[str, Any],
    *,
    workers: int,
    max_attempts: int,
    timeout: float,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be positive")
    name = str(model_cfg["name"])
    model_id = str(model_cfg["id"])
    public_by_id = {str(row["review_id"]): row for row in public_rows}
    existing: dict[str, dict[str, Any]] = {}
    if path.exists():
        for saved in read_jsonl(path):
            review_id = str(saved.get("review_id") or "")
            packet = public_by_id.get(review_id)
            if (
                packet is None
                or saved.get("model") != model_id
                or saved.get("public_packet_sha256") != relabel_packet_digest(packet)
            ):
                continue
            try:
                validate_relabel(saved)
            except ValueError:
                continue
            existing[review_id] = dict(saved)
    pending = [row for row in public_rows if str(row["review_id"]) not in existing]
    print(f"{name}: complete={len(existing)} pending={len(pending)}", flush=True)
    lock = threading.Lock()

    def call_one(packet: Mapping[str, Any]) -> dict[str, Any]:
        result = request_relabel(
            packet,
            model=model_id,
            max_attempts=max_attempts,
            timeout=timeout,
            reasoning_effort=model_cfg.get("reasoning_effort"),
            verbosity=model_cfg.get("verbosity"),
        )
        return {
            "review_id": str(packet["review_id"]),
            "model": model_id,
            "review_kind": "rv_instance_relabel_v2",
            "public_packet_sha256": relabel_packet_digest(packet),
            **result,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_one, row): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                existing[result["review_id"]] = result
                _write_ordered(path, public_rows, existing)
                print(f"{name}: {len(existing)}/{len(public_rows)} {result['review_id']}", flush=True)
    return _write_ordered(path, public_rows, existing)


def _write_ordered(
    path: Path,
    public_rows: Sequence[Mapping[str, Any]],
    indexed: Mapping[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    ordered = [
        indexed[str(row["review_id"])]
        for row in public_rows
        if str(row["review_id"]) in indexed
    ]
    write_jsonl_atomic(path, ordered)
    return ordered


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
