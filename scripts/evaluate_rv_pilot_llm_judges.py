#!/usr/bin/env python
"""Run independent blind LLM reviews of RV pilot candidate labels."""

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

from feak_tc.rv.judge import (
    analyze_judgments,
    build_blind_packets,
    read_jsonl,
    request_judgment,
    review_packet_digest,
    validate_judgment,
    write_json_atomic,
    write_jsonl_atomic,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rv_llm_judge.yaml")
    parser.add_argument("--sample-states", type=int)
    parser.add_argument("--output-prefix")
    args = parser.parse_args()

    config = _read_yaml(Path(args.config))
    sample_states = args.sample_states or int(config["sample_states"])
    prefix = Path(args.output_prefix or config["output"]["prefix"])
    dataset_path = Path(config["input"]["dataset"])
    model_configs = list(config["models"])
    model_ids = [str(item["id"]) for item in model_configs]
    if len(model_ids) < 2 or len(set(model_ids)) != len(model_ids):
        raise SystemExit("configure at least two distinct model IDs")

    public_rows, key_rows = build_blind_packets(
        read_jsonl(dataset_path),
        sample_states=sample_states,
        seed=int(config["seed"]),
    )
    public_path = Path(f"{prefix}_public.jsonl")
    key_path = Path(f"{prefix}_hidden_key.jsonl")
    write_jsonl_atomic(public_path, public_rows)
    write_jsonl_atomic(key_path, key_rows)
    print(
        f"packet: states={len(public_rows)} public={public_path} key={key_path}",
        flush=True,
    )

    request_cfg = config["request"]
    results_by_model: dict[str, list[dict[str, Any]]] = {}
    for model_config in model_configs:
        model_name = str(model_config["name"])
        result_path = Path(f"{prefix}_{model_name}.jsonl")
        results_by_model[model_name] = _review_model(
            public_rows=public_rows,
            path=result_path,
            model_config=model_config,
            workers=int(request_cfg["workers"]),
            max_attempts=int(request_cfg["max_attempts"]),
            timeout=float(request_cfg["timeout"]),
        )

    # The hidden key is first consumed after every model-facing call is done.
    report, disagreements = analyze_judgments(results_by_model, key_rows)
    report.update(
        {
            "version": str(config["version"]),
            "seed": int(config["seed"]),
            "dataset": str(dataset_path),
            "public_packet": str(public_path),
            "hidden_key": str(key_path),
            "model_ids": {
                str(item["name"]): str(item["id"])
                for item in model_configs
            },
        }
    )
    report_path = Path(f"{prefix}_report.json")
    disagreements_path = Path(f"{prefix}_disagreements.jsonl")
    write_json_atomic(report_path, report)
    write_jsonl_atomic(disagreements_path, disagreements)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print(f"disagreements: {disagreements_path}", flush=True)
    return 0


def _review_model(
    *,
    public_rows: Sequence[Mapping[str, Any]],
    path: Path,
    model_config: Mapping[str, Any],
    workers: int,
    max_attempts: int,
    timeout: float,
) -> list[dict[str, Any]]:
    if workers < 1:
        raise ValueError("workers must be at least 1")
    model_name = str(model_config["name"])
    model_id = str(model_config["id"])
    existing = read_jsonl(path) if path.exists() else []
    public_by_id = {str(row["review_id"]): row for row in public_rows}
    existing_by_id = {}
    for saved in existing:
        review_id = str(saved.get("review_id") or "")
        public_row = public_by_id.get(review_id)
        if public_row is None or saved.get("model") != model_id:
            continue
        expected_digest = review_packet_digest(public_row)
        saved_digest = saved.get("public_packet_sha256")
        if saved_digest != expected_digest:
            continue
        if not _valid_saved_judgment(saved):
            continue
        existing_by_id[review_id] = dict(saved)
    pending = [
        row for row in public_rows
        if str(row["review_id"]) not in existing_by_id
    ]
    print(
        f"{model_name}: model={model_id} complete={len(public_rows) - len(pending)} "
        f"pending={len(pending)}",
        flush=True,
    )
    lock = threading.Lock()

    def call_one(public_row: Mapping[str, Any]) -> dict[str, Any]:
        judgment = request_judgment(
            public_row,
            model=model_id,
            max_attempts=max_attempts,
            timeout=timeout,
            reasoning_effort=model_config.get("reasoning_effort"),
            verbosity=model_config.get("verbosity"),
        )
        return {
            "review_id": str(public_row["review_id"]),
            "model": model_id,
            "review_kind": "openai_api_independent_blind",
            "public_packet_sha256": review_packet_digest(public_row),
            **judgment,
        }

    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(call_one, row): row for row in pending}
        for future in as_completed(futures):
            result = future.result()
            with lock:
                existing_by_id[result["review_id"]] = result
                ordered = [
                    existing_by_id[str(row["review_id"])]
                    for row in public_rows
                    if str(row["review_id"]) in existing_by_id
                ]
                write_jsonl_atomic(path, ordered)
                print(
                    f"{model_name}: {len(ordered)}/{len(public_rows)} "
                    f"{result['review_id']}",
                    flush=True,
                )
    ordered = [existing_by_id[str(row["review_id"])] for row in public_rows]
    write_jsonl_atomic(path, ordered)
    return ordered


def _valid_saved_judgment(row: Mapping[str, Any]) -> bool:
    try:
        validate_judgment(row)
    except ValueError:
        return False
    return row.get("review_kind") == "openai_api_independent_blind"


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as file:
        payload = yaml.safe_load(file)
    if not isinstance(payload, dict):
        raise ValueError(f"config must be a mapping: {path}")
    return payload


if __name__ == "__main__":
    raise SystemExit(main())
