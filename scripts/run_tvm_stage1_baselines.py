#!/usr/bin/env python
"""Run fixed-split baselines after TVM validation-only selection is frozen."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.tvm.baselines import run_fixed_baselines
from feak_tc.tvm.data import (
    build_tvm_pairs,
    file_sha256,
    load_pair_similarities,
    make_tvm_split,
    read_jsonl,
    row_key,
    split_sha256,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--config", default="configs/tvm_stage1.yaml")
    parser.add_argument("--heuristic-config", default="configs/heuristic_stage_a_soft.yaml")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    args = parser.parse_args()

    selection = json.loads(Path(args.selection_manifest).read_text(encoding="utf-8"))
    if selection.get("gate") != "tvm_stage1_validation_selection" or bool(
        selection.get("test_metrics_read")
    ):
        raise SystemExit("baselines require a frozen validation-only selection manifest")
    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    with Path(args.heuristic_config).open(encoding="utf-8") as file:
        heuristic = yaml.safe_load(file) or {}
    data_config = config["data"]
    rows = sorted(read_jsonl(data_config["path"]), key=row_key)
    similarities, similarity_info = load_pair_similarities(
        data_config["similarity_cache"], rows
    )
    with Path(data_config["elite_stats"]).open(encoding="utf-8") as file:
        elite_payload = yaml.safe_load(file) or {}
    pairs = build_tvm_pairs(rows, elite_payload.get("features", elite_payload), similarities)
    split_config = config["split"]
    split = make_tvm_split(
        rows,
        folds=int(split_config["folds"]),
        test_fold=int(split_config["test_fold"]),
        validation_fold=int(split_config["validation_fold"]),
        seed=int(config["seed"]),
    )
    data_digest = file_sha256(data_config["path"])
    split_digest = split_sha256(split)
    if any(
        str(row.get("data_sha256")) != data_digest
        or str(row.get("split_digest")) != split_digest
        for row in selection.get("selected", [])
    ):
        raise SystemExit("baseline data/split differs from the selected TVM runs")
    cache = np.load(data_config["similarity_cache"], allow_pickle=False)
    report, predictions = run_fixed_baselines(
        rows,
        pairs,
        cache["embeddings"],
        split,
        seed=int(config["seed"]),
        heuristic_config=heuristic,
    )
    payload = {
        "gate": "tvm_stage1_fixed_split_baselines",
        "selection_manifest": str(Path(args.selection_manifest).resolve()),
        "test_opened_after_selection": True,
        "data": {
            "path": str(data_config["path"]),
            "sha256": data_digest,
            "pairs": len(rows),
            "similarity_cache": similarity_info,
        },
        "split": {"digest": split_digest, "summary": split["summary"]},
        **report,
    }
    Path(args.report_out).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    _write_jsonl(Path(args.predictions_out), predictions)
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
