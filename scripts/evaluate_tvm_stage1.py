#!/usr/bin/env python
"""Evaluate a validation-selected TVM once on the held-out synthetic test split."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.tvm.data import (
    PairwiseCollator,
    PairwisePromptDataset,
    build_tvm_pairs,
    file_sha256,
    load_pair_similarities,
    make_tvm_split,
    prompt_sha256,
    read_jsonl,
    row_key,
    split_sha256,
)
from feak_tc.tvm.modeling import load_reward_model, load_tokenizer
from feak_tc.tvm.training import evaluate_pairwise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selection-manifest", required=True)
    parser.add_argument("--config", default="configs/tvm_stage1.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    selection = json.loads(
        Path(args.selection_manifest).read_text(encoding="utf-8")
    )
    if selection.get("gate") != "tvm_stage1_validation_selection" or bool(
        selection.get("test_metrics_read")
    ):
        raise SystemExit("invalid validation-only selection manifest")
    selected_by_dir = {
        str(Path(row["run_dir"]).resolve()): row
        for row in selection.get("selected", [])
    }
    selected_entry = selected_by_dir.get(str(run_dir.resolve()))
    if selected_entry is None:
        raise SystemExit("run directory was not selected by the validation sweep")
    validation_path = run_dir / "validation_report.json"
    split_path = run_dir / "split.json"
    if not validation_path.exists() or not split_path.exists():
        raise SystemExit(f"incomplete TVM run directory: {run_dir}")
    report_path = run_dir / "test_report.json"
    predictions_path = run_dir / "test_predictions.jsonl"
    if not args.overwrite and (report_path.exists() or predictions_path.exists()):
        raise SystemExit(f"test output already exists for {run_dir}")

    validation_report = json.loads(validation_path.read_text(encoding="utf-8"))
    stored_split = json.loads(split_path.read_text(encoding="utf-8"))
    if bool(
        validation_report.get(
            "test_split_evaluated", validation_report.get("test_split_read")
        )
    ):
        raise SystemExit("run was not produced by a validation-only sweep")
    if (
        str(selected_entry.get("model_key")) != str(validation_report["model_key"])
        or str(selected_entry.get("feature_variant"))
        != str(validation_report["feature_variant"])
        or float(selected_entry.get("learning_rate"))
        != float(validation_report["learning_rate"])
    ):
        raise SystemExit("selection entry does not match the run validation report")
    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    data_config = config["data"]
    data_path = Path(data_config["path"])
    if file_sha256(data_path) != str(validation_report["data"]["sha256"]):
        raise SystemExit("training data changed after validation training")
    rows = sorted(read_jsonl(data_path), key=row_key)
    similarities, similarity_info = load_pair_similarities(
        data_config["similarity_cache"], rows
    )
    if similarity_info["sha256"] != validation_report["data"]["similarity_cache"]["sha256"]:
        raise SystemExit("similarity cache changed after validation training")
    with Path(data_config["elite_stats"]).open(encoding="utf-8") as file:
        elite_payload = yaml.safe_load(file) or {}
    pairs = build_tvm_pairs(rows, elite_payload.get("features", elite_payload), similarities)
    split_config = config["split"]
    computed_split = make_tvm_split(
        rows,
        folds=int(split_config["folds"]),
        test_fold=int(split_config["test_fold"]),
        validation_fold=int(split_config["validation_fold"]),
        seed=int(config["seed"]),
    )
    if split_sha256(stored_split) != split_sha256(computed_split):
        raise SystemExit("stored and recomputed split assignments differ")
    if split_sha256(stored_split) != str(validation_report["split"]["digest"]):
        raise SystemExit("validation report split digest differs")
    feature_variant = str(validation_report["feature_variant"])
    if prompt_sha256(pairs, feature_variant=feature_variant) != str(
        validation_report["data"]["prompt_sha256"]
    ):
        raise SystemExit("TVM prompt content changed after validation training")

    model_name = str(validation_report["model"]["model_name"])
    tokenizer = load_tokenizer(model_name, local_files_only=True)
    dataset = PairwisePromptDataset(
        pairs,
        stored_split["test"],
        tokenizer,
        feature_variant=feature_variant,
        max_length=int(validation_report["training_config"]["max_length"]),
    )
    from torch.utils.data import DataLoader

    loader = DataLoader(
        dataset,
        batch_size=int(validation_report["training_config"]["pair_batch_size"]),
        shuffle=False,
        collate_fn=PairwiseCollator(tokenizer),
        num_workers=0,
        pin_memory=True,
    )
    model, model_info = load_reward_model(
        model_name,
        tokenizer=tokenizer,
        load_in_4bit=bool(validation_report["training_config"]["load_in_4bit"]),
        gradient_checkpointing=False,
        adapter_path=run_dir / "adapter",
        local_files_only=True,
    )
    test, predictions = evaluate_pairwise(
        model,
        loader,
        margin_per_stage=float(
            validation_report["training_config"]["margin_per_stage"]
        ),
        normalization_mean=float(validation_report["normalization"]["mean"]),
    )
    report = {
        "gate": "tvm_stage1_synthetic_test",
        "model_key": validation_report["model_key"],
        "model_role": validation_report["model_role"],
        "model": model_info,
        "feature_variant": feature_variant,
        "learning_rate": validation_report["learning_rate"],
        "validation": validation_report["validation"],
        "test": test,
        "data_sha256": validation_report["data"]["sha256"],
        "split_digest": validation_report["split"]["digest"],
        "normalization": validation_report["normalization"],
        "limitations": [
            "this test is synthetic corruption preference, not human preference",
            "the test result is valid only if this run was selected without test metrics",
        ],
    }
    _write_json(report_path, report)
    _write_jsonl(predictions_path, predictions)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    raise SystemExit(main())
