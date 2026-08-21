#!/usr/bin/env python
"""Train one Stage-1 TVM configuration without reading the held-out test split."""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
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
from feak_tc.tvm.training import (
    evaluate_pairwise,
    save_adapter,
    score_normalization_mean,
    set_training_seed,
    train_one_epoch,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tvm_stage1.yaml")
    parser.add_argument("--model-key", required=True, choices=["qwen", "kanana"])
    parser.add_argument(
        "--feature-variant", required=True, choices=["full", "scorer_free"]
    )
    parser.add_argument("--learning-rate", required=True, type=float)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--max-length", type=int)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    training_config = config["training"]
    if int(training_config["epochs"]) != 1:
        raise SystemExit("TVM Stage-1 requires exactly one epoch")
    allowed_rates = {float(value) for value in training_config["learning_rates"]}
    if float(args.learning_rate) not in allowed_rates:
        raise SystemExit(
            f"learning rate must be one of the configured sweep values: {sorted(allowed_rates)}"
        )
    output_dir = Path(args.output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise SystemExit(f"output directory is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)

    data_config = config["data"]
    data_path = Path(data_config["path"])
    all_rows = sorted(read_jsonl(data_path), key=row_key)
    similarities, similarity_info = load_pair_similarities(
        data_config["similarity_cache"], all_rows
    )
    rows = _stratified_limit(all_rows, args.limit) if args.limit is not None else all_rows
    if len(rows) < 10:
        raise SystemExit("TVM training requires at least 10 pairs")
    with Path(data_config["elite_stats"]).open(encoding="utf-8") as file:
        elite_payload = yaml.safe_load(file) or {}
    elite_stats = elite_payload.get("features", elite_payload)
    pairs = build_tvm_pairs(rows, elite_stats, similarities)

    split_config = config["split"]
    operator_counts: dict[str, int] = defaultdict(int)
    for row in rows:
        operator_counts[str(row["corruption_op"])] += 1
    fold_count = min(
        int(split_config["folds"]),
        len({row["essay_id"] for row in rows}),
        min(operator_counts.values()),
    )
    split = make_tvm_split(
        rows,
        folds=fold_count,
        test_fold=int(split_config["test_fold"]),
        validation_fold=int(split_config["validation_fold"]),
        seed=int(config["seed"]),
    )
    model_config = config["models"][args.model_key]
    model_name = str(model_config["name"])
    max_length = int(args.max_length or training_config["max_length"])
    set_training_seed(int(config["seed"]))
    tokenizer = load_tokenizer(model_name, local_files_only=True)
    datasets = {
        name: PairwisePromptDataset(
            pairs,
            split[name],
            tokenizer,
            feature_variant=args.feature_variant,
            max_length=max_length,
        )
        for name in ("train", "validation")
    }

    import torch
    from torch.utils.data import DataLoader

    collator = PairwiseCollator(tokenizer)
    generator = torch.Generator()
    generator.manual_seed(int(config["seed"]))
    train_loader = DataLoader(
        datasets["train"],
        batch_size=int(training_config["pair_batch_size"]),
        shuffle=True,
        collate_fn=collator,
        generator=generator,
        num_workers=0,
        pin_memory=True,
    )
    train_score_loader = DataLoader(
        datasets["train"],
        batch_size=int(training_config["pair_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    validation_loader = DataLoader(
        datasets["validation"],
        batch_size=int(training_config["pair_batch_size"]),
        shuffle=False,
        collate_fn=collator,
        num_workers=0,
        pin_memory=True,
    )
    model, model_info = load_reward_model(
        model_name,
        tokenizer=tokenizer,
        load_in_4bit=bool(config["quantization"]["load_in_4bit"])
        and not args.no_4bit,
        gradient_checkpointing=bool(training_config["gradient_checkpointing"]),
        lora=config["lora"],
        local_files_only=True,
    )
    training = train_one_epoch(
        model,
        train_loader,
        learning_rate=float(args.learning_rate),
        weight_decay=float(training_config["weight_decay"]),
        gradient_accumulation_steps=int(
            training_config["gradient_accumulation_steps"]
        ),
        warmup_steps=int(training_config["warmup_steps"]),
        margin_per_stage=float(training_config["margin_per_stage"]),
        max_grad_norm=float(training_config["max_grad_norm"]),
        log_every=int(training_config["log_every_updates"]),
    )
    normalization_mean = score_normalization_mean(model, train_score_loader)
    validation, predictions = evaluate_pairwise(
        model,
        validation_loader,
        margin_per_stage=float(training_config["margin_per_stage"]),
        normalization_mean=normalization_mean,
    )
    save_adapter(model, tokenizer, output_dir)
    report = {
        "gate": "tvm_stage1_validation",
        "test_split_evaluated": False,
        "run_dir": str(output_dir.resolve()),
        "config": str(args.config),
        "model_key": args.model_key,
        "model_role": str(model_config["role"]),
        "model": model_info,
        "feature_variant": args.feature_variant,
        "learning_rate": float(args.learning_rate),
        "seed": int(config["seed"]),
        "data": {
            "path": str(data_path),
            "sha256": file_sha256(data_path),
            "pairs": len(rows),
            "source_pairs": len(all_rows),
            "similarity_cache": similarity_info,
            "prompt_sha256": prompt_sha256(
                pairs, feature_variant=args.feature_variant
            ),
        },
        "split": {
            "metadata": split["metadata"],
            "summary": split["summary"],
            "digest": split_sha256(split),
        },
        "training_config": {
            **dict(training_config),
            "max_length": max_length,
            "load_in_4bit": bool(config["quantization"]["load_in_4bit"])
            and not args.no_4bit,
            "lora": dict(config["lora"]),
        },
        "training": training,
        "normalization": {
            "method": "mean train-state score subtraction",
            "mean": normalization_mean,
        },
        "validation": validation,
        "limitations": [
            "synthetic corruption validation is not human preference validation",
            "held-out test split is intentionally not evaluated by this script",
        ],
    }
    _write_json(output_dir / "validation_report.json", report)
    _write_jsonl(output_dir / "validation_predictions.jsonl", predictions)
    _write_json(output_dir / "split.json", split)
    print(
        json.dumps(
            {
                "output_dir": str(output_dir),
                "model": model_name,
                "feature_variant": args.feature_variant,
                "learning_rate": args.learning_rate,
                "validation_accuracy": validation["pairwise_accuracy"],
                "validation_loss": validation["mean_loss"],
                "test_split_evaluated": False,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stratified_limit(
    rows: Sequence[Mapping[str, Any]], limit: int
) -> list[Mapping[str, Any]]:
    """Build an operator-balanced, longest-first stress subset for smoke runs."""

    if limit < 1:
        raise SystemExit("--limit must be positive")
    by_operator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_operator[str(row["corruption_op"])].append(row)
    for bucket in by_operator.values():
        bucket.sort(
            key=lambda row: len(str(row["text_before"])) + len(str(row["text"])),
            reverse=True,
        )
    selected = []
    position = 0
    operators = sorted(by_operator)
    while len(selected) < min(limit, len(rows)):
        added = False
        for operator in operators:
            bucket = by_operator[operator]
            if position < len(bucket) and len(selected) < limit:
                selected.append(bucket[position])
                added = True
        if not added:
            break
        position += 1
    return sorted(selected, key=row_key)


if __name__ == "__main__":
    raise SystemExit(main())
