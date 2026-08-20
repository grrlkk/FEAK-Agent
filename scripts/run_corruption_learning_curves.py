#!/usr/bin/env python
"""Compare feature-only and frozen-BGE pairwise corruption learning curves."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.g2 import TRANSITION_FEATURES, build_gbm_pairs
from feak_tc.corruption.learning_curve import (
    assess_generation_need,
    make_grouped_folds,
    run_feature_learning_curve,
    run_text_learning_curve,
    state_prompt,
)
from feak_tc.corruption.quality import (
    _local_huggingface_snapshot,
    _semantic_embedding_model,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--config", default="configs/corruption_learning_curve.yaml")
    parser.add_argument("--elite-stats", default="configs/elite_features.yaml")
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--predictions-out", required=True)
    parser.add_argument("--embeddings-cache", required=True)
    parser.add_argument("--refresh-embeddings", action="store_true")
    parser.add_argument("--model")
    parser.add_argument("--device")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--folds", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--train-sizes", type=int, nargs="+")
    parser.add_argument("--plateau-max-gain", type=float)
    parser.add_argument("--minimum-text-accuracy", type=float)
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file) or {}
    embedding_cfg = cfg.get("embedding", {})
    decision_cfg = cfg.get("generation_decision", {})
    seed = int(args.seed if args.seed is not None else cfg["seed"])
    fold_count = int(args.folds if args.folds is not None else cfg["folds"])
    train_sizes = list(
        args.train_sizes if args.train_sizes is not None else cfg["train_sizes"]
    )
    model_name = str(args.model or embedding_cfg["model"])
    device = str(args.device or embedding_cfg["device"])
    batch_size = int(args.batch_size or embedding_cfg["batch_size"])
    plateau_max_gain = float(
        args.plateau_max_gain
        if args.plateau_max_gain is not None
        else decision_cfg["plateau_max_gain"]
    )
    minimum_text_accuracy = float(
        args.minimum_text_accuracy
        if args.minimum_text_accuracy is not None
        else decision_cfg["minimum_text_accuracy"]
    )

    rows = sorted(_read_jsonl(Path(args.data)), key=_row_key)
    if len(rows) < 2:
        raise SystemExit("learning curves require at least two rows")
    with open(args.elite_stats, encoding="utf-8") as file:
        elite_payload = yaml.safe_load(file) or {}
    elite_stats = elite_payload.get("features", elite_payload)

    pairs = build_gbm_pairs(
        rows,
        elite_stats,
        similarity_fn=lambda before, after: (1.0, {"method": "pair_constant"}),
    )
    folds = make_grouped_folds(rows, folds=fold_count, seed=seed)
    embeddings, embedding_info = _load_or_encode_embeddings(
        rows,
        cache_path=Path(args.embeddings_cache),
        model_name=model_name,
        device=device,
        batch_size=batch_size,
        refresh=args.refresh_embeddings,
    )

    full_feature_curve, full_feature_predictions = run_feature_learning_curve(
        rows,
        pairs,
        folds,
        train_sizes=train_sizes,
        transition_features=TRANSITION_FEATURES,
        seed=seed,
        model_name="feature_gbm_full",
    )
    no_target_features = tuple(
        feature for feature in TRANSITION_FEATURES if feature != "target_gain"
    )
    ablated_curve, ablated_predictions = run_feature_learning_curve(
        rows,
        pairs,
        folds,
        train_sizes=train_sizes,
        transition_features=no_target_features,
        seed=seed,
        model_name="feature_gbm_no_target_gain",
    )
    text_unconditioned_curve, text_unconditioned_predictions = run_text_learning_curve(
        rows,
        embeddings,
        folds,
        train_sizes=train_sizes,
        seed=seed,
        model_name="text_bge_m3_linear",
    )
    text_curve, text_predictions = run_text_learning_curve(
        rows,
        embeddings,
        folds,
        train_sizes=train_sizes,
        seed=seed,
        c=float(cfg.get("text_ranker", {}).get("c", 1.0)),
        condition_key=str(
            cfg.get("text_ranker", {}).get("condition_key", "reverse_action")
        ),
        model_name="text_bge_m3_action_linear",
    )
    decision = assess_generation_need(
        text_curve,
        plateau_max_gain=plateau_max_gain,
        minimum_accuracy=minimum_text_accuracy,
    )

    report = {
        "gate": "corruption_learning_curve",
        "config": args.config,
        "data": {
            "path": args.data,
            "pairs": len(rows),
            "essays": len({str(row["essay_id"]) for row in rows}),
            "operators": _counts(rows, "corruption_op"),
        },
        "split": {
            "folds": len(folds),
            "unit": "essay_id",
            "stratify": "corruption_op",
            "seed": seed,
            "train_sizes": train_sizes,
            "test_coverage": "each transition exactly once per curve point",
        },
        "embedding": embedding_info,
        "feature_gbm_full": full_feature_curve,
        "feature_gbm_no_target_gain": ablated_curve,
        "text_bge_m3_linear": text_unconditioned_curve,
        "text_bge_m3_action_linear": text_curve,
        "generation_decision": decision,
    }
    Path(args.report_out).write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    _write_jsonl(
        Path(args.predictions_out),
        [
            *full_feature_predictions,
            *ablated_predictions,
            *text_unconditioned_predictions,
            *text_predictions,
        ],
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _load_or_encode_embeddings(
    rows: Sequence[Mapping[str, Any]],
    *,
    cache_path: Path,
    model_name: str,
    device: str,
    batch_size: int,
    refresh: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    pair_ids = np.asarray([_pair_id(row) for row in rows])
    prompts = [
        prompt
        for row in rows
        for prompt in (
            state_prompt(row, str(row["text_before"])),
            state_prompt(row, str(row["text"])),
        )
    ]
    prompt_digest = _prompt_digest(prompts)
    snapshot = _local_huggingface_snapshot(model_name)
    if snapshot is None:
        raise RuntimeError(f"no complete local Hugging Face snapshot for {model_name}")
    snapshot_path = str(snapshot)
    if cache_path.is_file() and not refresh:
        cached = np.load(cache_path, allow_pickle=False)
        cached_ids = cached["pair_ids"]
        cached_model = str(cached["model"].item())
        cached_snapshot = (
            str(cached["snapshot"].item()) if "snapshot" in cached.files else ""
        )
        cached_digest = (
            str(cached["prompt_digest"].item())
            if "prompt_digest" in cached.files
            else ""
        )
        if (
            cached_model != model_name
            or cached_snapshot != snapshot_path
            or cached_digest != prompt_digest
            or not np.array_equal(cached_ids, pair_ids)
        ):
            raise ValueError("embedding cache does not match the requested data/model")
        embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
        return embeddings, {
            "model": model_name,
            "snapshot": snapshot_path,
            "device": str(cached["device"].item()),
            "cache": str(cache_path),
            "cache_validated": True,
            "dimensions": int(embeddings.shape[-1]),
            "state_prompts": int(embeddings.shape[0] * embeddings.shape[1]),
            "prompt_sha256": prompt_digest,
        }

    os.environ["FEAK_EMBEDDING_DEVICE"] = device
    model, model_info = _semantic_embedding_model(model_name)
    encoded = model.encode(
        prompts,
        batch_size=int(batch_size),
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    embeddings = np.asarray(encoded, dtype=np.float32).reshape(len(rows), 2, -1)
    np.savez_compressed(
        cache_path,
        pair_ids=pair_ids,
        model=np.asarray(model_name),
        snapshot=np.asarray(snapshot_path),
        device=np.asarray(device),
        prompt_digest=np.asarray(prompt_digest),
        embeddings=embeddings,
    )
    return embeddings, {
        **model_info,
        "cache": str(cache_path),
        "cache_validated": True,
        "dimensions": int(embeddings.shape[-1]),
        "state_prompts": len(prompts),
        "prompt_sha256": prompt_digest,
    }


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def _row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["essay_id"]), int(row["stage_k"])


def _pair_id(row: Mapping[str, Any]) -> str:
    return f"{row['essay_id']}:stage{int(row['stage_k'])}"


def _prompt_digest(prompts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    for prompt in prompts:
        encoded = prompt.encode("utf-8")
        digest.update(len(encoded).to_bytes(8, "big"))
        digest.update(encoded)
    return digest.hexdigest()


def _counts(rows: Sequence[Mapping[str, Any]], key: str) -> dict[str, int]:
    values: dict[str, int] = {}
    for row in rows:
        value = str(row[key])
        values[value] = values.get(value, 0) + 1
    return dict(sorted(values.items()))


if __name__ == "__main__":
    raise SystemExit(main())
