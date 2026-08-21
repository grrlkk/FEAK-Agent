"""One-epoch Bradley-Terry training loop and TVM metrics."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def pairwise_margin_loss(
    chosen_scores: Any,
    rejected_scores: Any,
    margins: Any,
) -> Any:
    import torch.nn.functional as functional

    return -functional.logsigmoid(chosen_scores - rejected_scores - margins).mean()


def set_training_seed(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_one_epoch(
    model: Any,
    train_loader: Any,
    *,
    learning_rate: float,
    weight_decay: float,
    gradient_accumulation_steps: int,
    warmup_steps: int,
    margin_per_stage: float,
    max_grad_norm: float,
    log_every: int = 10,
) -> dict[str, Any]:
    """Train exactly one pass over the pair dataset."""

    import torch
    from transformers import get_linear_schedule_with_warmup

    if gradient_accumulation_steps < 1:
        raise ValueError("gradient_accumulation_steps must be positive")
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        parameters,
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    update_steps = math.ceil(len(train_loader) / gradient_accumulation_steps)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=min(int(warmup_steps), max(0, update_steps - 1)),
        num_training_steps=update_steps,
    )
    model.train()
    optimizer.zero_grad(set_to_none=True)
    running_loss = 0.0
    examples = 0
    updates = 0
    history = []
    for step, batch in enumerate(train_loader, 1):
        chosen_scores = _forward_scores(model, batch["chosen"])
        rejected_scores = _forward_scores(model, batch["rejected"])
        margins = batch["stage_gap"].to(chosen_scores.device) * float(margin_per_stage)
        loss = pairwise_margin_loss(chosen_scores, rejected_scores, margins)
        if not torch.isfinite(loss):
            raise RuntimeError(f"non-finite TVM loss at batch {step}: {float(loss)}")
        window_start = ((step - 1) // gradient_accumulation_steps) * gradient_accumulation_steps
        window_batches = min(
            gradient_accumulation_steps,
            len(train_loader) - window_start,
        )
        (loss / window_batches).backward()
        batch_size = len(batch["pair_id"])
        running_loss += float(loss.detach()) * batch_size
        examples += batch_size
        should_update = step % gradient_accumulation_steps == 0 or step == len(train_loader)
        if should_update:
            torch.nn.utils.clip_grad_norm_(parameters, float(max_grad_norm))
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            updates += 1
            if updates % max(1, int(log_every)) == 0 or updates == update_steps:
                progress = {
                    "update": updates,
                    "examples": examples,
                    "mean_loss": running_loss / examples,
                    "learning_rate": float(scheduler.get_last_lr()[0]),
                }
                history.append(progress)
                print(
                    "TVM progress "
                    f"update={progress['update']}/{update_steps} "
                    f"examples={progress['examples']} "
                    f"mean_loss={progress['mean_loss']:.6f} "
                    f"lr={progress['learning_rate']:.3e}",
                    flush=True,
                )
    if updates != update_steps:
        raise AssertionError("TVM optimizer update count mismatch")
    return {
        "epochs": 1,
        "batches": len(train_loader),
        "examples": examples,
        "optimizer_updates": updates,
        "mean_loss": running_loss / examples,
        "history": history,
    }


def evaluate_pairwise(
    model: Any,
    data_loader: Any,
    *,
    margin_per_stage: float,
    normalization_mean: float = 0.0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    import torch

    model.eval()
    predictions = []
    losses = []
    with torch.inference_mode():
        for batch in data_loader:
            chosen = _forward_scores(model, batch["chosen"]).float()
            rejected = _forward_scores(model, batch["rejected"]).float()
            margins = batch["stage_gap"].to(chosen.device) * float(margin_per_stage)
            per_example_loss = -torch.nn.functional.logsigmoid(
                chosen - rejected - margins
            )
            losses.extend(float(value) for value in per_example_loss.cpu())
            for index, identifier in enumerate(batch["pair_id"]):
                chosen_score = float(chosen[index].cpu()) - float(normalization_mean)
                rejected_score = float(rejected[index].cpu()) - float(normalization_mean)
                gap = chosen_score - rejected_score
                predictions.append(
                    {
                        "pair_id": str(identifier),
                        "essay_id": str(batch["essay_id"][index]),
                        "corruption_op": str(batch["corruption_op"][index]),
                        "target_rubric": str(batch["target_rubric"][index]),
                        "stage_gap": int(batch["stage_gap"][index].item()),
                        "chosen_score": chosen_score,
                        "rejected_score": rejected_score,
                        "score_gap": gap,
                        "correct": gap > 0.0,
                        "tie": gap == 0.0,
                    }
                )
    return _metric_summary(predictions, losses), predictions


def score_normalization_mean(model: Any, data_loader: Any) -> float:
    import torch

    model.eval()
    total = 0.0
    count = 0
    with torch.inference_mode():
        for batch in data_loader:
            chosen = _forward_scores(model, batch["chosen"]).float()
            rejected = _forward_scores(model, batch["rejected"]).float()
            total += float(chosen.sum().cpu()) + float(rejected.sum().cpu())
            count += int(chosen.numel() + rejected.numel())
    if count == 0:
        raise ValueError("cannot normalize TVM scores on an empty dataset")
    return total / count


def save_adapter(
    model: Any,
    tokenizer: Any,
    output_dir: str | Path,
) -> None:
    target = Path(output_dir)
    target.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(target / "adapter", safe_serialization=True)
    tokenizer.save_pretrained(target / "tokenizer")


def _forward_scores(model: Any, encoding: Mapping[str, Any]) -> Any:
    device = next(model.parameters()).device
    inputs = {
        key: value.to(device, non_blocking=True)
        for key, value in encoding.items()
        if key in {"input_ids", "attention_mask"}
    }
    return model(**inputs).logits.view(-1)


def _metric_summary(
    predictions: Sequence[Mapping[str, Any]], losses: Sequence[float]
) -> dict[str, Any]:
    if not predictions:
        raise ValueError("TVM evaluation requires at least one pair")
    correct = sum(bool(row["correct"]) for row in predictions)
    groups: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        groups[str(row["corruption_op"])].append(row)
    ordered = sorted(predictions, key=lambda row: abs(float(row["score_gap"])))
    bins = []
    for indices in np.array_split(np.arange(len(ordered)), min(5, len(ordered))):
        selected = [ordered[int(index)] for index in indices]
        bins.append(
            {
                "pairs": len(selected),
                "mean_abs_gap": float(
                    np.mean([abs(float(row["score_gap"])) for row in selected])
                ),
                "accuracy": sum(bool(row["correct"]) for row in selected)
                / len(selected),
            }
        )
    all_scores = [
        float(score)
        for row in predictions
        for score in (row["chosen_score"], row["rejected_score"])
    ]
    return {
        "pairs": len(predictions),
        "correct": correct,
        "pairwise_accuracy": correct / len(predictions),
        "mean_loss": float(np.mean(losses)),
        "mean_score_gap": float(np.mean([row["score_gap"] for row in predictions])),
        "score_mean": float(np.mean(all_scores)),
        "score_std": float(np.std(all_scores)),
        "by_operator": {
            name: {
                "pairs": len(rows),
                "correct": sum(bool(row["correct"]) for row in rows),
                "pairwise_accuracy": sum(bool(row["correct"]) for row in rows)
                / len(rows),
                "mean_score_gap": float(np.mean([row["score_gap"] for row in rows])),
            }
            for name, rows in sorted(groups.items())
        },
        "confidence_bins_low_to_high": bins,
    }
