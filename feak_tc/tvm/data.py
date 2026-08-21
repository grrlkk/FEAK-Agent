"""Data preparation and prompt encoding for pairwise TVM training."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from feak_tc.corruption.g2 import NUMERIC_FEATURES, build_gbm_pairs
from feak_tc.corruption.learning_curve import make_grouped_folds


SCORER_DERIVED_FEATURES = ("target_gain", "non_target_drop")
TVM_FEATURE_VARIANTS = {
    "full": tuple(NUMERIC_FEATURES),
    "scorer_free": tuple(
        feature for feature in NUMERIC_FEATURES if feature not in SCORER_DERIVED_FEATURES
    ),
}


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    source = Path(path)
    with source.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_key(row: Mapping[str, Any]) -> tuple[str, int]:
    return str(row["essay_id"]), int(row["stage_k"])


def pair_id(row: Mapping[str, Any]) -> str:
    return f"{row['essay_id']}:stage{int(row['stage_k'])}"


def load_pair_similarities(
    path: str | Path,
    rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, float], dict[str, Any]]:
    """Load verified BGE state embeddings and derive pair cosine similarities."""

    cache_path = Path(path)
    cached = np.load(cache_path, allow_pickle=False)
    required = {"pair_ids", "embeddings", "model", "snapshot", "prompt_digest"}
    missing = sorted(required - set(cached.files))
    if missing:
        raise ValueError(f"similarity cache is missing fields: {missing}")
    cached_ids = [str(value) for value in cached["pair_ids"].tolist()]
    expected_ids = [pair_id(row) for row in rows]
    if cached_ids != expected_ids:
        raise ValueError("similarity cache pair IDs do not match sorted TVM rows")
    embeddings = np.asarray(cached["embeddings"], dtype=np.float32)
    if embeddings.ndim != 3 or embeddings.shape[:2] != (len(rows), 2):
        raise ValueError("similarity embeddings must have shape (pairs, 2, dimensions)")
    norms = np.linalg.norm(embeddings, axis=2)
    if np.any(norms <= 0):
        raise ValueError("similarity cache contains zero-length embeddings")
    normalized = embeddings / norms[:, :, None]
    values = np.sum(normalized[:, 0, :] * normalized[:, 1, :], axis=1)
    similarities = {
        identifier: float(value) for identifier, value in zip(expected_ids, values)
    }
    return similarities, {
        "path": str(cache_path),
        "sha256": file_sha256(cache_path),
        "model": str(cached["model"].item()),
        "snapshot": str(cached["snapshot"].item()),
        "prompt_sha256": str(cached["prompt_digest"].item()),
        "pairs": len(similarities),
        "similarity_min": float(values.min()),
        "similarity_mean": float(values.mean()),
        "similarity_max": float(values.max()),
    }


def build_tvm_pairs(
    rows: Sequence[Mapping[str, Any]],
    elite_stats: Mapping[str, Mapping[str, float]],
    pair_similarities: Mapping[str, float],
) -> list[dict[str, Any]]:
    """Build chosen/rejected transition records with measured feature values."""

    missing = [pair_id(row) for row in rows if pair_id(row) not in pair_similarities]
    if missing:
        raise ValueError(f"missing pair similarities for {len(missing)} rows")
    feature_pairs = build_gbm_pairs(
        rows,
        elite_stats,
        similarity_fn=lambda _before, _after: (1.0, {"method": "deferred_cache"}),
    )
    by_id = {str(pair["pair_id"]): pair for pair in feature_pairs}
    result = []
    for row in rows:
        identifier = pair_id(row)
        pair = by_id[identifier]
        similarity = float(pair_similarities[identifier])
        for direction in ("chosen", "rejected"):
            pair[direction]["goal_preservation"] = similarity
            pair[direction]["emb_sim"] = similarity
        result.append(
            {
                "pair_id": identifier,
                "essay_id": str(row["essay_id"]),
                "stage_k": int(row["stage_k"]),
                "stage_gap": int(pair.get("stage_gap", 1)),
                "corruption_op": str(row["corruption_op"]),
                "question": str(row.get("question") or "").strip(),
                "action_type": str(row["reverse_action"]),
                "intent": str(row.get("intent") or "").strip(),
                "target_rubric": str(row["target_rubric"]),
                "chosen": {
                    "before_text": str(row["text"]),
                    "after_text": str(row["text_before"]),
                    "features": dict(pair["chosen"]),
                },
                "rejected": {
                    "before_text": str(row["text_before"]),
                    "after_text": str(row["text"]),
                    "features": dict(pair["rejected"]),
                },
            }
        )
    return result


def make_tvm_split(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int = 10,
    test_fold: int = 1,
    validation_fold: int = 2,
    seed: int = 20260821,
) -> dict[str, Any]:
    """Create a fixed essay-disjoint stratified train/validation/test split."""

    grouped_folds = make_grouped_folds(rows, folds=folds, seed=seed)
    by_number = {int(fold["fold"]): fold for fold in grouped_folds}
    if test_fold == validation_fold:
        raise ValueError("test and validation folds must differ")
    if test_fold not in by_number or validation_fold not in by_number:
        raise ValueError("requested split fold does not exist")
    test_indices = sorted(int(index) for index in by_number[test_fold]["test_indices"])
    validation_indices = sorted(
        int(index) for index in by_number[validation_fold]["test_indices"]
    )
    held_out = set(test_indices) | set(validation_indices)
    train_indices = [index for index in range(len(rows)) if index not in held_out]
    split = {
        "train": train_indices,
        "validation": validation_indices,
        "test": test_indices,
        "metadata": {
            "method": "StratifiedGroupKFold held-out partitions",
            "stratify": "corruption_op",
            "group": "essay_id",
            "folds": len(grouped_folds),
            "test_fold": int(test_fold),
            "validation_fold": int(validation_fold),
            "seed": int(seed),
        },
    }
    _validate_tvm_split(rows, split)
    split["summary"] = {
        name: _split_summary(rows, indices)
        for name, indices in split.items()
        if name in {"train", "validation", "test"}
    }
    return split


def transition_prompt(
    pair: Mapping[str, Any],
    direction: str,
    *,
    feature_variant: str,
) -> dict[str, str]:
    """Build named prompt sections without exposing the preference label."""

    if direction not in {"chosen", "rejected"}:
        raise ValueError(f"unknown pair direction: {direction}")
    if feature_variant not in TVM_FEATURE_VARIANTS:
        raise ValueError(f"unknown TVM feature variant: {feature_variant}")
    transition = pair[direction]
    features = transition["features"]
    feature_lines = []
    for feature in TVM_FEATURE_VARIANTS[feature_variant]:
        feature_lines.append(f"{feature}={float(features[feature]):+.6f}")
    return {
        "instruction": (
            "다음 수정 transition이 과제와 수정 계획에 비추어 얼마나 좋은지 평가한다. "
            "점수나 정답을 생성하지 말고 마지막 상태 표현만 사용한다."
        ),
        "question": str(pair["question"]),
        "plan": (
            f"action={pair['action_type']}\n"
            f"intent={pair['intent']}\n"
            f"target_rubric={pair['target_rubric']}"
        ),
        "before": str(transition["before_text"]),
        "after": str(transition["after_text"]),
        "features": "\n".join(feature_lines),
        "suffix": "[가치 표현]",
    }


def encode_prompt_sections(
    tokenizer: Any,
    sections: Mapping[str, str],
    *,
    max_length: int,
) -> dict[str, list[int]]:
    """Tokenize while allocating independent budgets to before/after text."""

    if max_length < 256:
        raise ValueError("TVM max_length must be at least 256")
    encode = lambda text: list(tokenizer.encode(text, add_special_tokens=False))
    marker = encode("\n[…중략…]\n")
    fixed_chunks = {
        "instruction": encode(f"[지시]\n{sections['instruction']}\n"),
        "question_header": encode("[과제]\n"),
        "plan": encode(f"\n[수정 계획]\n{sections['plan']}\n"),
        "before_header": encode("[수정 전]\n"),
        "after_header": encode("\n[수정 후]\n"),
        "features": encode(f"\n[transition features]\n{sections['features']}\n"),
        "suffix": encode(str(sections["suffix"])),
    }
    question = encode(str(sections["question"]))
    before = encode(str(sections["before"]))
    after = encode(str(sections["after"]))
    bos = [tokenizer.bos_token_id] if tokenizer.bos_token_id is not None else []
    eos = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
    fixed_length = len(bos) + len(eos) + sum(len(chunk) for chunk in fixed_chunks.values())
    minimum_text_budget = 64
    question_budget = min(len(question), max(32, max_length // 10))
    available = max_length - fixed_length - question_budget
    if available < minimum_text_budget * 2:
        question_budget = max(8, question_budget - (minimum_text_budget * 2 - available))
        available = max_length - fixed_length - question_budget
    if available < 32:
        raise ValueError("TVM prompt metadata leaves no room for transition text")
    before_budget = available // 2
    after_budget = available - before_budget
    input_ids = [
        *bos,
        *fixed_chunks["instruction"],
        *fixed_chunks["question_header"],
        *_clip_middle(question, question_budget, marker),
        *fixed_chunks["plan"],
        *fixed_chunks["before_header"],
        *_clip_middle(before, before_budget, marker),
        *fixed_chunks["after_header"],
        *_clip_middle(after, after_budget, marker),
        *fixed_chunks["features"],
        *fixed_chunks["suffix"],
        *eos,
    ]
    if len(input_ids) > max_length:
        raise AssertionError("balanced TVM encoding exceeded max_length")
    return {"input_ids": input_ids, "attention_mask": [1] * len(input_ids)}


class PairwisePromptDataset:
    """Lazy tokenized chosen/rejected transition dataset."""

    def __init__(
        self,
        pairs: Sequence[Mapping[str, Any]],
        indices: Sequence[int],
        tokenizer: Any,
        *,
        feature_variant: str,
        max_length: int,
    ) -> None:
        self.pairs = pairs
        self.indices = [int(index) for index in indices]
        self.tokenizer = tokenizer
        self.feature_variant = feature_variant
        self.max_length = int(max_length)

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, position: int) -> dict[str, Any]:
        pair = self.pairs[self.indices[position]]
        chosen = encode_prompt_sections(
            self.tokenizer,
            transition_prompt(pair, "chosen", feature_variant=self.feature_variant),
            max_length=self.max_length,
        )
        rejected = encode_prompt_sections(
            self.tokenizer,
            transition_prompt(pair, "rejected", feature_variant=self.feature_variant),
            max_length=self.max_length,
        )
        return {
            "pair_id": str(pair["pair_id"]),
            "essay_id": str(pair["essay_id"]),
            "corruption_op": str(pair["corruption_op"]),
            "target_rubric": str(pair["target_rubric"]),
            "stage_gap": int(pair.get("stage_gap", 1)),
            "chosen": chosen,
            "rejected": rejected,
        }


class PairwiseCollator:
    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def __call__(self, examples: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        import torch

        chosen = self.tokenizer.pad(
            [example["chosen"] for example in examples], return_tensors="pt"
        )
        rejected = self.tokenizer.pad(
            [example["rejected"] for example in examples], return_tensors="pt"
        )
        return {
            "pair_id": [str(example["pair_id"]) for example in examples],
            "essay_id": [str(example["essay_id"]) for example in examples],
            "corruption_op": [str(example["corruption_op"]) for example in examples],
            "target_rubric": [str(example["target_rubric"]) for example in examples],
            "stage_gap": torch.tensor(
                [int(example["stage_gap"]) for example in examples], dtype=torch.float32
            ),
            "chosen": chosen,
            "rejected": rejected,
        }


def prompt_sha256(
    pairs: Sequence[Mapping[str, Any]],
    *,
    feature_variant: str,
) -> str:
    digest = hashlib.sha256()
    for pair in pairs:
        for direction in ("chosen", "rejected"):
            sections = transition_prompt(pair, direction, feature_variant=feature_variant)
            for key in ("instruction", "question", "plan", "before", "after", "features", "suffix"):
                encoded = sections[key].encode("utf-8")
                digest.update(len(encoded).to_bytes(8, "big"))
                digest.update(encoded)
    return digest.hexdigest()


def split_sha256(split: Mapping[str, Any]) -> str:
    """Hash the immutable split assignment and its construction metadata."""

    payload = {
        "train": [int(index) for index in split["train"]],
        "validation": [int(index) for index in split["validation"]],
        "test": [int(index) for index in split["test"]],
        "metadata": dict(split["metadata"]),
    }
    encoded = json.dumps(
        payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _clip_middle(tokens: Sequence[int], budget: int, marker: Sequence[int]) -> list[int]:
    if budget <= 0:
        return []
    values = list(tokens)
    if len(values) <= budget:
        return values
    if budget <= len(marker) + 2:
        return values[:budget]
    remaining = budget - len(marker)
    left = remaining // 2
    right = remaining - left
    return [*values[:left], *marker, *values[-right:]]


def _validate_tvm_split(
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int] | Mapping[str, Any]],
) -> None:
    names = ("train", "validation", "test")
    sets = {name: {int(index) for index in split[name]} for name in names}
    comparisons = (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    )
    if any(sets[left] & sets[right] for left, right in comparisons):
        raise ValueError("TVM split indices overlap")
    if set().union(*sets.values()) != set(range(len(rows))):
        raise ValueError("TVM split does not cover every row")
    groups = {
        name: {str(rows[index]["essay_id"]) for index in indices}
        for name, indices in sets.items()
    }
    if any(groups[left] & groups[right] for left, right in comparisons):
        raise ValueError("TVM split leaks essay groups")


def _split_summary(
    rows: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> dict[str, Any]:
    selected = [rows[int(index)] for index in indices]
    return {
        "pairs": len(selected),
        "essays": len({str(row["essay_id"]) for row in selected}),
        "operators": dict(sorted(Counter(str(row["corruption_op"]) for row in selected).items())),
    }
