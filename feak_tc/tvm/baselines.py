"""Fixed-split Stage-1 baselines comparable to TVM validation/test runs."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping, Sequence

import numpy as np

from feak_tc.corruption.g2 import CATEGORICAL_FEATURES
from feak_tc.tvm.data import TVM_FEATURE_VARIANTS


def run_fixed_baselines(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    embeddings: Any,
    split: Mapping[str, Sequence[int]],
    *,
    seed: int,
    heuristic_config: Mapping[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit on train only and evaluate the frozen baselines on val and test."""

    if len(rows) != len(pairs):
        raise ValueError("baseline rows and pairs must align")
    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[:2] != (len(rows), 2):
        raise ValueError("baseline embeddings must have shape (pairs, 2, dimensions)")
    train_indices = [int(index) for index in split["train"]]
    reports: dict[str, Any] = {}
    predictions = []

    for variant in ("full", "scorer_free"):
        name = f"feature_lightgbm_{variant}"
        scorer = _fit_feature_ranker(
            pairs,
            train_indices,
            features=(*CATEGORICAL_FEATURES, *TVM_FEATURE_VARIANTS[variant]),
            seed=seed,
        )
        report, rows_out = _evaluate_scorer(name, rows, pairs, split, scorer)
        report["feature_variant"] = variant
        reports[name] = report
        predictions.extend(rows_out)

    bge_scorer, bge_info = _fit_action_bge(rows, vectors, train_indices, seed=seed)
    report, rows_out = _evaluate_gap_scorer(
        "bge_m3_action_linear", rows, split, bge_scorer
    )
    report.update(bge_info)
    reports["bge_m3_action_linear"] = report
    predictions.extend(rows_out)

    for name, scorer in (
        (
            "heuristic_full",
            lambda pair: (
                _heuristic_score(
                    pair["chosen"]["features"], heuristic_config, scorer_free=False
                ),
                _heuristic_score(
                    pair["rejected"]["features"], heuristic_config, scorer_free=False
                ),
            ),
        ),
        (
            "heuristic_scorer_free",
            lambda pair: (
                _heuristic_score(
                    pair["chosen"]["features"], heuristic_config, scorer_free=True
                ),
                _heuristic_score(
                    pair["rejected"]["features"], heuristic_config, scorer_free=True
                ),
            ),
        ),
        (
            "immediate_target_gain",
            lambda pair: (
                float(pair["chosen"]["features"]["target_gain"]),
                float(pair["rejected"]["features"]["target_gain"]),
            ),
        ),
    ):
        report, rows_out = _evaluate_scorer(name, rows, pairs, split, scorer)
        reports[name] = report
        predictions.extend(rows_out)

    return {
        "train_pairs": len(train_indices),
        "fit_split": "train only",
        "evaluation": reports,
    }, predictions


def _fit_feature_ranker(
    pairs: Sequence[Mapping[str, Any]],
    train_indices: Sequence[int],
    *,
    features: Sequence[str],
    seed: int,
) -> Any:
    import pandas as pd
    from lightgbm import LGBMRanker
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    selected = [str(feature) for feature in features]
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in selected]
    numeric = [feature for feature in selected if feature not in categorical]
    transform = ColumnTransformer(
        [
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
            ("numeric", StandardScaler(), numeric),
        ]
    )
    records = []
    labels = []
    for index in train_indices:
        pair = pairs[index]
        for direction, label in (("chosen", 1), ("rejected", 0)):
            records.append(dict(pair[direction]["features"]))
            labels.append(label)
    frame = pd.DataFrame(records)
    matrix = transform.fit_transform(frame[selected])
    model = LGBMRanker(
        objective="lambdarank",
        metric="ndcg",
        label_gain=[0, 1],
        n_estimators=100,
        learning_rate=0.05,
        max_depth=2,
        num_leaves=4,
        min_child_samples=2,
        random_state=seed,
        verbosity=-1,
    )
    model.fit(matrix, labels, group=[2] * len(train_indices))

    def score(pair: Mapping[str, Any]) -> tuple[float, float]:
        test = pd.DataFrame(
            [dict(pair["chosen"]["features"]), dict(pair["rejected"]["features"])]
        )
        values = model.predict(transform.transform(test[selected]))
        return float(values[0]), float(values[1])

    return score


def _fit_action_bge(
    rows: Sequence[Mapping[str, Any]],
    embeddings: np.ndarray,
    train_indices: Sequence[int],
    *,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    from sklearn.linear_model import LogisticRegression

    differences = embeddings[:, 0, :] - embeddings[:, 1, :]
    categories = sorted({str(row["reverse_action"]) for row in rows})
    category_index = {category: index for index, category in enumerate(categories)}
    conditioned = np.zeros(
        (len(rows), differences.shape[1] * len(categories)), dtype=np.float32
    )
    for index, row in enumerate(rows):
        start = category_index[str(row["reverse_action"])] * differences.shape[1]
        conditioned[index, start : start + differences.shape[1]] = differences[index]
    train_diff = conditioned[list(train_indices)]
    matrix = np.concatenate([train_diff, -train_diff], axis=0)
    labels = np.concatenate(
        [
            np.ones(len(train_diff), dtype=np.int8),
            np.zeros(len(train_diff), dtype=np.int8),
        ]
    )
    model = LogisticRegression(
        C=1.0,
        fit_intercept=False,
        max_iter=2000,
        random_state=seed,
        solver="liblinear",
    )
    model.fit(matrix, labels)
    return (
        lambda index: float(model.decision_function(conditioned[[int(index)]])[0]),
        {
            "backend": "frozen_BGE-M3_plus_action-conditioned_logistic_regression",
            "condition_categories": categories,
            "dimensions": int(conditioned.shape[1]),
        },
    )


def _evaluate_scorer(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int]],
    scorer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = []
    summaries = {}
    for split_name in ("validation", "test"):
        predictions = []
        for index in split[split_name]:
            chosen, rejected = scorer(pairs[int(index)])
            predictions.append(
                _prediction(name, split_name, rows[int(index)], chosen - rejected)
            )
        summaries[split_name] = _summary(predictions)
        output.extend(predictions)
    return summaries, output


def _evaluate_gap_scorer(
    name: str,
    rows: Sequence[Mapping[str, Any]],
    split: Mapping[str, Sequence[int]],
    scorer: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    output = []
    summaries = {}
    for split_name in ("validation", "test"):
        predictions = [
            _prediction(name, split_name, rows[int(index)], scorer(int(index)))
            for index in split[split_name]
        ]
        summaries[split_name] = _summary(predictions)
        output.extend(predictions)
    return summaries, output


def _heuristic_score(
    features: Mapping[str, Any],
    config: Mapping[str, Any],
    *,
    scorer_free: bool,
) -> float:
    weights = config["weights"]
    rubric_range = max(1.0, float(config["rubric_score_range"]))
    score = (
        float(weights["target_gap_reduction"])
        * float(features["target_gap_reduction"])
        + float(weights["evidence_match"]) * float(features["evidence_match"])
        + float(weights["goal_preservation"])
        * float(features["goal_preservation"])
        + float(weights["edit_ratio"]) * float(features["edit_ratio"])
    )
    if not scorer_free:
        score += (
            float(weights["target_gain"])
            * float(features["target_gain"])
            / rubric_range
            + float(weights["non_target_drop"])
            * float(features["non_target_drop"])
            / rubric_range
        )
    return float(score)


def _prediction(
    model: str,
    split_name: str,
    row: Mapping[str, Any],
    score_gap: float,
) -> dict[str, Any]:
    return {
        "model": model,
        "split": split_name,
        "pair_id": f"{row['essay_id']}:stage{int(row['stage_k'])}",
        "essay_id": str(row["essay_id"]),
        "corruption_op": str(row["corruption_op"]),
        "target_rubric": str(row["target_rubric"]),
        "score_gap": float(score_gap),
        "correct": float(score_gap) > 0.0,
        "tie": float(score_gap) == 0.0,
    }


def _summary(predictions: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        grouped[str(row["corruption_op"])].append(row)
    correct = sum(bool(row["correct"]) for row in predictions)
    return {
        "pairs": len(predictions),
        "correct": correct,
        "pairwise_accuracy": correct / len(predictions),
        "mean_score_gap": float(np.mean([row["score_gap"] for row in predictions])),
        "by_operator": {
            operator: {
                "pairs": len(group),
                "correct": sum(bool(row["correct"]) for row in group),
                "pairwise_accuracy": sum(bool(row["correct"]) for row in group)
                / len(group),
            }
            for operator, group in sorted(grouped.items())
        },
    }
