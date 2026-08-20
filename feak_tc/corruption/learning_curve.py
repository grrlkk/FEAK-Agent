"""Essay-grouped learning curves for corruption transition baselines."""

from __future__ import annotations

import math
import random
from collections import defaultdict
from typing import Any, Mapping, Sequence

from feak_tc.corruption.g2 import (
    CATEGORICAL_FEATURES,
    NUMERIC_FEATURES,
    TRANSITION_FEATURES,
)


def make_grouped_folds(
    rows: Sequence[Mapping[str, Any]],
    *,
    folds: int = 5,
    seed: int = 20260820,
) -> list[dict[str, Any]]:
    """Create deterministic operator-stratified folds split by essay."""

    if not rows:
        raise ValueError("learning curves require at least one transition")
    from sklearn.model_selection import StratifiedGroupKFold

    groups = [str(row["essay_id"]) for row in rows]
    operators = [str(row["corruption_op"]) for row in rows]
    unique_groups = sorted(set(groups))
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError("learning curves require at least two essays")

    splitter = StratifiedGroupKFold(
        n_splits=n_splits,
        shuffle=True,
        random_state=seed,
    )
    result = []
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(rows, operators, groups),
        1,
    ):
        train = sorted(int(index) for index in train_indices)
        test = sorted(int(index) for index in test_indices)
        train_groups = sorted({groups[index] for index in train})
        rng = random.Random(seed + fold * 1009)
        rng.shuffle(train_groups)
        result.append(
            {
                "fold": fold,
                "train_indices": train,
                "test_indices": test,
                "train_group_order": train_groups,
            }
        )
    _validate_folds(result, groups)
    return result


def select_nested_train_indices(
    fold: Mapping[str, Any],
    rows: Sequence[Mapping[str, Any]],
    requested_pairs: int | None,
) -> list[int]:
    """Select complete essay groups in a stable nested order."""

    full_indices = [int(index) for index in fold["train_indices"]]
    if requested_pairs is None or requested_pairs >= len(full_indices):
        return full_indices
    if requested_pairs < 1:
        raise ValueError("requested training size must be positive")

    by_group: dict[str, list[int]] = defaultdict(list)
    for index in full_indices:
        by_group[str(rows[index]["essay_id"])].append(index)
    selected: list[int] = []
    for group in fold["train_group_order"]:
        selected.extend(by_group[str(group)])
        if len(selected) >= requested_pairs:
            break
    return sorted(selected)


def run_feature_learning_curve(
    rows: Sequence[Mapping[str, Any]],
    pairs: Sequence[Mapping[str, Any]],
    folds: Sequence[Mapping[str, Any]],
    *,
    train_sizes: Sequence[int],
    transition_features: Sequence[str] = TRANSITION_FEATURES,
    seed: int = 20260820,
    model_name: str = "feature_gbm",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit LightGBM pairwise rankers on nested essay-grouped subsets."""

    selected_features = _validate_features(transition_features)
    if len(rows) != len(pairs):
        raise ValueError("row and feature-pair counts must match")

    from lightgbm import LGBMRanker
    from sklearn.compose import ColumnTransformer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    points = []
    all_predictions = []
    for requested in [*train_sizes, None]:
        predictions = []
        fold_train_sizes = []
        for fold in folds:
            train_indices = select_nested_train_indices(fold, rows, requested)
            test_indices = [int(index) for index in fold["test_indices"]]
            train_pairs = [pairs[index] for index in train_indices]
            test_pairs = [pairs[index] for index in test_indices]
            train_frame, train_labels, train_groups = _feature_frame(train_pairs)
            test_frame, _, _ = _feature_frame(test_pairs)
            preprocess = _feature_preprocessor(
                ColumnTransformer,
                OneHotEncoder,
                StandardScaler,
                selected_features,
            )
            train_matrix = preprocess.fit_transform(train_frame[list(selected_features)])
            test_matrix = preprocess.transform(test_frame[list(selected_features)])
            model = LGBMRanker(
                objective="lambdarank",
                metric="ndcg",
                label_gain=[0, 1],
                n_estimators=100,
                learning_rate=0.05,
                max_depth=2,
                num_leaves=4,
                min_child_samples=2,
                random_state=seed + int(fold["fold"]),
                verbosity=-1,
            )
            model.fit(train_matrix, train_labels, group=train_groups)
            scores = model.predict(test_matrix)
            fold_train_sizes.append(len(train_indices))
            for offset, pair in enumerate(test_pairs):
                chosen_score = float(scores[offset * 2])
                rejected_score = float(scores[offset * 2 + 1])
                predictions.append(
                    _prediction_row(
                        pair,
                        fold=int(fold["fold"]),
                        model_name=model_name,
                        point=_point_name(requested),
                        score_gap=chosen_score - rejected_score,
                    )
                )
        summary = _curve_point_summary(
            predictions,
            requested=requested,
            fold_train_sizes=fold_train_sizes,
        )
        points.append(summary)
        all_predictions.extend(predictions)

    return {
        "model": model_name,
        "backend": "lightgbm.LGBMRanker",
        "text_used_by_model": False,
        "transition_features": list(selected_features),
        "excluded_transition_features": [
            feature for feature in TRANSITION_FEATURES if feature not in selected_features
        ],
        "points": points,
    }, all_predictions


def run_text_learning_curve(
    rows: Sequence[Mapping[str, Any]],
    embeddings: Any,
    folds: Sequence[Mapping[str, Any]],
    *,
    train_sizes: Sequence[int],
    seed: int = 20260820,
    c: float = 1.0,
    condition_key: str | None = None,
    model_name: str = "text_bge_m3_linear",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Fit an antisymmetric linear ranker over frozen clean/corrupt embeddings."""

    import numpy as np
    from sklearn.linear_model import LogisticRegression

    vectors = np.asarray(embeddings, dtype=np.float32)
    if vectors.ndim != 3 or vectors.shape[0] != len(rows) or vectors.shape[1] != 2:
        raise ValueError("embeddings must have shape (rows, 2, dimensions)")
    differences = vectors[:, 0, :] - vectors[:, 1, :]
    categories: list[str] = []
    if condition_key is not None:
        categories = sorted({str(row[condition_key]) for row in rows})
        category_index = {category: index for index, category in enumerate(categories)}
        conditioned = np.zeros(
            (len(rows), differences.shape[1] * len(categories)),
            dtype=np.float32,
        )
        for row_index, row in enumerate(rows):
            block = category_index[str(row[condition_key])]
            start = block * differences.shape[1]
            conditioned[row_index, start : start + differences.shape[1]] = differences[
                row_index
            ]
        differences = conditioned

    points = []
    all_predictions = []
    for requested in [*train_sizes, None]:
        predictions = []
        fold_train_sizes = []
        for fold in folds:
            train_indices = select_nested_train_indices(fold, rows, requested)
            test_indices = [int(index) for index in fold["test_indices"]]
            train_diff = differences[train_indices]
            train_matrix = np.concatenate([train_diff, -train_diff], axis=0)
            train_labels = np.concatenate(
                [
                    np.ones(len(train_diff), dtype=np.int8),
                    np.zeros(len(train_diff), dtype=np.int8),
                ]
            )
            model = LogisticRegression(
                C=float(c),
                fit_intercept=False,
                max_iter=2000,
                random_state=seed + int(fold["fold"]),
                solver="liblinear",
            )
            model.fit(train_matrix, train_labels)
            score_gaps = model.decision_function(differences[test_indices])
            fold_train_sizes.append(len(train_indices))
            for index, score_gap in zip(test_indices, score_gaps):
                predictions.append(
                    _prediction_row(
                        rows[index],
                        fold=int(fold["fold"]),
                        model_name=model_name,
                        point=_point_name(requested),
                        score_gap=float(score_gap),
                    )
                )
        summary = _curve_point_summary(
            predictions,
            requested=requested,
            fold_train_sizes=fold_train_sizes,
        )
        points.append(summary)
        all_predictions.extend(predictions)

    return {
        "model": model_name,
        "backend": "frozen_sentence_encoder_plus_pairwise_logistic_regression",
        "text_used_by_model": True,
        "pair_representation": "clean_state_embedding_minus_corrupt_state_embedding",
        "condition_key": condition_key,
        "condition_categories": categories,
        "pair_dimensions": int(differences.shape[1]),
        "fit_intercept": False,
        "c": float(c),
        "points": points,
    }, all_predictions


def state_prompt(row: Mapping[str, Any], text: str) -> str:
    """Build the state representation used by the frozen text baseline."""

    return (
        f"[과제]\n{str(row.get('question') or '').strip()}\n"
        f"[개선 행동]\n{str(row.get('reverse_action') or '').strip()}\n"
        f"[목표 평가 기준]\n{str(row.get('target_rubric') or '').strip()}\n"
        f"[글]\n{text.strip()}"
    )


def assess_generation_need(
    text_curve: Mapping[str, Any],
    *,
    plateau_max_gain: float = 0.015,
    minimum_accuracy: float = 0.75,
) -> dict[str, Any]:
    """Apply a predeclared stop/generate/investigate rule to the final slope."""

    points = list(text_curve.get("points") or [])
    if len(points) < 2:
        raise ValueError("generation decision requires at least two curve points")
    previous = points[-2]
    final = points[-1]
    gain = float(final["accuracy"]) - float(previous["accuracy"])
    if gain > plateau_max_gain:
        decision = "generate_more"
        reason = "text accuracy is still rising above the plateau threshold"
    elif float(final["accuracy"]) >= minimum_accuracy:
        decision = "stop_generation"
        reason = "text accuracy is adequate and the final learning-curve segment is flat"
    else:
        decision = "investigate_model_or_construct_before_generation"
        reason = "the curve is flat below the adequacy threshold"
    return {
        "decision": decision,
        "reason": reason,
        "previous_point": previous["point"],
        "final_point": final["point"],
        "previous_accuracy": previous["accuracy"],
        "final_accuracy": final["accuracy"],
        "final_gain": gain,
        "plateau_max_gain": float(plateau_max_gain),
        "minimum_accuracy": float(minimum_accuracy),
    }


def _feature_frame(
    pairs: Sequence[Mapping[str, Any]],
) -> tuple[Any, list[int], list[int]]:
    import pandas as pd

    records = []
    labels = []
    for pair in pairs:
        for direction, label in (("chosen", 1), ("rejected", 0)):
            records.append(
                {
                    "pair_id": str(pair["pair_id"]),
                    "essay_id": str(pair["essay_id"]),
                    "direction": direction,
                    **dict(pair[direction]),
                }
            )
            labels.append(label)
    return pd.DataFrame(records), labels, [2] * len(pairs)


def _feature_preprocessor(
    column_transformer: Any,
    encoder: Any,
    scaler: Any,
    features: Sequence[str],
) -> Any:
    categorical = [feature for feature in CATEGORICAL_FEATURES if feature in features]
    numeric = [feature for feature in NUMERIC_FEATURES if feature in features]
    transformers = []
    if categorical:
        transformers.append(
            (
                "categorical",
                encoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            )
        )
    if numeric:
        transformers.append(("numeric", scaler(), numeric))
    return column_transformer(transformers)


def _curve_point_summary(
    predictions: Sequence[Mapping[str, Any]],
    *,
    requested: int | None,
    fold_train_sizes: Sequence[int],
) -> dict[str, Any]:
    total = len(predictions)
    correct = sum(bool(row["correct"]) for row in predictions)
    if total == 0:
        raise ValueError("curve point has no held-out predictions")
    by_operator: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    by_fold: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for row in predictions:
        by_operator[str(row["corruption_op"])].append(row)
        by_fold[int(row["fold"])].append(row)
    low, high = _wilson_interval(correct, total)
    return {
        "point": _point_name(requested),
        "requested_train_pairs": requested,
        "train_pairs": {
            "min": min(fold_train_sizes),
            "mean": sum(fold_train_sizes) / len(fold_train_sizes),
            "max": max(fold_train_sizes),
        },
        "evaluation_pairs": total,
        "correct": correct,
        "accuracy": correct / total,
        "accuracy_wilson_95": {"low": low, "high": high},
        "by_operator": {
            name: {
                "pairs": len(group),
                "correct": sum(bool(row["correct"]) for row in group),
                "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
            }
            for name, group in sorted(by_operator.items())
        },
        "by_fold": {
            str(fold): {
                "pairs": len(group),
                "correct": sum(bool(row["correct"]) for row in group),
                "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
            }
            for fold, group in sorted(by_fold.items())
        },
    }


def _prediction_row(
    pair: Mapping[str, Any],
    *,
    fold: int,
    model_name: str,
    point: str,
    score_gap: float,
) -> dict[str, Any]:
    return {
        "model": model_name,
        "point": point,
        "fold": fold,
        "pair_id": str(
            pair.get("pair_id")
            or f"{pair['essay_id']}:stage{int(pair['stage_k'])}"
        ),
        "essay_id": str(pair["essay_id"]),
        "corruption_op": str(pair["corruption_op"]),
        "target_rubric": str(pair["target_rubric"]),
        "score_gap": float(score_gap),
        "correct": float(score_gap) > 0.0,
        "tie": float(score_gap) == 0.0,
    }


def _validate_folds(folds: Sequence[Mapping[str, Any]], groups: Sequence[str]) -> None:
    test_coverage = []
    for fold in folds:
        train = {int(index) for index in fold["train_indices"]}
        test = {int(index) for index in fold["test_indices"]}
        if train & test:
            raise ValueError("fold train/test indices overlap")
        train_groups = {groups[index] for index in train}
        test_groups = {groups[index] for index in test}
        if train_groups & test_groups:
            raise ValueError("fold train/test essay groups overlap")
        test_coverage.extend(test)
    if sorted(test_coverage) != list(range(len(groups))):
        raise ValueError("folds must evaluate every row exactly once")


def _validate_features(features: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(feature) for feature in features))
    if not selected:
        raise ValueError("at least one transition feature is required")
    unknown = [feature for feature in selected if feature not in TRANSITION_FEATURES]
    if unknown:
        raise ValueError(f"unknown transition features: {unknown}")
    return selected


def _point_name(requested: int | None) -> str:
    return "full" if requested is None else str(int(requested))


def _wilson_interval(
    successes: int,
    total: int,
    z: float = 1.959963984540054,
) -> tuple[float, float]:
    if total == 0:
        return 0.0, 0.0
    proportion = successes / total
    denominator = 1.0 + z * z / total
    center = (proportion + z * z / (2.0 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1.0 - proportion) / total
            + z * z / (4.0 * total * total)
        )
        / denominator
    )
    return max(0.0, center - margin), min(1.0, center + margin)
