"""G2 feature-only GBM sanity and blinded human-pair preparation."""

from __future__ import annotations

import math
import random
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from feak_tc.diagnose.base import select_weak_rubrics
from feak_tc.diagnose.constants import RUBRIC_FEATURE_MAP, RUBRIC_KEYS
from feak_tc.mvp.transition import edit_ratio, semantic_text_similarity


CATEGORICAL_FEATURES = ("action_type", "target_rubric")
NUMERIC_FEATURES = (
    "target_gain",
    "target_gap_reduction",
    "non_target_drop",
    "evidence_match",
    "edit_ratio",
    "goal_preservation",
    "emb_sim",
)
TRANSITION_FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

SimilarityFn = Callable[[str, str], tuple[float, Mapping[str, Any]]]


def build_gbm_pairs(
    audit_rows: Sequence[Mapping[str, Any]],
    elite_stats: Mapping[str, Mapping[str, float]],
    similarity_fn: SimilarityFn = semantic_text_similarity,
) -> list[dict[str, Any]]:
    """Build mirrored recovery/corruption examples from accepted G1 steps."""

    pairs = []
    for row in audit_rows:
        if not bool(row.get("accepted")):
            continue
        clean_text = str(row["text_before"])
        corrupted_text = str(row["text"])
        clean_rubrics = _float_mapping(row["measured_rubrics_before"])
        corrupted_rubrics = _float_mapping(row["measured_rubrics"])
        clean_features = _float_mapping(row["measured_features_before"])
        corrupted_features = _float_mapping(row["measured_features"])
        similarity, similarity_info = similarity_fn(corrupted_text, clean_text)
        shared = {
            "action_type": str(row["reverse_action"]),
            "target_rubric": str(row["target_rubric"]),
            "edit_ratio": float(edit_ratio(corrupted_text, clean_text)),
            "goal_preservation": float(similarity),
            "emb_sim": float(similarity),
        }
        chosen = _transition_features(
            before_text=corrupted_text,
            after_text=clean_text,
            before_rubrics=corrupted_rubrics,
            after_rubrics=clean_rubrics,
            before_features=corrupted_features,
            after_features=clean_features,
            edits=row["edits"],
            elite_stats=elite_stats,
            shared=shared,
        )
        rejected = _transition_features(
            before_text=clean_text,
            after_text=corrupted_text,
            before_rubrics=clean_rubrics,
            after_rubrics=corrupted_rubrics,
            before_features=clean_features,
            after_features=corrupted_features,
            edits=row["edits"],
            elite_stats=elite_stats,
            shared=shared,
        )
        pairs.append(
            {
                "pair_id": f"{row['essay_id']}:stage{row['stage_k']}",
                "essay_id": str(row["essay_id"]),
                "stage_k": int(row["stage_k"]),
                "stage_gap": 1,
                "corruption_op": str(row["corruption_op"]),
                "target_rubric": str(row["target_rubric"]),
                "target_drop": float(row["target_drop"]),
                "chosen": chosen,
                "rejected": rejected,
                "similarity": dict(similarity_info),
            }
        )
    return pairs


def run_grouped_gbm(
    pairs: Sequence[Mapping[str, Any]],
    *,
    folds: int = 5,
    seed: int = 20260722,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run essay-grouped cross-validation and return pairwise predictions."""

    if not pairs:
        raise ValueError("no accepted corruption pairs for G2")

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import GroupKFold
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    records = []
    pair_lookup = {}
    for pair in pairs:
        pair_id = str(pair["pair_id"])
        pair_lookup[pair_id] = pair
        for direction, label in (("chosen", 1), ("rejected", 0)):
            record = {
                "pair_id": pair_id,
                "essay_id": str(pair["essay_id"]),
                "direction": direction,
                "label": label,
                **dict(pair[direction]),
            }
            records.append(record)

    frame = pd.DataFrame(records)
    groups = frame["essay_id"]
    unique_groups = sorted(set(groups))
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError("G2 requires accepted pairs from at least two essays")

    predictions: dict[str, dict[str, Any]] = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(frame, frame["label"], groups),
        1,
    ):
        model = _gbm_pipeline(
            ColumnTransformer,
            GradientBoostingClassifier,
            OneHotEncoder,
            Pipeline,
            StandardScaler,
            seed,
        )
        train = frame.iloc[train_indices]
        test = frame.iloc[test_indices]
        model.fit(train[list(TRANSITION_FEATURES)], train["label"])
        probabilities = model.predict_proba(test[list(TRANSITION_FEATURES)])[:, 1]
        for (_, row), probability in zip(test.iterrows(), probabilities):
            pair_id = str(row["pair_id"])
            prediction = predictions.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "essay_id": str(row["essay_id"]),
                    "fold": fold,
                },
            )
            prediction[f"{row['direction']}_probability"] = float(probability)

    prediction_rows = []
    for pair_id in sorted(predictions):
        prediction = predictions[pair_id]
        chosen = float(prediction["chosen_probability"])
        rejected = float(prediction["rejected_probability"])
        source = pair_lookup[pair_id]
        prediction.update(
            {
                "corruption_op": source["corruption_op"],
                "target_rubric": source["target_rubric"],
                "target_drop": source["target_drop"],
                "stage_gap": int(source.get("stage_gap", 1)),
                "score_gap": chosen - rejected,
                "correct": chosen > rejected,
                "tie": chosen == rejected,
            }
        )
        prediction_rows.append(prediction)

    correct = sum(bool(row["correct"]) for row in prediction_rows)
    total = len(prediction_rows)
    accuracy = correct / total
    ci_low, ci_high = _wilson_interval(correct, total)
    p_value = _one_sided_binomial_p(correct, total)

    final_model = _gbm_pipeline(
        ColumnTransformer,
        GradientBoostingClassifier,
        OneHotEncoder,
        Pipeline,
        StandardScaler,
        seed,
    )
    final_model.fit(frame[list(TRANSITION_FEATURES)], frame["label"])
    names = list(final_model.named_steps["preprocess"].get_feature_names_out())
    importances = list(final_model.named_steps["model"].feature_importances_)
    ranked_importances = [
        {"feature": name, "importance": float(importance)}
        for name, importance in sorted(
            zip(names, importances),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    ambiguous_count = max(1, math.ceil(0.2 * total))
    ambiguous_ids = {
        row["pair_id"]
        for row in sorted(
            prediction_rows,
            key=lambda row: (abs(float(row["score_gap"])), row["pair_id"]),
        )[:ambiguous_count]
    }
    for row in prediction_rows:
        row["feature_ambiguous"] = row["pair_id"] in ambiguous_ids

    report = {
        "backend": "sklearn.ensemble.GradientBoostingClassifier",
        "transition_features": list(TRANSITION_FEATURES),
        "pairs": total,
        "essays": len(unique_groups),
        "folds": n_splits,
        "pairwise_correct": correct,
        "pairwise_accuracy": accuracy,
        "random_baseline": 0.5,
        "accuracy_wilson_95": {"low": ci_low, "high": ci_high},
        "one_sided_binomial_p": p_value,
        "significantly_above_random": ci_low > 0.5 and p_value < 0.05,
        "feature_ambiguous_pairs": ambiguous_count,
        "top_feature_importances": ranked_importances[:12],
        "by_operator": _prediction_group_stats(prediction_rows, "corruption_op"),
        "by_stage_gap": _prediction_group_stats(prediction_rows, "stage_gap"),
    }
    return report, prediction_rows


def run_grouped_lightgbm_ranker(
    pairs: Sequence[Mapping[str, Any]],
    *,
    folds: int = 5,
    seed: int = 20260722,
    transition_features: Sequence[str] = TRANSITION_FEATURES,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Run essay-grouped LightGBM pairwise ranking on adjacent transitions."""

    if not pairs:
        raise ValueError("no accepted corruption pairs for G2")
    selected_features = _validate_transition_features(transition_features)

    try:
        from lightgbm import LGBMRanker
    except ModuleNotFoundError as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "LightGBM is required for STEP 3; install the `g2` optional dependencies"
        ) from exc

    import pandas as pd
    from sklearn.compose import ColumnTransformer
    from sklearn.model_selection import GroupKFold
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    ordered_pairs = sorted(pairs, key=lambda pair: str(pair["pair_id"]))
    pair_frame = pd.DataFrame(
        {
            "pair_id": [str(pair["pair_id"]) for pair in ordered_pairs],
            "essay_id": [str(pair["essay_id"]) for pair in ordered_pairs],
        }
    )
    unique_groups = sorted(set(pair_frame["essay_id"]))
    n_splits = min(int(folds), len(unique_groups))
    if n_splits < 2:
        raise ValueError("G2 requires accepted pairs from at least two essays")

    predictions: dict[str, dict[str, Any]] = {}
    splitter = GroupKFold(n_splits=n_splits)
    for fold, (train_indices, test_indices) in enumerate(
        splitter.split(pair_frame, groups=pair_frame["essay_id"]),
        1,
    ):
        train_pairs = [ordered_pairs[index] for index in train_indices]
        test_pairs = [ordered_pairs[index] for index in test_indices]
        train_frame, train_labels, train_groups = _ranker_frame(train_pairs, pd)
        test_frame, _, _ = _ranker_frame(test_pairs, pd)
        preprocess = _ranker_preprocessor(
            ColumnTransformer,
            OneHotEncoder,
            StandardScaler,
            selected_features,
        )
        train_matrix = preprocess.fit_transform(train_frame[list(selected_features)])
        test_matrix = preprocess.transform(test_frame[list(selected_features)])
        model = _lightgbm_ranker(LGBMRanker, seed)
        model.fit(train_matrix, train_labels, group=train_groups)
        scores = model.predict(test_matrix)

        for row_index, (_, row) in enumerate(test_frame.iterrows()):
            pair_id = str(row["pair_id"])
            prediction = predictions.setdefault(
                pair_id,
                {
                    "pair_id": pair_id,
                    "essay_id": str(row["essay_id"]),
                    "fold": fold,
                },
            )
            prediction[f"{row['direction']}_score"] = float(scores[row_index])

    pair_lookup = {str(pair["pair_id"]): pair for pair in ordered_pairs}
    prediction_rows = []
    for pair_id in sorted(predictions):
        prediction = predictions[pair_id]
        chosen = float(prediction["chosen_score"])
        rejected = float(prediction["rejected_score"])
        source = pair_lookup[pair_id]
        prediction.update(
            {
                "corruption_op": source["corruption_op"],
                "target_rubric": source["target_rubric"],
                "target_drop": source["target_drop"],
                "stage_gap": int(source.get("stage_gap", 1)),
                "score_gap": chosen - rejected,
                "correct": chosen > rejected,
                "tie": chosen == rejected,
            }
        )
        prediction_rows.append(prediction)

    correct = sum(bool(row["correct"]) for row in prediction_rows)
    total = len(prediction_rows)
    accuracy = correct / total
    ci_low, ci_high = _wilson_interval(correct, total)
    p_value = _one_sided_binomial_p(correct, total)

    full_frame, full_labels, full_groups = _ranker_frame(ordered_pairs, pd)
    final_preprocess = _ranker_preprocessor(
        ColumnTransformer,
        OneHotEncoder,
        StandardScaler,
        selected_features,
    )
    full_matrix = final_preprocess.fit_transform(full_frame[list(selected_features)])
    final_model = _lightgbm_ranker(LGBMRanker, seed)
    final_model.fit(full_matrix, full_labels, group=full_groups)
    names = list(final_preprocess.get_feature_names_out())
    ranked_importances = [
        {"feature": name, "importance": float(importance)}
        for name, importance in sorted(
            zip(names, final_model.feature_importances_),
            key=lambda item: item[1],
            reverse=True,
        )
    ]

    ambiguous_count = max(1, math.ceil(0.2 * total))
    ambiguous_ids = {
        row["pair_id"]
        for row in sorted(
            prediction_rows,
            key=lambda row: (abs(float(row["score_gap"])), row["pair_id"]),
        )[:ambiguous_count]
    }
    for row in prediction_rows:
        row["feature_ambiguous"] = row["pair_id"] in ambiguous_ids

    report = {
        "backend": "lightgbm.LGBMRanker",
        "objective": "lambdarank",
        "input_scope": "accepted_adjacent_transitions",
        "transition_features": list(selected_features),
        "excluded_transition_features": [
            feature for feature in TRANSITION_FEATURES if feature not in selected_features
        ],
        "text_used_by_model": False,
        "pairs": total,
        "essays": len(unique_groups),
        "folds": n_splits,
        "split_unit": "essay_id",
        "pairwise_correct": correct,
        "pairwise_accuracy": accuracy,
        "random_baseline": 0.5,
        "accuracy_wilson_95": {"low": ci_low, "high": ci_high},
        "one_sided_binomial_p": p_value,
        "significantly_above_random": ci_low > 0.5 and p_value < 0.05,
        "feature_ambiguous_pairs": ambiguous_count,
        "top_feature_importances": ranked_importances[:12],
        "by_operator": _prediction_group_stats(prediction_rows, "corruption_op"),
        "by_stage_gap": _prediction_group_stats(prediction_rows, "stage_gap"),
    }
    return report, prediction_rows


def build_human_review_pairs(
    chains: Sequence[Mapping[str, Any]],
    audit_rows: Sequence[Mapping[str, Any]],
    *,
    count: int = 50,
    seed: int = 20260722,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Sample blinded A/B pairs whose intervening G1 steps all passed."""

    accepted = {
        (str(row["essay_id"]), int(row["stage_k"])): bool(row["accepted"])
        for row in audit_rows
    }
    audit = {
        (str(row["essay_id"]), int(row["stage_k"])): row
        for row in audit_rows
    }
    candidates = []
    for chain in sorted(chains, key=lambda item: str(item["record_id"])):
        essay_id = str(chain["record_id"])
        states = list(chain["states"])
        for cleaner_stage in range(len(states)):
            for corrupted_stage in range(cleaner_stage + 1, len(states)):
                if not all(
                    accepted.get((essay_id, stage), False)
                    for stage in range(cleaner_stage + 1, corrupted_stage + 1)
                ):
                    continue
                steps = [
                    audit[(essay_id, stage)]
                    for stage in range(cleaner_stage + 1, corrupted_stage + 1)
                ]
                candidates.append(
                    {
                        "essay_id": essay_id,
                        "question": str(chain.get("question") or "다음 글을 평가하세요."),
                        "cleaner_stage": cleaner_stage,
                        "corrupted_stage": corrupted_stage,
                        "cleaner_text": states[cleaner_stage],
                        "corrupted_text": states[corrupted_stage],
                        "stage_gap": corrupted_stage - cleaner_stage,
                        "operators": [str(step["corruption_op"]) for step in steps],
                        "target_rubrics": [str(step["target_rubric"]) for step in steps],
                        "target_drops": [float(step["target_drop"]) for step in steps],
                    }
                )

    if len(candidates) < count:
        raise ValueError(f"requested {count} human pairs but only {len(candidates)} are valid")
    rng = random.Random(seed)
    selected = rng.sample(candidates, count)
    review_rows = []
    key_rows = []
    for index, pair in enumerate(selected, 1):
        pair_id = f"G2H-{index:03d}"
        cleaner_is_a = bool(rng.getrandbits(1))
        review_rows.append(
            {
                "pair_id": pair_id,
                "essay_id": pair["essay_id"],
                "question": pair["question"],
                "text_a": pair["cleaner_text"] if cleaner_is_a else pair["corrupted_text"],
                "text_b": pair["corrupted_text"] if cleaner_is_a else pair["cleaner_text"],
                "preference": "",
                "notes": "",
            }
        )
        key_rows.append(
            {
                "pair_id": pair_id,
                "essay_id": pair["essay_id"],
                "expected_preference": "A" if cleaner_is_a else "B",
                "cleaner_stage": pair["cleaner_stage"],
                "corrupted_stage": pair["corrupted_stage"],
                "stage_gap": pair["stage_gap"],
                "operators": pair["operators"],
                "target_rubrics": pair["target_rubrics"],
                "target_drops": pair["target_drops"],
            }
        )
    summary = {
        "available_valid_pairs": len(candidates),
        "sampled_pairs": len(review_rows),
        "by_stage_gap": dict(
            sorted(Counter(row["stage_gap"] for row in key_rows).items())
        ),
        "unique_essays": len({row["essay_id"] for row in key_rows}),
        "seed": seed,
    }
    return review_rows, key_rows, summary


def evaluate_human_review(
    review_rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
    *,
    required: int = 50,
    threshold: float = 0.70,
) -> dict[str, Any]:
    """Evaluate filled A/B/TIE review rows without treating blanks as labels."""

    keys = {str(row["pair_id"]): str(row["expected_preference"]) for row in key_rows}
    valid = []
    for row in review_rows:
        preference = str(row.get("preference", "")).strip().upper()
        if preference in {"A", "B", "TIE"}:
            valid.append((str(row["pair_id"]), preference))
    matches = sum(preference == keys.get(pair_id) for pair_id, preference in valid)
    accuracy = matches / len(valid) if valid else None
    return {
        "required_pairs": required,
        "agreement_threshold": threshold,
        "labeled_pairs": len(valid),
        "matches": matches,
        "agreement": accuracy,
        "status": (
            "passed"
            if len(valid) >= required and accuracy is not None and accuracy >= threshold
            else "failed"
            if len(valid) >= required
            else "pending"
        ),
    }


def evaluate_two_human_reviews(
    rater_one_rows: Sequence[Mapping[str, Any]],
    rater_two_rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
    *,
    adjudication_rows: Sequence[Mapping[str, Any]] = (),
    threshold: float = 0.70,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Aggregate two independent blind reviews and require conflict resolution."""

    keys = {
        str(row["pair_id"]): str(row["expected_preference"]).strip().upper()
        for row in key_rows
    }
    if not keys or any(answer not in {"A", "B"} for answer in keys.values()):
        raise ValueError("human review key must contain A/B answers")

    rater_one = _review_preferences(rater_one_rows, set(keys), "rater_one")
    rater_two = _review_preferences(rater_two_rows, set(keys), "rater_two")
    adjudication = _review_preferences(
        adjudication_rows,
        set(keys),
        "adjudication",
        preference_key="adjudicated_preference",
    )

    overlap = sorted(set(rater_one) & set(rater_two))
    agreements = sum(rater_one[pair_id] == rater_two[pair_id] for pair_id in overlap)
    disagreements = []
    final_preferences = {}
    for pair_id in overlap:
        first = rater_one[pair_id]
        second = rater_two[pair_id]
        if first == second:
            final_preferences[pair_id] = first
            continue
        adjudicated = adjudication.get(pair_id)
        if adjudicated is not None:
            final_preferences[pair_id] = adjudicated
        else:
            disagreements.append(
                {
                    "pair_id": pair_id,
                    "rater_one_preference": first,
                    "rater_two_preference": second,
                    "adjudicated_preference": "",
                    "notes": "",
                }
            )

    expected = len(keys)
    both_complete = len(rater_one) == expected and len(rater_two) == expected
    final_matches = sum(
        preference == keys[pair_id]
        for pair_id, preference in final_preferences.items()
    )
    final_agreement = (
        final_matches / len(final_preferences) if final_preferences else None
    )
    if not both_complete:
        status = "pending_raters"
    elif disagreements:
        status = "pending_adjudication"
    elif len(final_preferences) != expected:
        status = "pending_adjudication"
    elif final_agreement is not None and final_agreement >= threshold:
        status = "passed"
    else:
        status = "failed"

    report = {
        "expected_pairs": expected,
        "agreement_threshold": threshold,
        "rater_one": _individual_human_result(rater_one, keys, expected, threshold),
        "rater_two": _individual_human_result(rater_two, keys, expected, threshold),
        "overlap_labeled_pairs": len(overlap),
        "inter_rater_exact_agreements": agreements,
        "inter_rater_exact_agreement": agreements / len(overlap) if overlap else None,
        "unresolved_disagreements": len(disagreements),
        "adjudicated_disagreements": sum(
            pair_id in adjudication
            and rater_one[pair_id] != rater_two[pair_id]
            for pair_id in overlap
        ),
        "final_labeled_pairs": len(final_preferences),
        "final_matches": final_matches,
        "final_agreement": final_agreement,
        "status": status,
    }
    return report, disagreements


def _review_preferences(
    rows: Sequence[Mapping[str, Any]],
    expected_pair_ids: set[str],
    source: str,
    *,
    preference_key: str = "preference",
) -> dict[str, str]:
    preferences = {}
    seen = set()
    for row in rows:
        pair_id = str(row.get("pair_id", ""))
        if pair_id not in expected_pair_ids:
            raise ValueError(f"{source} contains unknown pair_id: {pair_id}")
        if pair_id in seen:
            raise ValueError(f"{source} contains duplicate pair_id: {pair_id}")
        seen.add(pair_id)
        preference = str(row.get(preference_key, "")).strip().upper()
        if preference in {"A", "B", "TIE"}:
            preferences[pair_id] = preference
    return preferences


def _individual_human_result(
    preferences: Mapping[str, str],
    keys: Mapping[str, str],
    expected: int,
    threshold: float,
) -> dict[str, Any]:
    matches = sum(answer == keys[pair_id] for pair_id, answer in preferences.items())
    agreement = matches / len(preferences) if preferences else None
    return {
        "labeled_pairs": len(preferences),
        "matches": matches,
        "agreement": agreement,
        "complete": len(preferences) == expected,
        "meets_threshold": (
            len(preferences) == expected
            and agreement is not None
            and agreement >= threshold
        ),
    }


def _transition_features(
    *,
    before_text: str,
    after_text: str,
    before_rubrics: Mapping[str, float],
    after_rubrics: Mapping[str, float],
    before_features: Mapping[str, float],
    after_features: Mapping[str, float],
    edits: Sequence[Mapping[str, Any]],
    elite_stats: Mapping[str, Mapping[str, float]],
    shared: Mapping[str, Any],
) -> dict[str, Any]:
    target = str(shared["target_rubric"])
    non_target_drop = max(
        0.0,
        max(
            before_rubrics[key] - after_rubrics[key]
            for key in RUBRIC_KEYS
            if key != target
        ),
    )
    weak = select_weak_rubrics(before_rubrics)
    target_span = _matching_span(before_text, edits)
    evidence_match = (0.6 if target in weak else 0.0) + (0.3 if target_span else 0.0)
    return {
        **dict(shared),
        "target_gain": float(after_rubrics[target] - before_rubrics[target]),
        "target_gap_reduction": _target_gap_reduction(
            before_features,
            after_features,
            target,
            elite_stats,
        ),
        "non_target_drop": float(non_target_drop),
        "evidence_match": min(1.0, evidence_match),
    }


def _target_gap_reduction(
    before: Mapping[str, float],
    after: Mapping[str, float],
    target_rubric: str,
    elite_stats: Mapping[str, Mapping[str, float]],
) -> float:
    reductions = []
    for feature in RUBRIC_FEATURE_MAP.get(target_rubric, []):
        band = elite_stats.get(feature)
        if not band:
            continue
        low = float(band["low"])
        high = float(band["high"])
        scale = max(high - low, 1e-6)
        reductions.append(
            _band_gap(float(before.get(feature, 0.0)), low, high, scale)
            - _band_gap(float(after.get(feature, 0.0)), low, high, scale)
        )
    if not reductions:
        return 0.0
    return max(-1.0, min(1.0, sum(reductions) / len(reductions)))


def _band_gap(value: float, low: float, high: float, scale: float) -> float:
    if value < low:
        return (low - value) / scale
    if value > high:
        return (value - high) / scale
    return 0.0


def _matching_span(text: str, edits: Sequence[Mapping[str, Any]]) -> str:
    for edit in edits:
        for key in ("target_span", "text"):
            span = str(edit.get(key, "")).strip()
            if span and span in text:
                return span
    return ""


def _float_mapping(value: Any) -> dict[str, float]:
    if not isinstance(value, Mapping):
        raise ValueError("G2 requires measured rubric and feature mappings")
    return {str(key): float(number) for key, number in value.items()}


def _gbm_pipeline(
    column_transformer: Any,
    classifier: Any,
    encoder: Any,
    pipeline: Any,
    scaler: Any,
    seed: int,
) -> Any:
    preprocess = column_transformer(
        [
            (
                "categorical",
                encoder(handle_unknown="ignore", sparse_output=False),
                list(CATEGORICAL_FEATURES),
            ),
            ("numeric", scaler(), list(NUMERIC_FEATURES)),
        ]
    )
    return pipeline(
        [
            ("preprocess", preprocess),
            (
                "model",
                classifier(
                    n_estimators=100,
                    learning_rate=0.05,
                    max_depth=2,
                    random_state=seed,
                ),
            ),
        ]
    )


def _ranker_frame(
    pairs: Sequence[Mapping[str, Any]],
    pandas_module: Any,
) -> tuple[Any, list[int], list[int]]:
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
    return pandas_module.DataFrame(records), labels, [2] * len(pairs)


def _ranker_preprocessor(
    column_transformer: Any,
    encoder: Any,
    scaler: Any,
    transition_features: Sequence[str],
) -> Any:
    categorical = [
        feature for feature in CATEGORICAL_FEATURES if feature in transition_features
    ]
    numeric = [
        feature for feature in NUMERIC_FEATURES if feature in transition_features
    ]
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


def _validate_transition_features(features: Sequence[str]) -> tuple[str, ...]:
    selected = tuple(dict.fromkeys(str(feature) for feature in features))
    if not selected:
        raise ValueError("at least one transition feature is required")
    unknown = [feature for feature in selected if feature not in TRANSITION_FEATURES]
    if unknown:
        raise ValueError(f"unknown transition features: {unknown}")
    return selected


def _lightgbm_ranker(ranker: Any, seed: int) -> Any:
    return ranker(
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


def _wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
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


def _one_sided_binomial_p(successes: int, total: int) -> float:
    return sum(math.comb(total, value) for value in range(successes, total + 1)) / (2**total)


def _prediction_group_stats(
    rows: Sequence[Mapping[str, Any]],
    key: str,
) -> dict[str, dict[str, Any]]:
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row[key])].append(row)
    return {
        name: {
            "pairs": len(group),
            "correct": sum(bool(row["correct"]) for row in group),
            "accuracy": sum(bool(row["correct"]) for row in group) / len(group),
        }
        for name, group in sorted(grouped.items())
    }
