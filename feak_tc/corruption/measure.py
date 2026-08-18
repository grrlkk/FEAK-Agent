"""G1 diagnosis schema checks and strict target-rubric acceptance."""

from __future__ import annotations

from typing import Any, Mapping

from feak_tc.diagnose.constants import scores_to_rubric_dict


def validate_measurement_schema(
    row: Mapping[str, Any],
    schema: Mapping[str, Any],
) -> None:
    expected_rubrics = list(schema["rubrics"]["keys"])
    expected_features = list(schema["features"]["keys"])
    actual_rubrics = list(row["rubrics"])
    actual_features = list(row["features"])
    if set(actual_rubrics) != set(expected_rubrics):
        raise ValueError(
            f"rubric keys mismatch: expected {expected_rubrics}, got {actual_rubrics}"
        )
    if set(actual_features) != set(expected_features):
        raise ValueError(
            f"feature keys mismatch: expected {expected_features}, got {actual_features}"
        )


def measured_rubrics(
    row: Mapping[str, Any],
    score_basis: str,
) -> dict[str, float]:
    if score_basis == "rf_corrected":
        values = row.get("rf_corrected")
        if not isinstance(values, list):
            raise ValueError("rf_corrected score basis requested but values are missing")
        return scores_to_rubric_dict(values)
    if score_basis == "integer":
        return {str(key): float(value) for key, value in row["rubrics"].items()}
    raise ValueError(f"unknown measurement score basis: {score_basis}")


def evaluate_chain(
    chain: Mapping[str, Any],
    measurements: Mapping[tuple[str, int], Mapping[str, Any]],
    schema: Mapping[str, Any],
    measurement_cfg: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Evaluate target drops and reject measured cross-axis confounds."""

    score_basis = str(measurement_cfg.get("score_basis", "rf_corrected"))
    default_minimum = float(measurement_cfg.get("target_drop_min", 0.0))
    operator_minimums = dict(
        measurement_cfg.get("target_drop_min_by_operator", {})
    )
    record_id = str(chain["record_id"])
    evaluated: list[dict[str, Any]] = []

    for step_index, step in enumerate(chain["steps"]):
        operator = str(step["corruption_op"])
        minimum = float(operator_minimums.get(operator, default_minimum))
        before = measurements.get((record_id, step_index))
        after = measurements.get((record_id, step_index + 1))
        reason = ""
        target_drop: float | None = None
        quality_checks: dict[str, Any] = {}

        if before is None or after is None:
            accepted = False
            reason = "missing_measurement"
        else:
            validate_measurement_schema(before, schema)
            validate_measurement_schema(after, schema)
            before_scores = measured_rubrics(before, score_basis)
            after_scores = measured_rubrics(after, score_basis)
            target = str(step["target_rubric"])
            if target not in schema["rubrics"]["keys"]:
                raise ValueError(f"unknown target_rubric in step: {target}")
            target_drop = before_scores[target] - after_scores[target]
            accepted = target_drop > minimum
            reason = "target_rubric_decreased" if accepted else "target_rubric_not_decreased"
            if str(step["corruption_op"]) == "DELETE_SPECIFICS":
                delete_check = _delete_confound_check(
                    before_scores=before_scores,
                    after_scores=after_scores,
                    before_features=before["features"],
                    after_features=after["features"],
                    target_rubric=target,
                    cfg=measurement_cfg.get("delete_confound_guard", {}),
                )
                quality_checks["delete_confound"] = delete_check
                if accepted and delete_check["flagged"]:
                    accepted = False
                    reason = "delete_specifics_cross_axis_improvement"

        before_rubrics = measured_rubrics(before, score_basis) if before is not None else None
        after_rubrics = measured_rubrics(after, score_basis) if after is not None else None
        evaluated.append(
            {
                "essay_id": record_id,
                "chain_id": f"{record_id}:g1",
                "question": str(chain.get("question") or ""),
                "stage_k": step_index + 1,
                "corruption_op": step["corruption_op"],
                "target_rubric": step["target_rubric"],
                "target_features": [],
                "text_before": chain["states"][step_index],
                "text": chain["states"][step_index + 1],
                "measured_rubrics_before": before_rubrics,
                "measured_rubrics": after_rubrics,
                "measured_features_before": dict(before["features"]) if before is not None else None,
                "measured_features": dict(after["features"]) if after is not None else None,
                "generator": step["generator"],
                "requested_generator": step.get("requested_generator", step["generator"]),
                "fallback": bool(step.get("fallback", False)),
                "model": step.get("model"),
                "reasoning_effort": step.get("reasoning_effort"),
                "verbosity": step.get("verbosity"),
                "normalized": step["normalized"],
                "normalization": step["normalization"],
                "reverse_action": step["reverse_action"],
                "intent": step["intent"],
                "preserve_constraints": list(step["preserve_constraints"]),
                "preservation_check": dict(step.get("preservation_check", {})),
                "edits": list(step["edits"]),
                "score_basis": score_basis,
                "target_drop_min": minimum,
                "target_drop": target_drop,
                "accepted": accepted,
                "acceptance_reason": reason,
                "quality_checks": quality_checks,
            }
        )
    return evaluated


def _delete_confound_check(
    *,
    before_scores: Mapping[str, float],
    after_scores: Mapping[str, float],
    before_features: Mapping[str, Any],
    after_features: Mapping[str, Any],
    target_rubric: str,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    enabled = bool(cfg.get("enabled", True))
    nn_reduction = float(before_features.get("NN_repRatio", 0.0)) - float(
        after_features.get("NN_repRatio", 0.0)
    )
    lemma_gain = float(after_features.get("lemma_MATTR", 0.0)) - float(
        before_features.get("lemma_MATTR", 0.0)
    )
    non_target_gains = {
        rubric: float(after_scores[rubric]) - float(before_scores[rubric])
        for rubric in before_scores
        if rubric != target_rubric
    }
    max_non_target_gain = max(non_target_gains.values(), default=0.0)
    repetition_flag = (
        nn_reduction > float(cfg.get("nn_rep_ratio_reduction_min", 0.01))
        and lemma_gain > float(cfg.get("lemma_mattr_gain_min", 0.02))
    )
    rubric_flag = max_non_target_gain > float(
        cfg.get("max_non_target_rubric_gain", 0.3)
    )
    reasons = []
    if repetition_flag:
        reasons.append("repetition_metrics_improved")
    if rubric_flag:
        reasons.append("non_target_rubric_improved")
    return {
        "enabled": enabled,
        "flagged": enabled and bool(reasons),
        "reasons": reasons if enabled else [],
        "nn_rep_ratio_reduction": nn_reduction,
        "lemma_mattr_gain": lemma_gain,
        "max_non_target_rubric_gain": max_non_target_gain,
        "max_non_target_rubric": (
            max(non_target_gains, key=non_target_gains.get)
            if non_target_gains
            else None
        ),
    }
