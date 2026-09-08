"""Schema constants and validation for Revision Verifier pilot samples."""

from __future__ import annotations

from typing import Any, Mapping


CANDIDATE_TYPES = (
    "correct_repair",
    "partial_repair",
    "wrong_target",
    "over_edit",
    "further_corruption",
    "no_edit",
)
LABEL_FIELDS = (
    "target_fulfillment",
    "preservation",
    "edit_appropriateness",
    "action_consistency",
)
LABEL_VALUES = ("pass", "partial", "fail")
CANDIDATE_SOURCES = (
    "trajectory_previous",
    "corruption_edit_replay",
    "llm",
    "trajectory_next",
    "trajectory_current",
)

REQUIRED_FIELDS = (
    "dataset_version",
    "sample_id",
    "essay_id",
    "chain_id",
    "state_id",
    "stage_k",
    "previous_state_id",
    "next_state_id",
    "source_transition_id",
    "question",
    "before_text",
    "after_text",
    "target_rubric",
    "intended_action",
    "intent",
    "corruption_type",
    "changed_spans",
    "candidate_type",
    "candidate_source",
    *LABEL_FIELDS,
    "weak_supervision",
    "label_source",
    "provenance",
)


RV_SAMPLE_JSON_SCHEMA: dict[str, Any] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "title": "FEAK-TC Revision Verifier pilot sample",
    "type": "object",
    "required": list(REQUIRED_FIELDS),
    "properties": {
        "dataset_version": {"type": "string"},
        "sample_id": {"type": "string", "minLength": 1},
        "essay_id": {"type": "string", "minLength": 1},
        "chain_id": {"type": "string", "minLength": 1},
        "state_id": {"type": "string", "minLength": 1},
        "stage_k": {"type": "integer", "minimum": 1},
        "previous_state_id": {"type": "string", "minLength": 1},
        "next_state_id": {"type": "string", "minLength": 1},
        "source_transition_id": {"type": "string", "minLength": 1},
        "question": {"type": "string"},
        "before_text": {"type": "string", "minLength": 1},
        "after_text": {"type": "string", "minLength": 1},
        "target_rubric": {"type": "string", "minLength": 1},
        "intended_action": {"type": "string", "minLength": 1},
        "intent": {"type": "string"},
        "corruption_type": {"type": "string", "minLength": 1},
        "changed_spans": {"type": "array", "items": {"type": "object"}},
        "candidate_type": {"enum": list(CANDIDATE_TYPES)},
        "candidate_source": {"enum": list(CANDIDATE_SOURCES)},
        "target_fulfillment": {"enum": list(LABEL_VALUES)},
        "preservation": {"enum": list(LABEL_VALUES)},
        "edit_appropriateness": {"enum": list(LABEL_VALUES)},
        "action_consistency": {"enum": list(LABEL_VALUES)},
        "weak_supervision": {"const": True},
        "label_source": {"type": "string", "minLength": 1},
        "provenance": {"type": "object"},
    },
    "additionalProperties": True,
}


def validate_rv_sample(sample: Mapping[str, Any]) -> None:
    """Validate the stable fields required by the RV pilot contract."""

    missing = [field for field in REQUIRED_FIELDS if field not in sample]
    if missing:
        raise ValueError(f"RV sample is missing fields: {missing}")
    for field in (
        "dataset_version",
        "sample_id",
        "essay_id",
        "chain_id",
        "state_id",
        "previous_state_id",
        "next_state_id",
        "source_transition_id",
        "before_text",
        "after_text",
        "target_rubric",
        "intended_action",
        "corruption_type",
        "label_source",
    ):
        if not isinstance(sample[field], str) or not sample[field].strip():
            raise ValueError(f"RV sample field {field} must be a non-empty string")
    candidate_type = str(sample["candidate_type"])
    if candidate_type not in CANDIDATE_TYPES:
        raise ValueError(f"unknown RV candidate_type: {candidate_type}")
    if sample["candidate_source"] not in CANDIDATE_SOURCES:
        raise ValueError(f"unknown RV candidate_source: {sample['candidate_source']}")
    if not isinstance(sample["stage_k"], int) or sample["stage_k"] < 1:
        raise ValueError("RV sample stage_k must be a positive integer")
    changed_spans = sample["changed_spans"]
    if not isinstance(changed_spans, list) or not changed_spans:
        raise ValueError("RV sample changed_spans must be a non-empty list")
    if not all(isinstance(span, Mapping) for span in changed_spans):
        raise ValueError("RV sample changed_spans entries must be mappings")
    if not isinstance(sample["provenance"], Mapping):
        raise ValueError("RV sample provenance must be a mapping")
    for field in LABEL_FIELDS:
        if sample[field] not in LABEL_VALUES:
            raise ValueError(f"unknown {field} label: {sample[field]}")
    if sample["weak_supervision"] is not True:
        raise ValueError("RV pilot labels must be marked as weak supervision")
