"""Synthetic weak-label assignment for controlled RV candidate types."""

from __future__ import annotations

from typing import Any, Mapping

from .schema import CANDIDATE_TYPES, LABEL_FIELDS, LABEL_VALUES


def validate_label_mapping(mapping: Mapping[str, Any]) -> None:
    """Require a complete candidate-type mapping with the fixed label vocabulary."""

    missing_types = [name for name in CANDIDATE_TYPES if name not in mapping]
    if missing_types:
        raise ValueError(f"weak label mapping is missing candidate types: {missing_types}")
    for candidate_type in CANDIDATE_TYPES:
        labels = mapping[candidate_type]
        if not isinstance(labels, Mapping):
            raise ValueError(f"weak labels for {candidate_type} must be a mapping")
        missing_fields = [field for field in LABEL_FIELDS if field not in labels]
        if missing_fields:
            raise ValueError(
                f"weak labels for {candidate_type} are missing: {missing_fields}"
            )
        for field in LABEL_FIELDS:
            if labels[field] not in LABEL_VALUES:
                raise ValueError(
                    f"invalid {field} label for {candidate_type}: {labels[field]}"
                )


def build_weak_labels(
    candidate_type: str,
    mapping: Mapping[str, Any],
    *,
    label_source: str,
) -> dict[str, Any]:
    """Return candidate-type supervision, explicitly marked as non-ground-truth."""

    validate_label_mapping(mapping)
    if candidate_type not in CANDIDATE_TYPES:
        raise ValueError(f"unknown RV candidate_type: {candidate_type}")
    labels = mapping[candidate_type]
    return {
        **{field: str(labels[field]) for field in LABEL_FIELDS},
        "weak_supervision": True,
        "label_source": label_source,
    }
