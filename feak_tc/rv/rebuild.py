"""Selective candidate regeneration for the second RV data pilot."""

from __future__ import annotations

import hashlib
import json
import re
import time
from difflib import SequenceMatcher
from typing import Any, Callable, Mapping, Sequence

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json

from .schema import LABEL_FIELDS


REGENERATION_PROMPT_VERSION = "rv-selective-regeneration-v2"
STRONG_RETRY_PROMPT_VERSION = "rv-selective-regeneration-v2-strong-retry"
REGENERATED_TYPES = ("wrong_target", "over_edit")

_SYSTEM_PROMPT = """You generate controlled Korean essay revisions for Revision Verifier data.
The requested error pattern must be clearly observable from the resulting full essay. Preserve the
writing question and return only strict JSON. Do not explain the answer outside the JSON field."""

_TEXT_SCHEMA = {
    "type": "object",
    "properties": {"revised_text": {"type": "string", "minLength": 1}},
    "required": ["revised_text"],
    "additionalProperties": False,
}


def select_regeneration_rows(
    rows: Sequence[Mapping[str, Any]],
    *,
    over_edit_actions: Sequence[str] = ("ADD_DETAIL",),
) -> list[dict[str, Any]]:
    """Return only the v1 candidates whose text must be regenerated."""

    actions = set(over_edit_actions)
    selected = [
        dict(row)
        for row in rows
        if row.get("candidate_type") == "wrong_target"
        or (
            row.get("candidate_type") == "over_edit"
            and row.get("intended_action") in actions
        )
    ]
    return sorted(selected, key=lambda row: str(row["sample_id"]))


def build_state_groups(
    rows: Sequence[Mapping[str, Any]],
) -> dict[str, dict[str, dict[str, Any]]]:
    groups: dict[str, dict[str, dict[str, Any]]] = {}
    for raw_row in rows:
        row = dict(raw_row)
        state_id = str(row["state_id"])
        candidate_type = str(row["candidate_type"])
        if candidate_type in groups.setdefault(state_id, {}):
            raise ValueError(f"duplicate {state_id}:{candidate_type}")
        groups[state_id][candidate_type] = row
    required = {
        "correct_repair",
        "partial_repair",
        "wrong_target",
        "over_edit",
        "further_corruption",
        "no_edit",
    }
    incomplete = {
        state_id: sorted(required - set(group))
        for state_id, group in groups.items()
        if set(group) != required
    }
    if incomplete:
        raise ValueError(f"incomplete RV state groups: {incomplete}")
    return groups


def generate_replacement(
    row: Mapping[str, Any],
    state_group: Mapping[str, Mapping[str, Any]],
    llm_cfg: Mapping[str, Any],
    *,
    generation_variant: str = "base",
    request_fn: Callable[..., dict[str, Any]] = request_json,
) -> tuple[str, dict[str, float]]:
    """Generate one replacement with deterministic semantic validation."""

    candidate_type = str(row["candidate_type"])
    if candidate_type not in REGENERATED_TYPES:
        raise ValueError(f"unsupported regeneration type: {candidate_type}")
    attempts = int(llm_cfg.get("max_attempts", 4))
    if attempts < 1:
        raise ValueError("max_attempts must be positive")
    model = str(llm_cfg["model"])
    reasoning = llm_cfg.get("reasoning", {})
    text_cfg = llm_cfg.get("text", {})
    validation_cfg = llm_cfg.get("validation", {})
    previous_error = ""
    errors: list[str] = []
    for attempt in range(1, attempts + 1):
        prompt = build_regeneration_prompt(
            row,
            state_group,
            require_safe_span=attempt >= 3,
            generation_variant=generation_variant,
        )
        if previous_error:
            prompt += (
                "\n\nThe previous candidate was rejected. Regenerate it and fix this exact "
                f"problem: {previous_error}"
            )
        try:
            payload = request_fn(
                system=_SYSTEM_PROMPT,
                user=prompt,
                model=model,
                temperature=None,
                reasoning_effort=_optional_string(reasoning.get("effort")),
                verbosity=_optional_string(text_cfg.get("verbosity")),
                json_schema=_TEXT_SCHEMA,
                response_format_name=f"rv_v2_{candidate_type}",
                timeout=(
                    float(llm_cfg["timeout"])
                    if llm_cfg.get("timeout") is not None
                    else None
                ),
            )
            text = str(payload["revised_text"]).strip()
            metrics = validate_replacement(
                row,
                state_group,
                text,
                validation_cfg,
                generation_variant=generation_variant,
            )
            return text, metrics
        except (KeyError, LLMResponseError, LLMUnavailable, ValueError) as exc:
            previous_error = str(exc)
            errors.append(f"attempt {attempt}: {exc}")
            if attempt < attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"replacement generation failed for {row.get('sample_id')}: "
        + " | ".join(errors)
    )


def build_regeneration_prompt(
    row: Mapping[str, Any],
    state_group: Mapping[str, Mapping[str, Any]],
    *,
    require_safe_span: bool = False,
    generation_variant: str = "base",
) -> str:
    candidate_type = str(row["candidate_type"])
    current = str(row["before_text"])
    reference = str(state_group["correct_repair"]["after_text"])
    payload = {
        "question": row.get("question", ""),
        "target_rubric": row["target_rubric"],
        "intended_action": row["intended_action"],
        "intent": row.get("intent", ""),
        "corruption_type": row["corruption_type"],
        "known_corruption_edits": row.get("changed_spans", []),
        "current_corrupted_state": current,
        "exact_target_repair_reference": reference,
    }
    if candidate_type == "wrong_target":
        wrong_spec = wrong_target_spec(str(row["target_rubric"]))
        payload["required_non_target_revision"] = wrong_spec
        safe_spans = _safe_non_target_sentences(row, current)
        if require_safe_span and safe_spans:
            payload["required_sentence_to_revise"] = safe_spans[0]
        if generation_variant == "strong":
            payload["required_non_target_sentences_to_revise"] = safe_spans[:3]
        instruction = (
            "Create a clear WRONG_TARGET candidate. Leave the specified target defect unresolved. "
            "Do not perform the intended action. Instead, successfully make a substantive but "
            "limited revision for required_non_target_revision. This must not be NO_EDIT or a tiny "
            "synonym replacement: revise at least one complete non-target sentence. Preserve every "
            "inserted/repeated corruption fragment verbatim and in its current order when present. "
            "For DELETE_SPECIFICS, do not restore or paraphrase the deleted target details."
        )
        if require_safe_span and safe_spans:
            instruction += (
                " You must replace required_sentence_to_revise with a noticeably clearer complete "
                "sentence that preserves its meaning. Do not edit the known corruption fragments."
            )
        if generation_variant == "strong":
            instruction += (
                " Make the wrong-target signal unmistakable. Revise every listed "
                "required_non_target_sentences_to_revise, or add two complete supporting sentences "
                "to a non-target claim when fewer than two safe sentences are listed. The target "
                "defect must remain clearly visible."
            )
    elif row.get("intended_action") == "ADD_DETAIL":
        payload["required_target_sentences"] = [
            str(span.get("target_span") or "")
            for span in row.get("changed_spans", [])
            if span.get("operation") == "delete" and span.get("target_span")
        ]
        instruction = (
            "Create a clear OVER_EDIT candidate for ADD_DETAIL. Start from the exact target repair "
            "reference and keep every required_target_sentence verbatim so the missing details are "
            "fully restored. Then make at least two substantial unnecessary changes outside those "
            "sentences. At least one must delete or distort a non-target claim, or add an unsupported "
            "claim. The result must remain recognizable as the same essay but must be clearly less "
            "preserving and less minimal than the exact repair."
        )
        if generation_variant == "strong":
            instruction += (
                " Make the over-edit signal unmistakable: unnecessarily rewrite at least three "
                "non-target sentences and add or distort at least one non-target claim."
            )
    else:
        raise ValueError(
            f"over_edit regeneration is only defined for ADD_DETAIL: {row.get('sample_id')}"
        )
    return (
        instruction
        + "\nReturn the complete Korean essay in revised_text. Do not return commentary.\n\n"
        + json.dumps(payload, ensure_ascii=False, indent=2)
    )


def validate_replacement(
    row: Mapping[str, Any],
    state_group: Mapping[str, Mapping[str, Any]],
    text: str,
    validation_cfg: Mapping[str, Any],
    *,
    generation_variant: str = "base",
) -> dict[str, float]:
    """Apply conservative structural checks before any LLM quality judgment."""

    value = text.strip()
    if not value:
        raise ValueError("candidate is empty")
    current = str(row["before_text"])
    reference = str(state_group["correct_repair"]["after_text"])
    reserved = {str(candidate["after_text"]) for candidate in state_group.values()}
    if value in reserved:
        raise ValueError("candidate copies an existing candidate or trajectory state")
    length_ratio = len(value) / max(1, len(current))
    minimum_length = float(validation_cfg.get("min_length_ratio", 0.60))
    maximum_length = float(validation_cfg.get("max_length_ratio", 2.00))
    if not minimum_length <= length_ratio <= maximum_length:
        raise ValueError(
            f"length ratio {length_ratio:.3f} is outside "
            f"[{minimum_length:.3f}, {maximum_length:.3f}]"
        )
    current_diff = _edit_ratio(current, value)
    reference_diff = _edit_ratio(reference, value)
    candidate_type = str(row["candidate_type"])
    if candidate_type == "wrong_target":
        minimum_diff = float(
            validation_cfg.get(
                "strong_wrong_target_min_edit_ratio"
                if generation_variant == "strong"
                else "wrong_target_min_edit_ratio",
                0.05 if generation_variant == "strong" else 0.02,
            )
        )
        maximum_diff = float(validation_cfg.get("wrong_target_max_edit_ratio", 0.40))
        if not minimum_diff <= current_diff <= maximum_diff:
            raise ValueError(
                f"wrong_target edit ratio {current_diff:.3f} is outside "
                f"[{minimum_diff:.3f}, {maximum_diff:.3f}]"
            )
        _validate_target_defect_preserved(row, current, value)
    elif candidate_type == "over_edit":
        minimum_ref_diff = float(
            validation_cfg.get(
                "strong_over_edit_min_reference_edit_ratio"
                if generation_variant == "strong"
                else "over_edit_min_reference_edit_ratio",
                0.10 if generation_variant == "strong" else 0.08,
            )
        )
        minimum_changed_chars = int(
            validation_cfg.get(
                "strong_over_edit_min_changed_chars"
                if generation_variant == "strong"
                else "over_edit_min_changed_chars",
                100 if generation_variant == "strong" else 70,
            )
        )
        estimated_changed_chars = reference_diff * max(len(reference), len(value))
        if (
            reference_diff < minimum_ref_diff
            and estimated_changed_chars < minimum_changed_chars
        ):
            raise ValueError(
                f"over_edit differs from exact repair by only {reference_diff:.3f}; "
                f"minimum is {minimum_ref_diff:.3f} or {minimum_changed_chars} changed chars"
            )
        required = [
            str(span.get("target_span") or "")
            for span in row.get("changed_spans", [])
            if span.get("operation") == "delete" and span.get("target_span")
        ]
        missing = [fragment for fragment in required if fragment not in value]
        if missing:
            raise ValueError(
                f"over_edit failed to preserve {len(missing)} required target repair sentences"
            )
    return {
        "length_ratio": round(length_ratio, 6),
        "edit_ratio_from_current": round(current_diff, 6),
        "edit_ratio_from_reference": round(reference_diff, 6),
    }


def apply_replacements(
    rows: Sequence[Mapping[str, Any]],
    replacements: Mapping[str, Mapping[str, Any]],
    *,
    dataset_version: str,
    model: str,
) -> list[dict[str, Any]]:
    """Create a non-trainable v2 staging dataset while preserving v1."""

    output = []
    for source in rows:
        row = dict(source)
        row["dataset_version"] = dataset_version
        row["legacy_labels_v1"] = {field: str(row[field]) for field in LABEL_FIELDS}
        row["label_status"] = "pending_instance_relabel"
        row["training_eligible"] = False
        provenance = dict(row.get("provenance") or {})
        provenance["carried_forward_from_dataset_version"] = source.get("dataset_version")
        replacement = replacements.get(str(row["sample_id"]))
        if replacement is not None:
            row["after_text"] = str(replacement["text"])
            provenance.update(
                candidate_method="llm_selective_regeneration_v2",
                candidate_model=model,
                generation_prompt_version=replacement.get(
                    "prompt_version", REGENERATION_PROMPT_VERSION
                ),
                generation_validation=dict(replacement.get("metrics") or {}),
            )
            if row["candidate_type"] == "wrong_target":
                provenance.update(wrong_target_spec(str(row["target_rubric"])))
        row["provenance"] = provenance
        output.append(row)
    return output


def regeneration_input_digest(
    row: Mapping[str, Any],
    state_group: Mapping[str, Mapping[str, Any]],
    *,
    generation_variant: str = "base",
) -> str:
    payload = {
        "prompt_version": (
            STRONG_RETRY_PROMPT_VERSION
            if generation_variant == "strong"
            else REGENERATION_PROMPT_VERSION
        ),
        "sample_id": row["sample_id"],
        "candidate_type": row["candidate_type"],
        "question": row.get("question", ""),
        "before_text": row["before_text"],
        "reference": state_group["correct_repair"]["after_text"],
        "target_rubric": row["target_rubric"],
        "intended_action": row["intended_action"],
        "changed_spans": row.get("changed_spans", []),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def wrong_target_spec(target_rubric: str) -> dict[str, str]:
    if target_rubric.startswith("expression_"):
        return {
            "requested_wrong_target_rubric": "content_2",
            "requested_wrong_target_action": "ADD_DETAIL",
            "requested_wrong_target_intent": "ADD_SUPPORTING_EXPLANATION",
            "requested_wrong_target_definition": (
                "Add one concrete supporting explanation to a non-target claim while leaving "
                "the expression defect untouched."
            ),
        }
    return {
        "requested_wrong_target_rubric": "expression_2",
        "requested_wrong_target_action": "STYLE_REFINE",
        "requested_wrong_target_intent": "IMPROVE_SENTENCE_CLARITY",
        "requested_wrong_target_definition": (
            "Substantively improve the clarity and wording of at least one complete non-target "
            "sentence while leaving the target content/organization defect untouched."
        ),
    }


def _validate_target_defect_preserved(
    row: Mapping[str, Any], current: str, candidate: str
) -> None:
    spans = row.get("changed_spans", [])
    operation = str(row.get("corruption_type") or "")
    if operation == "DELETE_SPECIFICS":
        restored = [
            str(span.get("target_span") or "")
            for span in spans
            if span.get("target_span") and span.get("target_span") in candidate
        ]
        if restored:
            raise ValueError("wrong_target restored a deleted target sentence")
        return
    required_fragments = [
        str(span.get("text") or "")
        for span in spans
        if span.get("text") and str(span.get("text")) in current
    ]
    missing = [fragment for fragment in required_fragments if fragment not in candidate]
    if missing:
        raise ValueError(
            f"wrong_target changed or removed {len(missing)} known corruption fragments"
        )
    current_order = [current.find(fragment) for fragment in required_fragments]
    candidate_order = [candidate.find(fragment) for fragment in required_fragments]
    if sorted(range(len(current_order)), key=current_order.__getitem__) != sorted(
        range(len(candidate_order)), key=candidate_order.__getitem__
    ):
        raise ValueError("wrong_target changed the order of known corruption fragments")


def _edit_ratio(left: str, right: str) -> float:
    return 1.0 - SequenceMatcher(None, left, right).ratio()


def _safe_non_target_sentences(row: Mapping[str, Any], current: str) -> list[str]:
    protected = {
        str(span.get(field) or "")
        for span in row.get("changed_spans", [])
        for field in ("target_span", "text")
        if span.get(field)
    }
    sentences = re.split(r"(?<=[.!?])\s+", current.strip())
    candidates = [
        sentence.strip()
        for sentence in sentences
        if 8 <= len(sentence.strip()) <= 180
        and not any(fragment in sentence for fragment in protected)
    ]
    return candidates


def _optional_string(value: Any) -> str | None:
    return str(value) if value is not None else None
