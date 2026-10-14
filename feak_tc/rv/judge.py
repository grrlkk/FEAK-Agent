"""Blind multi-model review utilities for the Revision Verifier pilot."""

from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json

from .schema import LABEL_FIELDS, LABEL_VALUES


REVIEWED_CANDIDATE_TYPES = ("wrong_target", "over_edit")
INFERRED_CANDIDATE_TYPES = (*REVIEWED_CANDIDATE_TYPES, "other")
JUDGE_PROTOCOL_VERSION = "rv-blind-candidate-judge-v1"

JUDGMENT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "candidate_a": {"$ref": "#/$defs/candidate_judgment"},
        "candidate_b": {"$ref": "#/$defs/candidate_judgment"},
    },
    "required": ["candidate_a", "candidate_b"],
    "additionalProperties": False,
    "$defs": {
        "candidate_judgment": {
            "type": "object",
            "properties": {
                "inferred_candidate_type": {
                    "type": "string",
                    "enum": list(INFERRED_CANDIDATE_TYPES),
                },
                **{
                    field: {"type": "string", "enum": list(LABEL_VALUES)}
                    for field in LABEL_FIELDS
                },
                "usable_for_weak_supervision": {"type": "boolean"},
                "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
                "notes": {"type": "string", "minLength": 1, "maxLength": 500},
            },
            "required": [
                "inferred_candidate_type",
                *LABEL_FIELDS,
                "usable_for_weak_supervision",
                "confidence",
                "notes",
            ],
            "additionalProperties": False,
        }
    },
}

SYSTEM_PROMPT = """당신은 한국어 글쓰기 수정 transition을 평가하는 독립 블라인드 검수자다.
현재 글에 같은 수정 action을 적용한 후보 A와 B를 각각 독립적으로 평가한다.

후보 유형:
- wrong_target: 글은 바뀌었지만 지정된 target rubric/action이 아닌 다른 문제를 주로 수정함
- over_edit: target 문제는 다루지만 필요 이상의 대규모 재작성, 내용 손실/추가, 문체 변형이 있음
- other: 위 두 유형 중 어느 것도 명확히 해당하지 않음

평가 축:
- target_fulfillment: 지정된 target 결함을 실제로 해결한 정도
- preservation: target 밖의 원래 의미, 근거, 문체와 구조를 보존한 정도
- edit_appropriateness: 수정 범위와 강도가 문제 해결에 필요한 수준인지
- action_consistency: 수정 결과가 intended action 및 intent와 일치하는지

각 축은 pass/partial/fail로 판정한다. usable_for_weak_supervision은 후보가 해당 판정을
학습시키기에 자연스럽고 구분이 명확할 때만 true다. 두 후보가 어떤 방식으로 생성됐는지,
숨은 정답이나 다른 평가자의 결과가 있다고 가정하지 말고 보이는 텍스트만 판정한다.
confidence는 반드시 0~100 척도(예: 높은 확신은 80 이상)로 쓰고, notes에는 핵심 근거를
한국어 한두 문장으로 적는다."""


def build_blind_packets(
    rows: Sequence[Mapping[str, Any]],
    *,
    sample_states: int,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Select a stratified state sample and separate public data from its key."""

    if sample_states < 1:
        raise ValueError("sample_states must be at least 1")
    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("candidate_type")) in REVIEWED_CANDIDATE_TYPES:
            grouped[str(row["state_id"])].append(row)

    states: dict[str, dict[str, Mapping[str, Any]]] = {}
    strata: dict[tuple[str, int], list[str]] = defaultdict(list)
    for state_id, state_rows in grouped.items():
        by_type = {str(row["candidate_type"]): row for row in state_rows}
        if set(by_type) != set(REVIEWED_CANDIDATE_TYPES):
            raise ValueError(f"state {state_id} does not have both reviewed candidate types")
        states[state_id] = by_type
        anchor = state_rows[0]
        strata[(str(anchor["corruption_type"]), int(anchor["stage_k"]))].append(state_id)

    if sample_states > len(states):
        raise ValueError(
            f"requested {sample_states} states but only {len(states)} are available"
        )
    selected_ids = _balanced_state_ids(strata, sample_states=sample_states, seed=seed)

    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for state_id in selected_ids:
        by_type = states[state_id]
        anchor = by_type[REVIEWED_CANDIDATE_TYPES[0]]
        side_types = list(REVIEWED_CANDIDATE_TYPES)
        if _stable_int(f"{seed}:{state_id}:side") % 2:
            side_types.reverse()
        side_rows = {side: by_type[kind] for side, kind in zip(("a", "b"), side_types)}
        review_id = f"rv-review:{state_id}"
        public_rows.append(
            {
                "review_id": review_id,
                "state_id": state_id,
                "essay_id": str(anchor["essay_id"]),
                "stage_k": int(anchor["stage_k"]),
                "corruption_type": str(anchor["corruption_type"]),
                "question": str(anchor.get("question") or ""),
                "current_state": _common_value(by_type, "before_text"),
                "reference_repair": _reference_repair(rows, state_id),
                "target_rubric": str(anchor["target_rubric"]),
                "intended_action": str(anchor["intended_action"]),
                "intent": str(anchor.get("intent") or ""),
                "known_corruption_edits": list(anchor.get("changed_spans") or []),
                "candidate_a_text": str(side_rows["a"]["after_text"]),
                "candidate_b_text": str(side_rows["b"]["after_text"]),
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "state_id": state_id,
                "stratum": {
                    "corruption_type": str(anchor["corruption_type"]),
                    "stage_k": int(anchor["stage_k"]),
                },
                "candidate_a": _candidate_key(side_rows["a"]),
                "candidate_b": _candidate_key(side_rows["b"]),
            }
        )
    return public_rows, key_rows


def request_judgment(
    row: Mapping[str, Any],
    *,
    model: str,
    max_attempts: int,
    timeout: float,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    requester: Callable[..., dict[str, Any]] = request_json,
) -> dict[str, Any]:
    """Request and validate one model's blind judgment of a public packet."""

    user_prompt = _user_prompt(row)
    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = requester(
                system=SYSTEM_PROMPT,
                user=user_prompt,
                model=model,
                temperature=None,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                json_schema=JUDGMENT_SCHEMA,
                response_format_name="rv_blind_candidate_judgment",
                timeout=timeout,
            )
            return validate_judgment(raw)
        except (LLMUnavailable, LLMResponseError, ValueError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"review failed for {row.get('review_id')} with {model} "
        f"after {max_attempts} attempts: {last_error}"
    )


def review_packet_digest(row: Mapping[str, Any]) -> str:
    """Bind cached model output to the exact public packet and judge protocol."""

    payload = {
        "protocol_version": JUDGE_PROTOCOL_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "schema": JUDGMENT_SCHEMA,
        "public_packet": dict(row),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def validate_judgment(raw: Mapping[str, Any]) -> dict[str, Any]:
    """Validate a strict judgment payload independently of the API schema."""

    return {
        side: _validate_candidate_judgment(raw.get(side), side)
        for side in ("candidate_a", "candidate_b")
    }


def analyze_judgments(
    results_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    key_rows: Sequence[Mapping[str, Any]],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Compare independent judgments with hidden types and pilot weak labels."""

    if len(results_by_model) < 2:
        raise ValueError("at least two model result sets are required")
    model_names = list(results_by_model)
    key_by_id = {str(row["review_id"]): row for row in key_rows}
    indexed: dict[str, dict[str, Mapping[str, Any]]] = {}
    for model, rows in results_by_model.items():
        model_rows = {str(row["review_id"]): row for row in rows}
        if set(model_rows) != set(key_by_id):
            raise ValueError(f"model {model} result IDs do not match the hidden key")
        indexed[model] = model_rows

    items = _flatten_items(indexed, key_by_id, model_names)
    per_model = {
        model: _score_model(items, model)
        for model in model_names
    }
    majority = _majority_report(items, model_names)
    agreement = {
        "models": model_names,
        "three_way_unanimity": {
            field: _unanimity(items, model_names, field)
            for field in ("inferred_candidate_type", *LABEL_FIELDS, "usable_for_weak_supervision")
        },
        "pairwise": {
            f"{left}__{right}": {
                field: _pair_agreement(items, left, right, field)
                for field in ("inferred_candidate_type", *LABEL_FIELDS, "usable_for_weak_supervision")
            }
            for left, right in combinations(model_names, 2)
        },
        "pairwise_chance_diagnostics": {
            f"{left}__{right}": {
                field: _pair_diagnostics(items, left, right, field)
                for field in ("inferred_candidate_type", *LABEL_FIELDS, "usable_for_weak_supervision")
            }
            for left, right in combinations(model_names, 2)
        },
        "fleiss_kappa": {
            field: fleiss_kappa(
                [[item["judgments"][model][field] for model in model_names] for item in items]
            )
            for field in ("inferred_candidate_type", *LABEL_FIELDS, "usable_for_weak_supervision")
        },
    }
    disagreements = _disagreement_rows(items, model_names)
    report = {
        "evaluation": "RV_PILOT_INDEPENDENT_LLM_BLIND_REVIEW",
        "reviewed_states": len(key_rows),
        "reviewed_candidates": len(items),
        "candidate_types": dict(sorted(Counter(item["expected_type"] for item in items).items())),
        "models": model_names,
        "key_excluded_from_model_inputs": True,
        "per_model": per_model,
        "majority_vote": majority,
        "inter_rater_agreement": agreement,
        "disagreement_candidates": len(disagreements),
    }
    return report, disagreements


def fleiss_kappa(ratings: Sequence[Sequence[Any]]) -> float | None:
    """Return Fleiss' kappa for equally sized categorical rating rows."""

    if not ratings:
        return None
    raters = len(ratings[0])
    if raters < 2 or any(len(row) != raters for row in ratings):
        raise ValueError("Fleiss kappa requires equally sized rows with at least two raters")
    categories = sorted({value for row in ratings for value in row}, key=str)
    counts = [Counter(row) for row in ratings]
    observed = sum(
        (sum(count[category] ** 2 for category in categories) - raters)
        / (raters * (raters - 1))
        for count in counts
    ) / len(counts)
    total = len(counts) * raters
    proportions = {
        category: sum(count[category] for count in counts) / total
        for category in categories
    }
    expected = sum(value ** 2 for value in proportions.values())
    if expected == 1.0:
        return 1.0 if observed == 1.0 else None
    return round((observed - expected) / (1.0 - expected), 6)


def _balanced_state_ids(
    strata: Mapping[tuple[str, int], Sequence[str]],
    *,
    sample_states: int,
    seed: int,
) -> list[str]:
    if not strata:
        raise ValueError("no reviewable states found")
    ordered_strata = sorted(strata)
    queues = {
        stratum: sorted(
            ids,
            key=lambda state_id: _stable_int(f"{seed}:{stratum}:{state_id}"),
        )
        for stratum, ids in strata.items()
    }
    selected: list[str] = []
    cursor = 0
    while len(selected) < sample_states:
        progressed = False
        for offset in range(len(ordered_strata)):
            stratum = ordered_strata[(cursor + offset) % len(ordered_strata)]
            if queues[stratum]:
                selected.append(queues[stratum].pop(0))
                cursor = (cursor + offset + 1) % len(ordered_strata)
                progressed = True
                break
        if not progressed:
            raise ValueError("not enough states to satisfy balanced sampling")
    return selected


def _reference_repair(rows: Sequence[Mapping[str, Any]], state_id: str) -> str:
    matches = [
        row for row in rows
        if str(row.get("state_id")) == state_id
        and str(row.get("candidate_type")) == "correct_repair"
    ]
    if len(matches) != 1:
        raise ValueError(f"state {state_id} must have exactly one correct_repair")
    return str(matches[0]["after_text"])


def _common_value(by_type: Mapping[str, Mapping[str, Any]], field: str) -> str:
    values = {str(row[field]) for row in by_type.values()}
    if len(values) != 1:
        raise ValueError(f"candidate rows disagree on {field}")
    return values.pop()


def _candidate_key(row: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "sample_id": str(row["sample_id"]),
        "candidate_type": str(row["candidate_type"]),
        "expected_labels": {field: str(row[field]) for field in LABEL_FIELDS},
    }


def _user_prompt(row: Mapping[str, Any]) -> str:
    edits = json.dumps(row.get("known_corruption_edits") or [], ensure_ascii=False)
    return (
        f"review_id: {row['review_id']}\n"
        f"질문: {row.get('question', '')}\n"
        f"target_rubric: {row['target_rubric']}\n"
        f"intended_action: {row['intended_action']}\n"
        f"intent: {row.get('intent', '')}\n"
        f"원래 발생한 corruption edit: {edits}\n\n"
        f"[현재 글]\n{row['current_state']}\n\n"
        f"[정확한 복원 참고문]\n{row['reference_repair']}\n\n"
        f"[후보 A]\n{row['candidate_a_text']}\n\n"
        f"[후보 B]\n{row['candidate_b_text']}\n\n"
        "후보 A와 B를 각각 평가하여 지정된 JSON 형식으로 답하라."
    )


def _validate_candidate_judgment(raw: Any, side: str) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError(f"{side} must be an object")
    inferred = str(raw.get("inferred_candidate_type") or "")
    if inferred not in INFERRED_CANDIDATE_TYPES:
        raise ValueError(f"invalid {side}.inferred_candidate_type: {inferred!r}")
    result: dict[str, Any] = {"inferred_candidate_type": inferred}
    for field in LABEL_FIELDS:
        value = str(raw.get(field) or "")
        if value not in LABEL_VALUES:
            raise ValueError(f"invalid {side}.{field}: {value!r}")
        result[field] = value
    usable = raw.get("usable_for_weak_supervision")
    if not isinstance(usable, bool):
        raise ValueError(f"invalid {side}.usable_for_weak_supervision: {usable!r}")
    confidence = raw.get("confidence")
    if not isinstance(confidence, int) or isinstance(confidence, bool) or not 0 <= confidence <= 100:
        raise ValueError(f"invalid {side}.confidence: {confidence!r}")
    notes = str(raw.get("notes") or "").strip()
    if not notes:
        raise ValueError(f"{side}.notes must not be empty")
    result.update(
        usable_for_weak_supervision=usable,
        confidence=confidence,
        notes=notes[:500],
    )
    return result


def _flatten_items(
    indexed: Mapping[str, Mapping[str, Mapping[str, Any]]],
    key_by_id: Mapping[str, Mapping[str, Any]],
    models: Sequence[str],
) -> list[dict[str, Any]]:
    items = []
    for review_id, key in key_by_id.items():
        for side in ("candidate_a", "candidate_b"):
            expected = key[side]
            judgments = {
                model: indexed[model][review_id][side]
                for model in models
            }
            items.append(
                {
                    "review_id": review_id,
                    "state_id": key["state_id"],
                    "side": side,
                    "sample_id": expected["sample_id"],
                    "expected_type": expected["candidate_type"],
                    "expected_labels": expected["expected_labels"],
                    "stratum": key["stratum"],
                    "judgments": judgments,
                }
            )
    return items


def _score_model(items: Sequence[Mapping[str, Any]], model: str) -> dict[str, Any]:
    def score(subset: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        total = len(subset)
        type_correct = sum(
            item["judgments"][model]["inferred_candidate_type"] == item["expected_type"]
            for item in subset
        )
        label_correct = {
            field: sum(
                item["judgments"][model][field] == item["expected_labels"][field]
                for item in subset
            )
            for field in LABEL_FIELDS
        }
        usable = sum(
            bool(item["judgments"][model]["usable_for_weak_supervision"])
            for item in subset
        )
        confidence = sum(item["judgments"][model]["confidence"] for item in subset)
        return {
            "candidates": total,
            "intended_type_accuracy": _rate(type_correct, total),
            "label_exact_agreement": {
                field: _rate(label_correct[field], total) for field in LABEL_FIELDS
            },
            "all_four_labels_exact_rate": _rate(
                sum(
                    all(
                        item["judgments"][model][field] == item["expected_labels"][field]
                        for field in LABEL_FIELDS
                    )
                    for item in subset
                ),
                total,
            ),
            "usable_for_weak_supervision_rate": _rate(usable, total),
            "mean_confidence": round(confidence / total, 3) if total else None,
        }

    return {
        "overall": score(items),
        "by_candidate_type": {
            candidate_type: score(
                [item for item in items if item["expected_type"] == candidate_type]
            )
            for candidate_type in REVIEWED_CANDIDATE_TYPES
        },
        "inferred_type_confusion": {
            candidate_type: dict(
                sorted(
                    Counter(
                        item["judgments"][model]["inferred_candidate_type"]
                        for item in items
                        if item["expected_type"] == candidate_type
                    ).items()
                )
            )
            for candidate_type in REVIEWED_CANDIDATE_TYPES
        },
        "inferred_type_classification": {
            candidate_type: _classification_metrics(
                items,
                [
                    item["judgments"][model]["inferred_candidate_type"]
                    for item in items
                ],
                candidate_type,
            )
            for candidate_type in REVIEWED_CANDIDATE_TYPES
        },
    }


def _majority_report(
    items: Sequence[Mapping[str, Any]], models: Sequence[str]
) -> dict[str, Any]:
    fields = ("inferred_candidate_type", *LABEL_FIELDS, "usable_for_weak_supervision")
    decisions: dict[str, list[Any | None]] = {
        field: [
            _majority([item["judgments"][model][field] for model in models])
            for item in items
        ]
        for field in fields
    }
    type_values = decisions["inferred_candidate_type"]
    available_types = sum(value is not None for value in type_values)
    type_correct = sum(
        value == item["expected_type"]
        for value, item in zip(type_values, items)
        if value is not None
    )
    label_agreement = {}
    for field in LABEL_FIELDS:
        values = decisions[field]
        available = sum(value is not None for value in values)
        correct = sum(
            value == item["expected_labels"][field]
            for value, item in zip(values, items)
            if value is not None
        )
        label_agreement[field] = {
            "decided": available,
            "coverage": _rate(available, len(items)),
            "exact_agreement_on_decided": _rate(correct, available),
        }
    all_four_available = 0
    all_four_correct = 0
    for index, item in enumerate(items):
        values = [decisions[field][index] for field in LABEL_FIELDS]
        if all(value is not None for value in values):
            all_four_available += 1
            all_four_correct += int(
                all(
                    decisions[field][index] == item["expected_labels"][field]
                    for field in LABEL_FIELDS
                )
            )
    usable_values = decisions["usable_for_weak_supervision"]
    usable_decided = [value for value in usable_values if value is not None]
    return {
        "intended_type": {
            "decided": available_types,
            "coverage": _rate(available_types, len(items)),
            "accuracy_on_decided": _rate(type_correct, available_types),
            "exact": type_correct,
            "full_sample_accuracy": _rate(type_correct, len(items)),
            "classification": {
                candidate_type: _classification_metrics(
                    items,
                    type_values,
                    candidate_type,
                )
                for candidate_type in REVIEWED_CANDIDATE_TYPES
            },
        },
        "label_exact_agreement": label_agreement,
        "all_four_labels": {
            "decided": all_four_available,
            "coverage": _rate(all_four_available, len(items)),
            "all_exact_on_decided": _rate(all_four_correct, all_four_available),
        },
        "usable_for_weak_supervision": {
            "decided": len(usable_decided),
            "coverage": _rate(len(usable_decided), len(items)),
            "true_rate_on_decided": _rate(sum(bool(value) for value in usable_decided), len(usable_decided)),
        },
        "by_candidate_type": {
            candidate_type: _majority_type_summary(
                items,
                decisions,
                candidate_type=candidate_type,
            )
            for candidate_type in REVIEWED_CANDIDATE_TYPES
        },
        "operator_contrasts": _operator_contrasts(items, decisions),
    }


def _majority_type_summary(
    items: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Sequence[Any | None]],
    *,
    candidate_type: str,
) -> dict[str, Any]:
    indices = [
        index for index, item in enumerate(items)
        if item["expected_type"] == candidate_type
    ]
    total = len(indices)
    inferred = [decisions["inferred_candidate_type"][index] for index in indices]
    usable = [decisions["usable_for_weak_supervision"][index] for index in indices]
    exact = sum(value == candidate_type for value in inferred)
    return {
        "candidates": total,
        "intended_type_exact": exact,
        "intended_type_exact_rate": _rate(exact, total),
        "intended_type_exact_wilson_95_ci": wilson_interval(exact, total),
        "inferred_type_distribution": dict(
            sorted(Counter("undecided" if value is None else value for value in inferred).items())
        ),
        "usable_true_rate": _rate(sum(value is True for value in usable), total),
        "label_exact_rate": {
            field: _rate(
                sum(
                    decisions[field][index] == items[index]["expected_labels"][field]
                    for index in indices
                ),
                total,
            )
            for field in LABEL_FIELDS
        },
        "label_decision_distribution": {
            field: dict(
                sorted(
                    Counter(
                        "undecided" if decisions[field][index] is None
                        else decisions[field][index]
                        for index in indices
                    ).items()
                )
            )
            for field in LABEL_FIELDS
        },
        "intended_type_by_stratum": {
            stratum: {
                "candidates": len(stratum_indices),
                "exact": sum(
                    decisions["inferred_candidate_type"][index] == candidate_type
                    for index in stratum_indices
                ),
                "exact_rate": _rate(
                    sum(
                        decisions["inferred_candidate_type"][index] == candidate_type
                        for index in stratum_indices
                    ),
                    len(stratum_indices),
                ),
            }
            for stratum, stratum_indices in _stratum_indices(items, indices).items()
        },
    }


def _stratum_indices(
    items: Sequence[Mapping[str, Any]], indices: Sequence[int]
) -> dict[str, list[int]]:
    grouped: dict[str, list[int]] = defaultdict(list)
    for index in indices:
        stratum = items[index]["stratum"]
        name = f"{stratum['corruption_type']}:stage{stratum['stage_k']}"
        grouped[name].append(index)
    return dict(sorted(grouped.items()))


def _unanimity(
    items: Sequence[Mapping[str, Any]], models: Sequence[str], field: str
) -> float | None:
    return _rate(
        sum(
            len({item["judgments"][model][field] for model in models}) == 1
            for item in items
        ),
        len(items),
    )


def _pair_agreement(
    items: Sequence[Mapping[str, Any]], left: str, right: str, field: str
) -> float | None:
    return _rate(
        sum(
            item["judgments"][left][field] == item["judgments"][right][field]
            for item in items
        ),
        len(items),
    )


def _pair_diagnostics(
    items: Sequence[Mapping[str, Any]], left: str, right: str, field: str
) -> dict[str, Any]:
    left_values = [item["judgments"][left][field] for item in items]
    right_values = [item["judgments"][right][field] for item in items]
    total = len(items)
    left_counts = Counter(left_values)
    right_counts = Counter(right_values)
    categories = set(left_counts) | set(right_counts)
    observed = sum(a == b for a, b in zip(left_values, right_values)) / total
    expected = sum(
        (left_counts[category] / total) * (right_counts[category] / total)
        for category in categories
    )
    kappa = None if expected == 1.0 else (observed - expected) / (1.0 - expected)
    return {
        "observed_agreement": round(observed, 6),
        "expected_agreement_from_marginals": round(expected, 6),
        "observed_minus_expected": round(observed - expected, 6),
        "cohen_kappa": round(kappa, 6) if kappa is not None else None,
        "left_distribution": dict(sorted(left_counts.items(), key=lambda item: str(item[0]))),
        "right_distribution": dict(sorted(right_counts.items(), key=lambda item: str(item[0]))),
    }


def _classification_metrics(
    items: Sequence[Mapping[str, Any]],
    predictions: Sequence[Any | None],
    positive_type: str,
) -> dict[str, Any]:
    true_positive = sum(
        prediction == positive_type and item["expected_type"] == positive_type
        for item, prediction in zip(items, predictions)
    )
    false_positive = sum(
        prediction == positive_type and item["expected_type"] != positive_type
        for item, prediction in zip(items, predictions)
    )
    false_negative = sum(
        prediction != positive_type and item["expected_type"] == positive_type
        for item, prediction in zip(items, predictions)
    )
    predicted_positive = true_positive + false_positive
    actual_positive = true_positive + false_negative
    return {
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": _rate(true_positive, predicted_positive),
        "recall": _rate(true_positive, actual_positive),
        "recall_wilson_95_ci": wilson_interval(true_positive, actual_positive),
    }


def _operator_contrasts(
    items: Sequence[Mapping[str, Any]],
    decisions: Mapping[str, Sequence[Any | None]],
) -> dict[str, Any]:
    selected = [
        index for index, item in enumerate(items)
        if item["expected_type"] == "over_edit"
    ]
    delete = [
        index for index in selected
        if items[index]["stratum"]["corruption_type"] == "DELETE_SPECIFICS"
    ]
    delete_set = set(delete)
    other = [index for index in selected if index not in delete_set]
    delete_exact = sum(
        decisions["inferred_candidate_type"][index] == "over_edit"
        for index in delete
    )
    other_exact = sum(
        decisions["inferred_candidate_type"][index] == "over_edit"
        for index in other
    )
    return {
        "over_edit_delete_specifics_vs_other_operators": {
            "delete_specifics": {
                "exact": delete_exact,
                "candidates": len(delete),
                "rate": _rate(delete_exact, len(delete)),
                "wilson_95_ci": wilson_interval(delete_exact, len(delete)),
            },
            "other_operators": {
                "exact": other_exact,
                "candidates": len(other),
                "rate": _rate(other_exact, len(other)),
                "wilson_95_ci": wilson_interval(other_exact, len(other)),
            },
            "fisher_exact_two_sided_p": fisher_exact_two_sided(
                delete_exact,
                len(delete) - delete_exact,
                other_exact,
                len(other) - other_exact,
            ),
        }
    }


def _disagreement_rows(
    items: Sequence[Mapping[str, Any]], models: Sequence[str]
) -> list[dict[str, Any]]:
    fields = ("inferred_candidate_type", *LABEL_FIELDS, "usable_for_weak_supervision")
    result = []
    for item in items:
        disagreements = {
            field: {
                model: item["judgments"][model][field]
                for model in models
            }
            for field in fields
            if len({item["judgments"][model][field] for model in models}) > 1
        }
        if disagreements:
            result.append(
                {
                    "review_id": item["review_id"],
                    "sample_id": item["sample_id"],
                    "side": item["side"],
                    "expected_type": item["expected_type"],
                    "expected_labels": item["expected_labels"],
                    "disagreements": disagreements,
                    "notes": {
                        model: item["judgments"][model]["notes"]
                        for model in models
                    },
                }
            )
    return result


def _majority(values: Sequence[Any]) -> Any | None:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value if count > len(values) / 2 else None


def _rate(numerator: int, denominator: int) -> float | None:
    return round(numerator / denominator, 6) if denominator else None


def wilson_interval(successes: int, total: int) -> dict[str, float] | None:
    """Return a two-sided Wilson 95% interval for a binomial proportion."""

    if total < 0 or successes < 0 or successes > total:
        raise ValueError("invalid binomial counts")
    if total == 0:
        return None
    z = 1.959963984540054
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
    return {"low": round(center - margin, 6), "high": round(center + margin, 6)}


def fisher_exact_two_sided(a: int, b: int, c: int, d: int) -> float:
    """Return the two-sided Fisher exact p-value for a 2x2 count table."""

    if any(not isinstance(value, int) or value < 0 for value in (a, b, c, d)):
        raise ValueError("Fisher exact counts must be non-negative integers")
    row_one = a + b
    row_two = c + d
    column_one = a + c
    total = row_one + row_two
    if total == 0:
        raise ValueError("Fisher exact table must not be empty")

    def probability(cell_a: int) -> float:
        return (
            math.comb(row_one, cell_a)
            * math.comb(row_two, column_one - cell_a)
            / math.comb(total, column_one)
        )

    lower = max(0, column_one - row_two)
    upper = min(row_one, column_one)
    observed = probability(a)
    p_value = sum(
        probability(cell_a)
        for cell_a in range(lower, upper + 1)
        if probability(cell_a) <= observed + 1e-15
    )
    return round(min(p_value, 1.0), 12)


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def write_jsonl_atomic(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(target)


def write_json_atomic(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(target)
