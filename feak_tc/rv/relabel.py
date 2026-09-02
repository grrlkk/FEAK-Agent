"""Instance-level multi-model relabeling for the Revision Verifier pilot."""

from __future__ import annotations

import hashlib
import json
import time
from collections import Counter, defaultdict
from typing import Any, Callable, Mapping, Sequence

from feak_tc.mvp.llm import LLMResponseError, LLMUnavailable, request_json

from .judge import fleiss_kappa
from .schema import CANDIDATE_TYPES, LABEL_FIELDS, LABEL_VALUES


RELABEL_PROTOCOL_VERSION = "rv-instance-relabel-v2"
OBSERVED_CANDIDATE_TYPES = (*CANDIDATE_TYPES, "other")
CANDIDATE_CODES = tuple(f"C{index}" for index in range(1, 7))

_JUDGMENT = {
    "type": "object",
    "properties": {
        "candidate_code": {"type": "string", "enum": list(CANDIDATE_CODES)},
        "observed_candidate_type": {
            "type": "string",
            "enum": list(OBSERVED_CANDIDATE_TYPES),
        },
        **{
            field: {"type": "string", "enum": list(LABEL_VALUES)}
            for field in LABEL_FIELDS
        },
        "confidence": {"type": "integer", "minimum": 0, "maximum": 100},
        "notes": {"type": "string", "minLength": 1, "maxLength": 500},
    },
    "required": [
        "candidate_code",
        "observed_candidate_type",
        *LABEL_FIELDS,
        "confidence",
        "notes",
    ],
    "additionalProperties": False,
}

RELABEL_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "judgments": {
            "type": "array",
            "minItems": 6,
            "maxItems": 6,
            "items": _JUDGMENT,
        }
    },
    "required": ["judgments"],
    "additionalProperties": False,
}

SYSTEM_PROMPT = """당신은 한국어 글쓰기 Revision Verifier 데이터의 독립 검수자다.
같은 현재 글에 대한 여섯 수정 후보를 각각 독립적으로 판정한다. 후보 코드와 제시 순서는
무작위이며 생성 방식이나 숨은 정답을 추정해서 점수를 배정하지 않는다.

네 평가축의 운영 정의:
1. target_fulfillment
- pass: 지정된 target 결함을 실질적으로 충분히 해결했다.
- partial: 실제 개선은 있으나 중요한 결함이 남았다.
- fail: target이 개선되지 않았거나 더 나빠졌다.

2. preservation
- pass: target 밖의 기존 주장, 근거, 의미를 불필요하게 바꾸거나 잃지 않았다.
- partial: 사소한 불필요 변경은 있지만 핵심 비대상 내용의 손상은 없다.
- fail: 비대상 주장/근거를 삭제, 왜곡하거나 근거 없는 내용을 실질적으로 추가했다.
현재 글에 이미 있던 결함 자체를 preservation 실패로 세지 않는다.

3. edit_appropriateness
- pass: intended action으로 target을 해결하는 데 필요하고 비례적인 수정이다.
- partial: intended action은 수행했지만 범위나 방법이 다소 불필요하거나 불완전하다.
- fail: intended action이 사실상 없고 엉뚱한 수정이 지배적이거나, 중대한 과수정/유해 변경이다.
단순히 '무슨 일이 일어났나'가 아니라 그 편집이 이 action에 적절했는지를 판정한다.

4. action_consistency
- pass: 관찰된 수정이 intended action 및 intent와 명확히 일치한다.
- partial: intended action이 일부 나타나지만 다른 수정과 섞였거나 불완전하다.
- fail: intended action이 없거나 다른 종류의 수정이 지배적이다.

관찰 후보 유형은 진단용이다:
- correct_repair: target을 충분히 복구하면서 나머지를 보존
- partial_repair: target을 실제로 일부만 복구
- wrong_target: target 대신 다른 부분을 주로 수정
- over_edit: target을 다루지만 불필요한 대규모/유해 변경 포함
- further_corruption: 현재 상태보다 추가로 악화
- no_edit: 실질적인 수정 없음
- other: 위 유형 중 하나로 명확히 분류되지 않음

reference repair는 synthetic 생성 과정에서 알려진 비교 기준일 뿐, 후보가 문자열 그대로 같다는
이유만으로 자동 판정하지 말고 실제 변화와 네 축을 확인한다. 여섯 후보를 빠짐없이 정확히 한 번씩
판정하고 notes에는 텍스트에서 관찰한 핵심 근거를 한국어 한두 문장으로 쓴다."""


def build_relabel_packets(
    rows: Sequence[Mapping[str, Any]], *, seed: int
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Create six-candidate blind packets and a separate candidate key."""

    grouped: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["state_id"])].append(row)
    public_rows: list[dict[str, Any]] = []
    key_rows: list[dict[str, Any]] = []
    for state_id in sorted(grouped, key=lambda value: _stable_int(f"{seed}:{value}")):
        state_rows = grouped[state_id]
        by_type = {str(row["candidate_type"]): row for row in state_rows}
        if set(by_type) != set(CANDIDATE_TYPES) or len(state_rows) != len(CANDIDATE_TYPES):
            raise ValueError(f"state {state_id} must have exactly six candidate types")
        ordered = sorted(
            state_rows,
            key=lambda row: _stable_int(f"{seed}:{state_id}:{row['sample_id']}"),
        )
        code_rows = list(zip(CANDIDATE_CODES, ordered))
        anchor = state_rows[0]
        review_id = f"rv-relabel:{state_id}"
        public_rows.append(
            {
                "review_id": review_id,
                "state_id": state_id,
                "essay_id": str(anchor["essay_id"]),
                "stage_k": int(anchor["stage_k"]),
                "question": str(anchor.get("question") or ""),
                "current_state": _common(state_rows, "before_text"),
                "reference_repair": str(by_type["correct_repair"]["after_text"]),
                "target_rubric": str(anchor["target_rubric"]),
                "intended_action": str(anchor["intended_action"]),
                "intent": str(anchor.get("intent") or ""),
                "known_corruption_type": str(anchor["corruption_type"]),
                "known_corruption_edits": list(anchor.get("changed_spans") or []),
                "candidates": [
                    {"candidate_code": code, "revised_text": str(row["after_text"])}
                    for code, row in code_rows
                ],
            }
        )
        key_rows.append(
            {
                "review_id": review_id,
                "state_id": state_id,
                "candidates": [
                    {
                        "candidate_code": code,
                        "sample_id": str(row["sample_id"]),
                        "candidate_type": str(row["candidate_type"]),
                    }
                    for code, row in code_rows
                ],
            }
        )
    return public_rows, key_rows


def request_relabel(
    row: Mapping[str, Any],
    *,
    model: str,
    max_attempts: int,
    timeout: float,
    reasoning_effort: str | None = None,
    verbosity: str | None = None,
    requester: Callable[..., dict[str, Any]] = request_json,
) -> dict[str, Any]:
    """Request one strict six-candidate judgment packet."""

    last_error: Exception | None = None
    for attempt in range(1, max_attempts + 1):
        try:
            raw = requester(
                system=SYSTEM_PROMPT,
                user=json.dumps(row, ensure_ascii=False, indent=2),
                model=model,
                temperature=None,
                reasoning_effort=reasoning_effort,
                verbosity=verbosity,
                json_schema=RELABEL_SCHEMA,
                response_format_name="rv_instance_relabel_v2",
                timeout=timeout,
            )
            return validate_relabel(raw)
        except (LLMUnavailable, LLMResponseError, ValueError) as exc:
            last_error = exc
            if attempt < max_attempts:
                time.sleep(min(2 ** (attempt - 1), 8))
    raise RuntimeError(
        f"relabel failed for {row.get('review_id')} with {model} after "
        f"{max_attempts} attempts: {last_error}"
    )


def validate_relabel(raw: Mapping[str, Any]) -> dict[str, Any]:
    judgments = raw.get("judgments")
    if not isinstance(judgments, list) or len(judgments) != 6:
        raise ValueError("relabel response must contain six judgments")
    validated = [_validate_judgment(item) for item in judgments]
    codes = [item["candidate_code"] for item in validated]
    if set(codes) != set(CANDIDATE_CODES) or len(codes) != len(set(codes)):
        raise ValueError("candidate codes must contain C1..C6 exactly once")
    return {"judgments": sorted(validated, key=lambda item: item["candidate_code"])}


def relabel_packet_digest(row: Mapping[str, Any]) -> str:
    payload = {
        "protocol_version": RELABEL_PROTOCOL_VERSION,
        "system_prompt": SYSTEM_PROMPT,
        "schema": RELABEL_SCHEMA,
        "public_packet": dict(row),
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def strict_candidate_quality(
    key_rows: Sequence[Mapping[str, Any]],
    judge_results: Mapping[str, Sequence[Mapping[str, Any]]],
) -> dict[str, dict[str, Any]]:
    """Require both non-generator type judges to identify generated candidates."""

    if len(judge_results) < 2:
        raise ValueError("candidate quality gate requires at least two judges")
    indexed = {
        name: {str(row["review_id"]): row for row in rows}
        for name, rows in judge_results.items()
    }
    result: dict[str, dict[str, Any]] = {}
    for key in key_rows:
        review_id = str(key["review_id"])
        for side in ("candidate_a", "candidate_b"):
            candidate = key[side]
            sample_id = str(candidate["sample_id"])
            expected = str(candidate["candidate_type"])
            votes = {
                name: str(rows[review_id][side]["inferred_candidate_type"])
                for name, rows in indexed.items()
            }
            passed = all(vote == expected for vote in votes.values())
            result[sample_id] = {
                "passed": passed,
                "reason": (
                    "two_non_generator_type_judges_agree"
                    if passed
                    else "non_generator_type_judges_do_not_both_agree"
                ),
                "expected_type": expected,
                "votes": votes,
            }
    return result


def aggregate_relabels(
    rows: Sequence[Mapping[str, Any]],
    key_rows: Sequence[Mapping[str, Any]],
    results_by_model: Mapping[str, Sequence[Mapping[str, Any]]],
    generated_quality: Mapping[str, Mapping[str, Any]],
    *,
    dataset_version: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    """Attach per-instance votes, majority labels, and conservative trainability."""

    if len(results_by_model) != 3:
        raise ValueError("instance relabeling requires exactly three models")
    models = list(results_by_model)
    key_by_review = {str(item["review_id"]): item for item in key_rows}
    keyed: dict[str, dict[str, Any]] = {}
    for review_id, key in key_by_review.items():
        for candidate in key["candidates"]:
            keyed[str(candidate["sample_id"])] = {
                "review_id": review_id,
                "candidate_code": str(candidate["candidate_code"]),
            }
    indexed = {
        model: {str(item["review_id"]): item for item in values}
        for model, values in results_by_model.items()
    }
    all_rows: list[dict[str, Any]] = []
    rating_rows: dict[str, list[list[str]]] = defaultdict(list)
    for source in rows:
        row = dict(source)
        sample_id = str(row["sample_id"])
        locator = keyed[sample_id]
        review_id = locator["review_id"]
        code = locator["candidate_code"]
        votes: dict[str, dict[str, Any]] = {}
        for model in models:
            judgments = indexed[model][review_id]["judgments"]
            votes[model] = next(item for item in judgments if item["candidate_code"] == code)
        label_votes = {
            field: {model: str(votes[model][field]) for model in models}
            for field in LABEL_FIELDS
        }
        consensuses = {
            field: _majority(list(label_votes[field].values()))
            for field in LABEL_FIELDS
        }
        observed_votes = {
            model: str(votes[model]["observed_candidate_type"]) for model in models
        }
        observed_consensus = _majority(list(observed_votes.values()))
        for field in LABEL_FIELDS:
            rating_rows[field].append(list(label_votes[field].values()))
        rating_rows["observed_candidate_type"].append(list(observed_votes.values()))

        candidate_type = str(row["candidate_type"])
        if candidate_type in {"wrong_target", "over_edit"}:
            quality = dict(generated_quality[sample_id])
        else:
            quality = {
                "passed": True,
                "reason": "trajectory_or_deterministic_candidate",
            }
        labels_resolved = all(value is not None for value in consensuses.values())
        eligible = bool(quality["passed"] and labels_resolved)
        row.update(consensuses)
        row.update(
            {
                "dataset_version": dataset_version,
                "weak_supervision": True,
                "label_source": "three_non_generator_llm_majority_v2",
                "label_models": models,
                "label_votes": label_votes,
                "observed_candidate_type_votes": observed_votes,
                "observed_candidate_type_consensus": observed_consensus,
                "review_confidence": {
                    model: int(votes[model]["confidence"]) for model in models
                },
                "review_notes": {model: str(votes[model]["notes"]) for model in models},
                "candidate_quality_gate": quality,
                "label_status": (
                    "consensus_all_axes" if labels_resolved else "unresolved_axis"
                ),
                "training_eligible": eligible,
            }
        )
        all_rows.append(row)

    train_rows = [row for row in all_rows if row["training_eligible"]]
    report = {
        "dataset_version": dataset_version,
        "models": models,
        "states": len(key_rows),
        "all_candidates": len(all_rows),
        "training_candidates": len(train_rows),
        "training_coverage": round(len(train_rows) / len(all_rows), 6),
        "quality_gate_passed": sum(
            bool(row["candidate_quality_gate"]["passed"]) for row in all_rows
        ),
        "quality_gate_passed_by_candidate_type": _boolean_counts_by_type(
            all_rows, "candidate_quality_gate", "passed"
        ),
        "unresolved_by_axis": {
            field: sum(row[field] is None for row in all_rows) for field in LABEL_FIELDS
        },
        "unresolved_rows_by_candidate_type": _counts(
            [row for row in all_rows if row["label_status"] == "unresolved_axis"],
            "candidate_type",
        ),
        "training_by_candidate_type": _counts(train_rows, "candidate_type"),
        "all_by_candidate_type": _counts(all_rows, "candidate_type"),
        "label_distribution_training": {
            field: _counts(train_rows, field) for field in LABEL_FIELDS
        },
        "label_distribution_training_by_candidate_type": {
            candidate_type: {
                field: _counts(
                    [row for row in train_rows if row["candidate_type"] == candidate_type],
                    field,
                )
                for field in LABEL_FIELDS
            }
            for candidate_type in CANDIDATE_TYPES
        },
        "observed_candidate_type_consensus": {
            "unresolved": sum(
                row["observed_candidate_type_consensus"] is None for row in all_rows
            ),
            "matches_provenance": sum(
                row["observed_candidate_type_consensus"] == row["candidate_type"]
                for row in all_rows
            ),
            "matches_provenance_by_candidate_type": {
                candidate_type: {
                    "matches": sum(
                        row["observed_candidate_type_consensus"] == candidate_type
                        for row in all_rows
                        if row["candidate_type"] == candidate_type
                    ),
                    "candidates": sum(
                        row["candidate_type"] == candidate_type for row in all_rows
                    ),
                }
                for candidate_type in CANDIDATE_TYPES
            },
            "distribution_by_provenance_type": {
                candidate_type: _counts(
                    [row for row in all_rows if row["candidate_type"] == candidate_type],
                    "observed_candidate_type_consensus",
                )
                for candidate_type in CANDIDATE_TYPES
            },
        },
        "inter_rater_fleiss_kappa": {
            field: fleiss_kappa(values) for field, values in rating_rows.items()
        },
        "unanimous_rate": {
            field: round(
                sum(len(set(values)) == 1 for values in rating_rows[field])
                / len(rating_rows[field]),
                6,
            )
            for field in (*LABEL_FIELDS, "observed_candidate_type")
        },
    }
    return all_rows, train_rows, report


def _validate_judgment(raw: Any) -> dict[str, Any]:
    if not isinstance(raw, Mapping):
        raise ValueError("candidate judgment must be an object")
    code = str(raw.get("candidate_code") or "")
    observed = str(raw.get("observed_candidate_type") or "")
    if code not in CANDIDATE_CODES:
        raise ValueError(f"invalid candidate code: {code}")
    if observed not in OBSERVED_CANDIDATE_TYPES:
        raise ValueError(f"invalid observed candidate type: {observed}")
    result = {"candidate_code": code, "observed_candidate_type": observed}
    for field in LABEL_FIELDS:
        value = raw.get(field)
        if value not in LABEL_VALUES:
            raise ValueError(f"invalid {field}: {value}")
        result[field] = value
    confidence = raw.get("confidence")
    if not isinstance(confidence, int) or isinstance(confidence, bool) or not 0 <= confidence <= 100:
        raise ValueError("confidence must be an integer from 0 to 100")
    notes = raw.get("notes")
    if not isinstance(notes, str) or not notes.strip() or len(notes) > 500:
        raise ValueError("notes must contain 1..500 characters")
    result.update(confidence=confidence, notes=notes.strip())
    return result


def _majority(values: Sequence[str]) -> str | None:
    counts = Counter(values)
    value, count = counts.most_common(1)[0]
    return value if count >= 2 else None


def _common(rows: Sequence[Mapping[str, Any]], field: str) -> str:
    values = {str(row[field]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"state rows disagree on {field}")
    return values.pop()


def _counts(rows: Sequence[Mapping[str, Any]], field: str) -> dict[str, int]:
    return dict(sorted(Counter(str(row[field]) for row in rows).items()))


def _boolean_counts_by_type(
    rows: Sequence[Mapping[str, Any]], outer: str, inner: str
) -> dict[str, dict[str, int]]:
    result = {}
    for candidate_type in CANDIDATE_TYPES:
        selected = [row for row in rows if row["candidate_type"] == candidate_type]
        result[candidate_type] = {
            "passed": sum(bool(row[outer][inner]) for row in selected),
            "candidates": len(selected),
        }
    return result


def _stable_int(value: str) -> int:
    return int(hashlib.sha256(value.encode("utf-8")).hexdigest(), 16)
