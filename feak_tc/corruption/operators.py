"""FEAK-TC v2 G1 corruption operators.

Operators are concrete text interventions. They never receive feature values
and prompts never ask an LLM to lower a rubric. Rubrics select the audit axis;
features are measured only after generation.
"""

from __future__ import annotations

import random
import re
from collections import Counter
from difflib import SequenceMatcher
from itertools import combinations, permutations
from typing import Any, Mapping

from feak_tc.diagnose.stub import split_sentences
from feak_tc.mvp.validity import is_complete_sentence


MAIN_CHAIN_OPERATORS = (
    "DELETE_SPECIFICS",
    "SHUFFLE_FLOW",
    "INSERT_OFFTOPIC",
    "INJECT_LEX_REPEAT",
)
SURFACE_VALIDATION_OPERATOR = "INJECT_GRAMMAR_ERR"
LLM_ONLY_OPERATORS = ("INSERT_OFFTOPIC", "INJECT_LEX_REPEAT")

OPERATOR_SPECS: dict[str, dict[str, Any]] = {
    "DELETE_SPECIFICS": {
        "reverse_action": "ADD_DETAIL",
        "intent": "ADD_SUPPORTING_EXPLANATION",
        "target_rubric": "content_2",
        "preserve_constraints": (
            "문장 문법을 바꾸지 않는다.",
            "글의 전체 논지와 남은 문장은 바꾸지 않는다.",
        ),
    },
    "SHUFFLE_FLOW": {
        "reverse_action": "RESTRUCTURE",
        "intent": "RESTORE_LOGICAL_ORDER",
        "target_rubric": "organization_1",
        "preserve_constraints": (
            "문장의 내용과 문법을 바꾸지 않는다.",
            "문장 자체는 그대로 두고 위치만 이동한다.",
        ),
    },
    "INSERT_OFFTOPIC": {
        "reverse_action": "DELETE_OR_FOCUS",
        "intent": "REMOVE_REDUNDANCY",
        "target_rubric": "organization_2",
        "preserve_constraints": (
            "기존 문장과 기존 문법을 바꾸지 않는다.",
            "원래 글의 사실과 주장을 수정하지 않는다.",
        ),
    },
    "INJECT_LEX_REPEAT": {
        "reverse_action": "STYLE_REFINE",
        "intent": "REFINE_FORMAL_STYLE",
        "target_rubric": "expression_1",
        "preserve_constraints": (
            "핵심 주장과 사실을 바꾸지 않는다.",
            "기존 문장은 그대로 보존하고 어휘 반복 표현만 추가한다.",
        ),
    },
    "INJECT_GRAMMAR_ERR": {
        "reverse_action": "STYLE_REFINE",
        "intent": "REFINE_FORMAL_STYLE",
        "target_rubric": "expression_2",
        "preserve_constraints": (
            "내용과 어휘 선택을 바꾸지 않는다.",
            "문장 순서와 핵심 사실을 바꾸지 않는다.",
            "어법·조사·맞춤법 오류만 주입한다.",
        ),
    },
}

_VARIANT_GUIDANCE = {
    "conservative": "최소한의 국소 편집으로 요청한 결함만 분명하게 만든다.",
    "student_natural": "실제 학생 글에서 생길 법한 자연스러운 결함으로 만들되 다른 품질 축은 건드리지 않는다.",
}

_COMMON_RULES = (
    "공통 규칙:\n"
    "- span 필드는 글에 정확히 그대로 존재하는 연속 부분 문자열이어야 한다.\n"
    "- 서로 다른 edit은 같은 문장을 건드리지 않는다.\n"
    "- 지정한 보존 조건을 반드시 지킨다.\n"
    "- 글의 다른 부분은 바꾸지 않는다.\n"
    "- JSON 객체만 반환한다."
)

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_WORD_RE = re.compile(r"[가-힣A-Za-z]{2,}")
_DETAIL_MARKERS = ("예를 들어", "예컨대", "가령", "사례", "통계", "조사", "때문", "따르면")
_FLOW_DEPENDENCY_PREFIXES = (
    "그래서",
    "따라서",
    "그러므로",
    "그러나",
    "하지만",
    "반면",
    "또한",
    "게다가",
    "예를 들어",
    "예컨대",
    "가령",
    "먼저",
    "다음으로",
    "마지막으로",
    "결국",
    "이처럼",
    "이러한",
    "이것",
    "이를",
    "이는",
    "그 결과",
    "이로 인해",
    "그로 인해",
    "반대로",
    "그런데",
    "한편",
    "비록",
    "이때",
)
def build_prompt(
    operator: str,
    text: str,
    question: str,
    spec: Mapping[str, Any],
    variant: str,
) -> str:
    """Build an operator-only prompt with explicit preservation constraints."""

    if operator not in OPERATOR_SPECS:
        raise ValueError(f"Unknown operator: {operator}")
    if variant not in _VARIANT_GUIDANCE:
        raise ValueError(f"Unknown prompt variant: {variant}")

    n = int(spec.get("edits_per_step", 1))
    preserve = "\n".join(f"- {item}" for item in spec["preserve_constraints"])
    header = (
        f"[질문]\n{question}\n\n[글]\n{text}\n\n"
        f"[보존 조건]\n{preserve}\n\n"
        f"[편집 방식]\n{_VARIANT_GUIDANCE[variant]}\n\n"
    )
    if operator == "DELETE_SPECIFICS":
        task = (
            f"구체적인 사례·근거·수치를 담은 서로 다른 완결 문장 {n}개를 고른다. "
            "그 문장만 삭제하면 전체 논지는 유지되지만 뒷받침의 구체성이 낮아져야 한다.\n"
            "각 문자열은 원문에서 복사하고 문장부호까지 완전히 동일하게 반환한다.\n"
            '반환: {"target_spans": ["<삭제 문장>", "..."]}'
        )
    elif operator == "SHUFFLE_FLOW":
        task = (
            f"접속어·지시어·시간·인과 표현으로 앞 문장에 의존하는 완결 문장 {n}개를 골라 "
            "각각 원래 선행 문장에서 멀어지도록 다른 위치로 옮긴다. 문장 내용은 한 글자도 "
            "바꾸지 않고, 의존 관계가 실제로 끊어져 논리적 연결 순서가 어색해지게 한다. "
            "anchor는 moved 문장의 현재 바로 앞 문장이 아니어야 하며, 두 span 모두 원문에서 "
            "문장부호까지 그대로 복사한다.\n"
            '반환: {"moves": [{"moved_span": "<옮길 문장>", '
            '"anchor_span": "<이 문장 뒤로 이동>"}, ...]}'
        )
    elif operator == "INSERT_OFFTOPIC":
        task = (
            f"글의 주제·과제와 무관한 여담 문장 {n}개를 서로 다른 위치에 삽입한다. "
            "각 여담은 15~80자의 간결한 한 문장으로 쓰고, 기존 내용을 반박하거나 "
            "수정하지 말고 통일성만 낮춘다.\n"
            '반환: {"edits": [{"anchor_span": "<이 문장 뒤에 삽입>", '
            '"insertion": "<주제와 무관한 완결 문장>"}, ...]}'
        )
    elif operator == "INJECT_LEX_REPEAT":
        task = (
            f"서로 다른 기존 문장 {n}개를 anchor로 고르고, 각 문장 뒤에 같은 핵심 어휘를 "
            "부자연스럽게 세 번 이상 정확히 같은 표기로 반복하는 한 문장을 추가한다. "
            "단어만 나열하지 말고 서술어와 종결어미를 포함한 15~60자의 완결문으로 쓴다. "
            "숫자를 새로 쓰지 않고 반드시 온점·물음표·느낌표로 끝내며 새 사실은 만들지 않는다.\n"
            '반환: {"edits": [{"anchor_span": "<기존 문장>", '
            '"repetition": "<어휘 반복이 심한 완결 문장>"}, ...]}'
        )
    elif operator == "INJECT_GRAMMAR_ERR":
        task = (
            f"서로 다른 완결 문장 {n}개를 골라 내용·어휘·수치·순서는 유지하면서 조사 중복, "
            "호응 오류, 맞춤법 오류 중 하나만 주입한 문장으로 바꾼다. replacement는 반드시 "
            "target_span과 달라야 하고 완결 문장부호는 유지한다. target_span은 원문에서 "
            "문장부호까지 그대로 복사한다.\n"
            '반환: {"edits": [{"target_span": "<원래 문장>", '
            '"replacement": "<어법 오류가 생긴 문장>", '
            '"error_type": "<particle_swap|spacing|spelling_typo>"}, ...]}'
        )
    else:  # pragma: no cover - guarded above
        raise ValueError(operator)
    return header + task + "\n\n" + _COMMON_RULES


def build_payload_schema(
    operator: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the strict JSON Schema for one operator payload."""

    n = int(spec.get("edits_per_step", 1))
    string = {"type": "string"}
    if operator == "DELETE_SPECIFICS":
        properties = {
            "target_spans": _fixed_array(string, n),
        }
    elif operator == "SHUFFLE_FLOW":
        properties = {
            "moves": _fixed_array(
                _strict_object(
                    {
                        "moved_span": string,
                        "anchor_span": string,
                    }
                ),
                n,
            )
        }
    elif operator == "INSERT_OFFTOPIC":
        properties = {
            "edits": _fixed_array(
                _strict_object(
                    {
                        "anchor_span": string,
                        "insertion": string,
                    }
                ),
                n,
            )
        }
    elif operator == "INJECT_LEX_REPEAT":
        properties = {
            "edits": _fixed_array(
                _strict_object(
                    {
                        "anchor_span": string,
                        "repetition": string,
                    }
                ),
                n,
            )
        }
    elif operator == "INJECT_GRAMMAR_ERR":
        properties = {
            "edits": _fixed_array(
                _strict_object(
                    {
                        "target_span": string,
                        "replacement": string,
                        "error_type": {
                            "type": "string",
                            "enum": ["particle_swap", "spacing", "spelling_typo"],
                        },
                    }
                ),
                n,
            )
        }
    else:
        raise ValueError(f"Unknown operator: {operator}")
    return _strict_object(properties)


def generate_rule_payload(
    operator: str,
    text: str,
    spec: Mapping[str, Any],
    rng: random.Random,
    source_text: str | None = None,
) -> dict[str, Any]:
    """Generate a deterministic corruption payload without feature targets."""

    sentences = split_sentences(text)
    source_text = source_text or text
    n = int(spec.get("edits_per_step", 1))
    if len(sentences) < max(4, n + 2):
        raise ValueError(f"not enough sentences for {operator}: {len(sentences)}")

    if operator == "DELETE_SPECIFICS":
        candidates = [
            sentence for sentence in sentences[1:-1] if sentence in source_text
        ]
        max_removed = 0.35 * len(text)
        viable = [
            spans
            for spans in combinations(candidates, n)
            if sum(len(span) for span in spans) <= max_removed
        ]
        if not viable:
            raise ValueError("no deletion set satisfies the source-length preservation budget")
        selected = max(
            viable,
            key=lambda spans: (
                sum(_specificity_score(span) for span in spans),
                sum(len(span) for span in spans),
            ),
        )
        return {"target_spans": list(selected)}

    if operator == "SHUFFLE_FLOW":
        candidates = sorted(
            [
                sentence
                for index, sentence in enumerate(sentences)
                if index > 0
                and sentences.count(sentence) == 1
                and sentence in source_text
                and is_complete_sentence(sentence)
            ],
            key=lambda sentence: (
                bool(_flow_dependency_markers(sentence)),
                -sentences.index(sentence),
            ),
            reverse=True,
        )
        if len(candidates) < n:
            raise ValueError(
                f"not enough movable sentences for SHUFFLE_FLOW: "
                f"{len(candidates)} < {n}"
            )
        payload = _find_effective_shuffle_payload(
            sentences,
            candidates,
            n,
            source_text,
        )
        if payload is None:
            raise ValueError("no move set can break two source-order dependencies")
        return payload

    if operator in LLM_ONLY_OPERATORS:
        raise ValueError(f"{operator} is LLM-only in corruption rule v4")

    if operator == "INJECT_GRAMMAR_ERR":
        if n > len(_GRAMMAR_ERROR_TYPES):
            raise ValueError(
                f"grammar rule supports at most {len(_GRAMMAR_ERROR_TYPES)} distinct error types"
            )
        candidates = sorted(
            sentences,
            key=lambda sentence: len(_WORD_RE.findall(sentence)),
            reverse=True,
        )
        edits = []
        used: set[str] = set()
        for error_type in _GRAMMAR_ERROR_TYPES[:n]:
            for span in candidates:
                if span in used:
                    continue
                replacement = _inject_grammar_error(span, error_type)
                if replacement is None:
                    continue
                edits.append(
                    {
                        "target_span": span,
                        "replacement": replacement,
                        "error_type": error_type,
                    }
                )
                used.add(span)
                break
            else:
                raise ValueError(f"no sentence supports grammar error type: {error_type}")
        return {"edits": edits}

    raise ValueError(f"Unknown operator: {operator}")


def parse_and_apply(
    operator: str,
    payload: Mapping[str, Any],
    text: str,
    source_text: str,
    validity_cfg: Mapping[str, Any],
    spec: Mapping[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Apply one controlled intervention and return exact edit records."""

    if operator not in OPERATOR_SPECS:
        raise ValueError(f"Unknown operator: {operator}")
    expected = int(spec.get("edits_per_step", 1))
    new_text = text
    edits: list[dict[str, str]] = []

    if operator == "DELETE_SPECIFICS":
        spans = payload.get("target_spans")
        _require_count(spans, expected)
        _require_distinct(spans)
        normalized_spans = [
            _require_span({"target_span": raw_span}, "target_span", new_text)
            for raw_span in spans
        ]
        if any(span not in source_text for span in normalized_spans):
            raise ValueError(
                "DELETE_SPECIFICS confound: target must originate in the source text"
            )
        _require_clean_delete_spans(new_text, normalized_spans, validity_cfg)
        for span in normalized_spans:
            new_text = _replace_once(new_text, span, "")
            edits.append({"operation": "delete", "target_span": span, "text": ""})

    elif operator == "SHUFFLE_FLOW":
        moves = payload.get("moves")
        _require_count(moves, expected)
        _require_distinct([raw.get("moved_span", "") for raw in moves])
        for raw in moves:
            moved = _require_span(raw, "moved_span", new_text)
            anchor = _require_span(raw, "anchor_span", new_text)
            _require_source_origin(moved, source_text, operator, "moved_span")
            _require_source_origin(anchor, source_text, operator, "anchor_span")
            if moved == anchor or moved in anchor or anchor in moved:
                raise ValueError("moved_span and anchor_span must be distinct")
            without = _replace_once(new_text, moved, "")
            if anchor not in without:
                raise ValueError("anchor_span lost after removing moved_span")
            candidate = _normalize_spaces(_replace_once(without, anchor, anchor + " " + moved))
            if candidate == _normalize_spaces(new_text):
                raise ValueError("move produced no change")
            new_text = candidate
            edits.append({"operation": "move_after", "target_span": moved, "text": anchor})

    elif operator == "INSERT_OFFTOPIC":
        raw_edits = payload.get("edits")
        _require_count(raw_edits, expected)
        used_anchors: set[str] = set()
        for raw in raw_edits:
            requested_anchor = str(raw.get("anchor_span", "")).strip()
            anchor, repaired = _resolve_source_anchor(
                requested_anchor,
                new_text,
                source_text,
                used_anchors,
            )
            used_anchors.add(anchor)
            insertion = str(raw.get("insertion", "")).strip()
            _require_insertion(insertion, text, validity_cfg)
            new_text = _replace_once(new_text, anchor, anchor + " " + insertion)
            edit = {"operation": "insert_after", "target_span": anchor, "text": insertion}
            if repaired:
                edit.update(
                    {
                        "anchor_repaired": True,
                        "requested_target_span": requested_anchor,
                    }
                )
            edits.append(edit)

    elif operator == "INJECT_LEX_REPEAT":
        raw_edits = payload.get("edits")
        _require_count(raw_edits, expected)
        used_anchors = set()
        for raw in raw_edits:
            requested_anchor = str(raw.get("anchor_span", "")).strip()
            anchor, repaired = _resolve_source_anchor(
                requested_anchor,
                new_text,
                source_text,
                used_anchors,
            )
            used_anchors.add(anchor)
            repetition = str(raw.get("repetition", "")).strip()
            _require_insertion(repetition, text, validity_cfg)
            if _max_word_frequency(repetition) < 3:
                raise ValueError("lexical repetition must repeat a word at least three times")
            new_text = _replace_once(new_text, anchor, anchor + " " + repetition)
            edit = {"operation": "insert_after", "target_span": anchor, "text": repetition}
            if repaired:
                edit.update(
                    {
                        "anchor_repaired": True,
                        "requested_target_span": requested_anchor,
                    }
                )
            edits.append(edit)

    elif operator == "INJECT_GRAMMAR_ERR":
        raw_edits = payload.get("edits")
        _require_count(raw_edits, expected)
        _require_distinct([raw.get("target_span", "") for raw in raw_edits])
        for raw in raw_edits:
            span = _require_span(raw, "target_span", new_text)
            replacement = str(raw.get("replacement", "")).strip()
            _require_grammar_replacement(span, replacement, validity_cfg)
            new_text = _replace_once(new_text, span, replacement)
            edit = {"operation": "replace", "target_span": span, "text": replacement}
            error_type = str(raw.get("error_type", "")).strip()
            if error_type:
                edit["error_type"] = error_type
            edits.append(edit)

    new_text = _normalize_spaces(new_text)
    lo = float(validity_cfg["min_ratio_vs_source"]) * len(source_text)
    hi = float(validity_cfg["max_ratio_vs_source"]) * len(source_text)
    if not lo <= len(new_text) <= hi:
        raise ValueError(f"state length {len(new_text)} outside [{lo:.0f}, {hi:.0f}]")
    if new_text == _normalize_spaces(text):
        raise ValueError("no-effect corruption step")
    return new_text, edits


def validate_operator_preservation(
    operator: str,
    before_text: str,
    after_text: str,
    edits: list[Mapping[str, Any]],
) -> dict[str, Any]:
    """Validate operator-specific preservation beyond rubric score changes."""

    before = _normalize_spaces(before_text)
    after = _normalize_spaces(after_text)

    if operator == "DELETE_SPECIFICS":
        expected = before
        deleted_spans = []
        for edit in edits:
            if str(edit.get("operation")) != "delete":
                raise ValueError("DELETE_SPECIFICS preservation requires delete edits only")
            target = str(edit.get("target_span", "")).strip()
            if not target or target not in expected:
                raise ValueError("DELETE_SPECIFICS target span is missing from source")
            expected = _replace_once(expected, target, "")
            deleted_spans.append(target)
        expected = _normalize_spaces(expected)
        if after != expected:
            raise ValueError(
                "DELETE_SPECIFICS preservation violation: text outside target spans changed"
            )
        return {
            "passed": True,
            "method": "exact_target_span_subtraction",
            "deleted_spans": len(deleted_spans),
        }

    if operator == "SHUFFLE_FLOW":
        before_sentences = split_sentences(before)
        after_sentences = split_sentences(after)
        if Counter(before_sentences) != Counter(after_sentences):
            raise ValueError(
                "SHUFFLE_FLOW preservation violation: sentence text or membership changed"
            )

        dependency_breaks = []
        for edit in edits:
            if str(edit.get("operation")) != "move_after":
                raise ValueError("SHUFFLE_FLOW preservation requires move_after edits only")
            moved = str(edit.get("target_span", "")).strip()
            if moved not in before_sentences or moved not in after_sentences:
                raise ValueError("SHUFFLE_FLOW moved sentence is missing")
            markers = _flow_dependency_markers(moved) or ["implicit_adjacency"]
            before_index = before_sentences.index(moved)
            after_index = after_sentences.index(moved)
            if before_index == 0:
                raise ValueError("SHUFFLE_FLOW moved sentence has no original predecessor")
            original_predecessor = before_sentences[before_index - 1]
            new_predecessor = after_sentences[after_index - 1] if after_index > 0 else None
            if new_predecessor == original_predecessor:
                raise ValueError(
                    "SHUFFLE_FLOW ineffective: original discourse dependency remains adjacent"
                )
            dependency_breaks.append(
                {
                    "markers": markers,
                    "original_predecessor": original_predecessor,
                    "new_predecessor": new_predecessor,
                }
            )
        return {
            "passed": True,
            "method": "sentence_multiset_and_discourse_dependency",
            "dependency_breaks": dependency_breaks,
        }

    return {"passed": True, "method": "operator_postcondition"}


def validate_normalized_corruption(
    operator: str,
    edits: list[Mapping[str, Any]],
    normalized_text: str,
) -> None:
    """Reject normalization output that removes or rewrites the corruption."""

    for index, edit in enumerate(edits, 1):
        operation = str(edit.get("operation", ""))
        target = str(edit.get("target_span", ""))
        inserted = str(edit.get("text", ""))
        survived = False

        if operation == "delete":
            survived = bool(target) and target not in normalized_text
        elif operation == "move_after":
            survived = (
                bool(target)
                and bool(inserted)
                and target in normalized_text
                and inserted in normalized_text
                and normalized_text.index(inserted) < normalized_text.index(target)
            )
        elif operation == "insert_after":
            survived = bool(target) and bool(inserted) and f"{target} {inserted}" in normalized_text
        elif operation == "replace":
            survived = (
                bool(target)
                and bool(inserted)
                and inserted in normalized_text
                and target not in normalized_text
            )

        if not survived:
            raise ValueError(
                f"normalization removed or rewrote {operator} corruption edit {index}"
            )


def _require_count(items: Any, expected: int) -> None:
    if not isinstance(items, list):
        raise ValueError("edits must be a list")
    if len(items) != expected:
        raise ValueError(f"expected {expected} edits, got {len(items)}")


def _require_distinct(items: list[Any]) -> None:
    values = [str(item).strip() for item in items]
    if any(not item for item in values) or len(values) != len(set(values)):
        raise ValueError("edit spans must be non-empty and distinct")


def _require_clean_delete_spans(
    text: str,
    spans: list[str],
    validity_cfg: Mapping[str, Any],
) -> None:
    """Reject deletions that would also remove an obvious repetition defect."""

    repeated_size = int(validity_cfg.get("delete_repeated_ngram_size", 5))
    repeated_occurrences = int(
        validity_cfg.get("delete_repeated_ngram_occurrences", 2)
    )
    for span in spans:
        counts = Counter(_word_ngrams(span, repeated_size))
        if counts and max(counts.values()) >= repeated_occurrences:
            repeated = " ".join(counts.most_common(1)[0][0])
            raise ValueError(
                "DELETE_SPECIFICS confound: target contains repeated "
                f"{repeated_size}-gram: {repeated}"
            )

    remaining = text
    for span in spans:
        remaining = _replace_once(remaining, span, "")
    cross_size = int(validity_cfg.get("delete_cross_text_ngram_size", 8))
    remaining_ngrams = set(_word_ngrams(remaining, cross_size))
    for span in spans:
        overlap = remaining_ngrams & set(_word_ngrams(span, cross_size))
        if overlap:
            repeated = " ".join(sorted(overlap)[0])
            raise ValueError(
                "DELETE_SPECIFICS confound: target duplicates retained text "
                f"at {cross_size}-gram: {repeated}"
            )


def _word_ngrams(text: str, size: int) -> list[tuple[str, ...]]:
    words = _WORD_RE.findall(text.lower())
    if size <= 0:
        return []
    return [
        tuple(words[index : index + size])
        for index in range(max(0, len(words) - size + 1))
    ]


def _require_span(payload: Mapping[str, Any], key: str, text: str) -> str:
    span = str(payload.get(key, "")).strip()
    if not span:
        raise ValueError(f"missing {key}")
    if span not in text:
        raise ValueError(f"{key} is not an exact substring")
    return span


def _require_source_origin(
    span: str,
    source_text: str,
    operator: str,
    field: str,
) -> None:
    if span not in source_text:
        raise ValueError(
            f"{operator} confound: {field} must originate in the source text"
        )


def _resolve_source_anchor(
    requested: str,
    current_text: str,
    source_text: str,
    used: set[str],
) -> tuple[str, bool]:
    """Use an exact source anchor, repairing only the LLM's placement field."""

    if requested and requested in current_text and requested in source_text and requested not in used:
        return requested, False
    candidates = [
        sentence
        for sentence in split_sentences(source_text)
        if sentence in current_text and sentence not in used and is_complete_sentence(sentence)
    ]
    if not candidates:
        raise ValueError("no unused source-origin insertion anchor remains")
    if not requested:
        return candidates[len(candidates) // 2], True
    anchor = max(
        candidates,
        key=lambda candidate: (
            SequenceMatcher(a=requested, b=candidate).ratio(),
            len(candidate),
        ),
    )
    return anchor, True


def _require_insertion(
    insertion: str,
    original_text: str,
    validity_cfg: Mapping[str, Any],
) -> None:
    lo = int(validity_cfg["insertion_min_chars"])
    hi = int(validity_cfg["insertion_max_chars"])
    if not lo <= len(insertion) <= hi:
        raise ValueError(f"insertion length {len(insertion)} outside [{lo}, {hi}]")
    if insertion in original_text:
        raise ValueError("insertion already exists in text")
    if not is_complete_sentence(insertion):
        raise ValueError("insertion is not a complete sentence")
    if _NUMBER_RE.search(insertion):
        raise ValueError("insertion must not introduce numeric facts")


def _require_grammar_replacement(
    before: str,
    after: str,
    validity_cfg: Mapping[str, Any],
) -> None:
    if not after or after == before:
        raise ValueError("empty or unchanged grammar replacement")
    if not is_complete_sentence(after):
        raise ValueError("grammar replacement must remain a complete sentence")
    if set(_NUMBER_RE.findall(before)) != set(_NUMBER_RE.findall(after)):
        raise ValueError("grammar replacement changed numeric expressions")
    before_words = set(_WORD_RE.findall(before))
    after_words = set(_WORD_RE.findall(after))
    retention = len(before_words & after_words) / max(1, len(before_words))
    minimum = float(validity_cfg.get("grammar_min_token_retention", 0.6))
    if retention < minimum:
        raise ValueError(f"grammar replacement token retention {retention:.3f} below {minimum}")
    if not 0.6 <= len(after) / max(1, len(before)) <= 1.4:
        raise ValueError("grammar replacement changed sentence length too much")


def _replace_once(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1)


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())


def _specificity_score(sentence: str) -> int:
    marker_hits = sum(marker in sentence for marker in _DETAIL_MARKERS)
    return marker_hits * 2 + int(bool(_NUMBER_RE.search(sentence)))


def _find_effective_shuffle_payload(
    sentences: list[str],
    candidates: list[str],
    count: int,
    source_text: str,
) -> dict[str, Any] | None:
    """Find sequential moves whose final order breaks every old adjacency."""

    movable_sets = list(combinations(candidates[: min(len(candidates), 8)], count))
    for moved in movable_sets:
        original_predecessors = {
            sentence: sentences[sentences.index(sentence) - 1] for sentence in moved
        }
        anchor_pool = [
            sentence
            for sentence in sentences
            if sentence not in moved
            and sentence in source_text
            and sentences.count(sentence) == 1
            and is_complete_sentence(sentence)
        ]
        ranked_assignments = sorted(
            permutations(anchor_pool, count),
            key=lambda anchors: sum(
                abs(sentences.index(anchor) - sentences.index(sentence))
                for sentence, anchor in zip(moved, anchors)
            ),
            reverse=True,
        )
        for anchors in ranked_assignments:
            if any(
                anchor == original_predecessors[sentence]
                for sentence, anchor in zip(moved, anchors)
            ):
                continue
            current = list(sentences)
            effective = True
            for sentence, anchor in zip(moved, anchors):
                before = list(current)
                current.remove(sentence)
                current.insert(current.index(anchor) + 1, sentence)
                if current == before:
                    effective = False
                    break
            if not effective:
                continue
            if all(
                current.index(sentence) > 0
                and current[current.index(sentence) - 1]
                != original_predecessors[sentence]
                for sentence in moved
            ):
                return {
                    "moves": [
                        {"moved_span": sentence, "anchor_span": anchor}
                        for sentence, anchor in zip(moved, anchors)
                    ]
                }
    return None


def _flow_dependency_markers(sentence: str) -> list[str]:
    compact = sentence.lstrip(" \t\"'“‘(")
    return [marker for marker in _FLOW_DEPENDENCY_PREFIXES if compact.startswith(marker)]


_GRAMMAR_ERROR_TYPES = ("particle_swap", "spacing", "spelling_typo")
_PARTICLE_SWAPS = {
    "은": "는",
    "는": "은",
    "이": "가",
    "가": "이",
    "을": "를",
    "를": "을",
}
_PREDICATE_LIKE_STEM_ENDINGS = (
    "하",
    "되",
    "있",
    "없",
    "않",
    "싶",
    "남",
)
_SPELLING_TYPOS = (
    ("습니다", "슴니다"),
    ("있다", "잇다"),
    ("없다", "업다"),
    ("했다", "햇다"),
    ("됐다", "됬다"),
)


def _inject_grammar_error(sentence: str, error_type: str) -> str | None:
    if error_type == "particle_swap":
        # Prefer object/subject particles and require a multi-syllable nominal
        # stem. This avoids treating adnominal endings such as 남는→남은 as
        # particles while still producing a locally obvious 조사 mismatch.
        for particle_group in (("을", "를"), ("이", "가"), ("은", "는")):
            alternatives = "|".join(particle_group)
            pattern = re.compile(
                rf"(?P<stem>[가-힣]{{2,}})(?P<particle>{alternatives})(?=\s|[,.!?])"
            )
            for match in pattern.finditer(sentence):
                stem = match.group("stem")
                if stem.endswith(_PREDICATE_LIKE_STEM_ENDINGS):
                    continue
                particle = match.group("particle")
                start, end = match.span("particle")
                return sentence[:start] + _PARTICLE_SWAPS[particle] + sentence[end:]
        return None

    if error_type == "spacing":
        match = re.search(r"(?<=[가-힣])\s+(?=[가-힣])", sentence)
        if not match:
            return None
        return sentence[: match.start()] + sentence[match.end() :]

    if error_type == "spelling_typo":
        for correct, typo in _SPELLING_TYPOS:
            if correct in sentence:
                return sentence.replace(correct, typo, 1)
        return _inject_batchim_typo(sentence)

    raise ValueError(f"unknown grammar error type: {error_type}")


def _inject_batchim_typo(sentence: str) -> str | None:
    """Replace one 받침 with an implausible one to create an obvious typo."""

    match = re.search(r"[가-힣]{3,}", sentence)
    if not match:
        return None
    chars = list(match.group())
    index = max(0, len(chars) - 2)
    code = ord(chars[index]) - 0xAC00
    if not 0 <= code < 11172:
        return None
    initial_vowel, final = divmod(code, 28)
    replacement_final = 19 if final != 19 else 7  # ㅅ, or ㄷ when already ㅅ
    chars[index] = chr(0xAC00 + initial_vowel * 28 + replacement_final)
    return sentence[: match.start()] + "".join(chars) + sentence[match.end() :]


def _strict_object(properties: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": dict(properties),
        "required": list(properties),
        "additionalProperties": False,
    }


def _fixed_array(items: Mapping[str, Any], count: int) -> dict[str, Any]:
    return {
        "type": "array",
        "items": dict(items),
        "minItems": count,
        "maxItems": count,
    }


def _max_word_frequency(text: str) -> int:
    words = [word for word in _WORD_RE.findall(text) if len(word) >= 2]
    counts = Counter(words)
    compact = re.sub(r"\s+", "", text)
    substring_max = max((compact.count(word) for word in words), default=0)
    return max(max(counts.values(), default=0), substring_max)
