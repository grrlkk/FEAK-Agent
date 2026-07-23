"""Corruption operators — forward degradations that invert action types.

Each operator asks the LLM to pick spans and produce degraded text, then
applies the edit locally so the exact spans are recorded. The reverse
direction of every step is therefore a labeled (action_type, target_span)
transition for TVM training data.
"""

from __future__ import annotations

from typing import Any, Mapping

from feak_tc.mvp.validity import is_complete_sentence


OPERATOR_SPECS: dict[str, dict[str, Any]] = {
    "DROP_DETAIL": {
        "reverse_action": "ADD_DETAIL",
        "intended_rubrics": ["content_2", "content_1"],
    },
    "INSERT_OFFTOPIC": {
        "reverse_action": "DELETE_OR_FOCUS",
        "intended_rubrics": ["content_3", "organization_2"],
    },
    "VERBOSE_REPEAT": {
        "reverse_action": "COMPRESS",
        # Length/repetition features are the primary axis; rubric axis is
        # confirmed empirically in the pilot report.
        "intended_rubrics": [],
    },
    "SHUFFLE_FLOW": {
        "reverse_action": "RESTRUCTURE",
        "intended_rubrics": ["organization_1"],
    },
    "FLATTEN_STYLE": {
        "reverse_action": "STYLE_REFINE",
        "intended_rubrics": ["expression_1", "expression_2"],
    },
}

_COMMON_RULES = (
    "규칙:\n"
    "- span 필드는 모두 글에 '정확히 그대로' 존재하는 연속 부분 문자열이어야 한다 (복사해서 붙여넣기).\n"
    "- 글의 다른 부분은 절대 바꾸지 않는다.\n"
    "- JSON 객체만 반환한다."
)


def build_prompt(operator: str, text: str, question: str) -> str:
    header = f"[질문]\n{question}\n\n[글]\n{text}\n\n"
    if operator == "DROP_DETAIL":
        return header + (
            "과제: 이 글에서 '구체적인 사례, 근거, 부연 설명'을 담은 문장 하나를 골라라. "
            "그 문장이 삭제되면 글이 일반론만 남아 설명의 구체성이 떨어지는 문장이어야 한다.\n"
            '반환 형식: {"target_span": "<삭제할 문장 (글에서 그대로 복사)>"}\n' + _COMMON_RULES
        )
    if operator == "INSERT_OFFTOPIC":
        return header + (
            "과제: 이 글의 주제와 직접 관련 없는 여담 문장 1개를 만들어, 글 중간의 한 문장 뒤에 끼워 넣어라. "
            "여담은 글쓴이의 말투를 흉내 내되 내용은 주제에서 벗어나야 한다.\n"
            '반환 형식: {"anchor_span": "<이 문장 뒤에 삽입 (글에서 그대로 복사)>", '
            '"insertion": "<주제와 무관한 여담 문장 1개>"}\n' + _COMMON_RULES
        )
    if operator == "VERBOSE_REPEAT":
        return header + (
            "과제: 이 글에서 문장 하나를 골라, 같은 내용을 반복하고 군더더기를 붙여 장황한 2~3문장으로 바꿔라. "
            "의미는 유지하되 불필요하게 길어져야 한다.\n"
            '반환 형식: {"target_span": "<원래 문장 (글에서 그대로 복사)>", '
            '"replacement": "<장황하게 늘린 2~3문장>"}\n' + _COMMON_RULES
        )
    if operator == "SHUFFLE_FLOW":
        return header + (
            "과제: 이 글에서 문장 하나를 골라 다른 위치로 옮겨서 글의 흐름(연결성)이 어색해지게 하라. "
            "문법적으로는 읽히지만 논리 전개 순서가 부자연스러워지는 이동을 골라라.\n"
            '반환 형식: {"moved_span": "<옮길 문장 (그대로 복사)>", '
            '"anchor_span": "<이 문장 바로 뒤로 이동 (그대로 복사, moved_span과 달라야 함)>"}\n' + _COMMON_RULES
        )
    if operator == "FLATTEN_STYLE":
        return header + (
            "과제: 이 글에서 문장 2~3개를 골라, 어휘가 단조롭고 종결어미가 똑같이 반복되며 표현이 어색한 버전으로 바꿔라. "
            "내용(의미)은 유지하고 길이는 비슷해야 하며, 표현의 질만 떨어져야 한다.\n"
            '반환 형식: {"edits": [{"target_span": "<원래 문장>", "replacement": "<표현이 나빠진 문장>"}, ...]}\n'
            + _COMMON_RULES
        )
    raise ValueError(f"Unknown operator: {operator}")


def parse_and_apply(
    operator: str,
    payload: Mapping[str, Any],
    text: str,
    source_text: str,
    validity_cfg: Mapping[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Apply the LLM-chosen edit locally. Raises ValueError when invalid."""

    if operator == "DROP_DETAIL":
        span = _require_span(payload, "target_span", text)
        edits = [{"operation": "delete", "target_span": span, "text": ""}]
        new_text = _replace_once(text, span, "")
    elif operator == "INSERT_OFFTOPIC":
        anchor = _require_span(payload, "anchor_span", text)
        insertion = str(payload.get("insertion", "")).strip()
        lo, hi = int(validity_cfg["insertion_min_chars"]), int(validity_cfg["insertion_max_chars"])
        if not (lo <= len(insertion) <= hi):
            raise ValueError(f"insertion length {len(insertion)} outside [{lo}, {hi}]")
        if insertion in text:
            raise ValueError("insertion already exists in text")
        if not is_complete_sentence(insertion):
            raise ValueError("insertion is not a complete sentence")
        edits = [{"operation": "insert_after", "target_span": anchor, "text": insertion}]
        new_text = _replace_once(text, anchor, anchor + " " + insertion)
    elif operator == "VERBOSE_REPEAT":
        span = _require_span(payload, "target_span", text)
        replacement = str(payload.get("replacement", "")).strip()
        if len(replacement) <= len(span):
            raise ValueError("replacement must be longer than target_span")
        if not is_complete_sentence(replacement):
            raise ValueError("replacement is not a complete sentence")
        edits = [{"operation": "replace", "target_span": span, "text": replacement}]
        new_text = _replace_once(text, span, replacement)
    elif operator == "SHUFFLE_FLOW":
        moved = _require_span(payload, "moved_span", text)
        anchor = _require_span(payload, "anchor_span", text)
        if moved == anchor or moved in anchor or anchor in moved:
            raise ValueError("moved_span and anchor_span must be distinct sentences")
        without = _replace_once(text, moved, "")
        if anchor not in without:
            raise ValueError("anchor_span lost after removing moved_span")
        new_text = _normalize_spaces(_replace_once(without, anchor, anchor + " " + moved))
        if _normalize_spaces(new_text) == _normalize_spaces(text):
            raise ValueError("move produced no change")
        edits = [{"operation": "move_after", "target_span": moved, "text": anchor}]
    elif operator == "FLATTEN_STYLE":
        raw_edits = payload.get("edits")
        if not isinstance(raw_edits, list) or not 2 <= len(raw_edits) <= 3:
            raise ValueError("edits must be a list of 2-3 items")
        new_text, edits = text, []
        for raw in raw_edits:
            span = _require_span(raw, "target_span", new_text)
            replacement = str(raw.get("replacement", "")).strip()
            if not replacement or replacement == span:
                raise ValueError("empty or unchanged replacement")
            if not 0.5 <= len(replacement) / len(span) <= 2.0:
                raise ValueError("style replacement changes length too much")
            new_text = _replace_once(new_text, span, replacement)
            edits.append({"operation": "replace", "target_span": span, "text": replacement})
    else:
        raise ValueError(f"Unknown operator: {operator}")

    new_text = _normalize_spaces(new_text)
    lo = float(validity_cfg["min_ratio_vs_source"]) * len(source_text)
    hi = float(validity_cfg["max_ratio_vs_source"]) * len(source_text)
    if not lo <= len(new_text) <= hi:
        raise ValueError(f"state length {len(new_text)} outside [{lo:.0f}, {hi:.0f}]")
    if _normalize_spaces(new_text) == _normalize_spaces(text):
        raise ValueError("no-effect corruption step")
    return new_text, edits


def _require_span(payload: Mapping[str, Any], key: str, text: str) -> str:
    span = str(payload.get(key, "")).strip()
    if not span:
        raise ValueError(f"missing {key}")
    if span not in text:
        raise ValueError(f"{key} is not an exact substring")
    return span


def _replace_once(text: str, old: str, new: str) -> str:
    return text.replace(old, new, 1)


def _normalize_spaces(text: str) -> str:
    return " ".join(text.split())
