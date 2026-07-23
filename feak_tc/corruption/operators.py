"""Corruption operators — forward degradations that invert action types.

Each operator asks the LLM to pick spans and produce degraded text, then
applies the edits locally so the exact spans are recorded. The reverse
direction of every step is therefore a labeled (action_type, target spans)
transition for TVM training data.

Pilot v1 showed single-sentence corruption moves essay-level rubric scores
by less than scorer rerun noise, so every operator degrades several
sentences per step (counts in configs/corruption.yaml `intensity`).
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

DEFAULT_INTENSITY = {
    "drop_detail_spans": 2,
    "insert_offtopic_count": 2,
    "verbose_spans": 2,
    "shuffle_moves": 2,
    "flatten_spans": 4,
}

_COMMON_RULES = (
    "규칙:\n"
    "- span 필드는 모두 글에 '정확히 그대로' 존재하는 연속 부분 문자열이어야 한다 (복사해서 붙여넣기).\n"
    "- 서로 다른 edit이 같은 문장을 건드리면 안 된다.\n"
    "- 글의 다른 부분은 절대 바꾸지 않는다.\n"
    "- JSON 객체만 반환한다."
)


def build_prompt(operator: str, text: str, question: str, intensity: Mapping[str, Any]) -> str:
    header = f"[질문]\n{question}\n\n[글]\n{text}\n\n"
    if operator == "DROP_DETAIL":
        n = int(intensity.get("drop_detail_spans", 2))
        return header + (
            f"과제: 이 글에서 '구체적인 사례, 근거, 부연 설명'을 담은 문장 {n}개를 골라라. "
            "그 문장들이 삭제되면 글이 일반론만 남아 설명의 구체성이 크게 떨어져야 한다.\n"
            '반환 형식: {"target_spans": ["<삭제할 문장 1 (그대로 복사)>", "<삭제할 문장 2>"]}\n' + _COMMON_RULES
        )
    if operator == "INSERT_OFFTOPIC":
        n = int(intensity.get("insert_offtopic_count", 2))
        return header + (
            f"과제: 이 글의 주제와 직접 관련 없는 여담 문장 {n}개를 만들어, 글의 서로 다른 위치에 끼워 넣어라. "
            "여담은 글쓴이의 말투를 흉내 내되 내용은 주제에서 벗어나야 한다.\n"
            '반환 형식: {"edits": [{"anchor_span": "<이 문장 뒤에 삽입 (그대로 복사)>", '
            '"insertion": "<주제와 무관한 여담 문장>"}, ...]}\n' + _COMMON_RULES
        )
    if operator == "VERBOSE_REPEAT":
        n = int(intensity.get("verbose_spans", 2))
        return header + (
            f"과제: 이 글에서 문장 {n}개를 골라, 각각 같은 내용을 반복하고 군더더기를 붙여 장황한 2~3문장으로 바꿔라. "
            "의미는 유지하되 불필요하게 길어져야 한다.\n"
            '반환 형식: {"edits": [{"target_span": "<원래 문장 (그대로 복사)>", '
            '"replacement": "<장황하게 늘린 2~3문장>"}, ...]}\n' + _COMMON_RULES
        )
    if operator == "SHUFFLE_FLOW":
        n = int(intensity.get("shuffle_moves", 2))
        return header + (
            f"과제: 이 글에서 서로 다른 문장 {n}개를 골라 각각 다른 위치로 옮겨서 글의 논리 전개 순서가 부자연스러워지게 하라. "
            "문법적으로는 읽히지만 흐름(연결성)이 어색해지는 이동이어야 한다.\n"
            '반환 형식: {"moves": [{"moved_span": "<옮길 문장 (그대로 복사)>", '
            '"anchor_span": "<이 문장 바로 뒤로 이동 (그대로 복사)>"}, ...]}\n' + _COMMON_RULES
        )
    if operator == "FLATTEN_STYLE":
        n = int(intensity.get("flatten_spans", 4))
        return header + (
            f"과제: 이 글에서 문장 {n}개를 골라, 어휘가 단조롭고 종결어미가 똑같이 반복되며 표현이 어색한 버전으로 바꿔라. "
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
    intensity: Mapping[str, Any],
) -> tuple[str, list[dict[str, str]]]:
    """Apply the LLM-chosen edits locally. Raises ValueError when invalid."""

    new_text, edits = text, []
    if operator == "DROP_DETAIL":
        spans = payload.get("target_spans")
        _require_count(spans, int(intensity.get("drop_detail_spans", 2)))
        for raw_span in spans:
            span = _require_span({"target_span": raw_span}, "target_span", new_text)
            new_text = _replace_once(new_text, span, "")
            edits.append({"operation": "delete", "target_span": span, "text": ""})
    elif operator == "INSERT_OFFTOPIC":
        raw_edits = payload.get("edits")
        _require_count(raw_edits, int(intensity.get("insert_offtopic_count", 2)))
        lo, hi = int(validity_cfg["insertion_min_chars"]), int(validity_cfg["insertion_max_chars"])
        for raw in raw_edits:
            anchor = _require_span(raw, "anchor_span", new_text)
            insertion = str(raw.get("insertion", "")).strip()
            if not (lo <= len(insertion) <= hi):
                raise ValueError(f"insertion length {len(insertion)} outside [{lo}, {hi}]")
            if insertion in new_text:
                raise ValueError("insertion already exists in text")
            if not is_complete_sentence(insertion):
                raise ValueError("insertion is not a complete sentence")
            new_text = _replace_once(new_text, anchor, anchor + " " + insertion)
            edits.append({"operation": "insert_after", "target_span": anchor, "text": insertion})
    elif operator == "VERBOSE_REPEAT":
        raw_edits = payload.get("edits")
        _require_count(raw_edits, int(intensity.get("verbose_spans", 2)))
        for raw in raw_edits:
            span = _require_span(raw, "target_span", new_text)
            replacement = str(raw.get("replacement", "")).strip()
            if len(replacement) <= len(span):
                raise ValueError("replacement must be longer than target_span")
            if not is_complete_sentence(replacement):
                raise ValueError("replacement is not a complete sentence")
            new_text = _replace_once(new_text, span, replacement)
            edits.append({"operation": "replace", "target_span": span, "text": replacement})
    elif operator == "SHUFFLE_FLOW":
        moves = payload.get("moves")
        _require_count(moves, int(intensity.get("shuffle_moves", 2)))
        for raw in moves:
            moved = _require_span(raw, "moved_span", new_text)
            anchor = _require_span(raw, "anchor_span", new_text)
            if moved == anchor or moved in anchor or anchor in moved:
                raise ValueError("moved_span and anchor_span must be distinct sentences")
            without = _replace_once(new_text, moved, "")
            if anchor not in without:
                raise ValueError("anchor_span lost after removing moved_span")
            candidate = _normalize_spaces(_replace_once(without, anchor, anchor + " " + moved))
            if candidate == _normalize_spaces(new_text):
                raise ValueError("move produced no change")
            new_text = candidate
            edits.append({"operation": "move_after", "target_span": moved, "text": anchor})
    elif operator == "FLATTEN_STYLE":
        raw_edits = payload.get("edits")
        _require_count(raw_edits, int(intensity.get("flatten_spans", 4)), tolerance=1)
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
    if new_text == _normalize_spaces(text):
        raise ValueError("no-effect corruption step")
    return new_text, edits


def _require_count(items: Any, expected: int, tolerance: int = 0) -> None:
    if not isinstance(items, list):
        raise ValueError("edits must be a list")
    if not expected - tolerance <= len(items) <= expected + tolerance:
        raise ValueError(f"expected ~{expected} edits, got {len(items)}")


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
