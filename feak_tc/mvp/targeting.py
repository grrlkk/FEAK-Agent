"""Rubric-aware local target span selection for MVP proposals."""

from __future__ import annotations

from typing import Any, Optional

from feak_tc.diagnose.constants import RUBRIC_KEYS
from feak_tc.diagnose.stub import split_sentences, tokenize


_CONNECTORS = ("그리고", "그러나", "따라서", "또한", "즉", "그래서", "반면")
_EXAMPLE_MARKERS = ("예를 들어", "사례", "예시", "구체적으로", "5.18", "헌법")
_TOPIC_MARKERS = ("인권", "권리", "존중", "인간", "사람", "침해", "자유", "평등")
_STYLE_RED_FLAGS = (
    "  ",
    ",,",
    "맛있는 밥",
    "화장실을 자유롭게 갈 수 있다",
    "목숨을 가져갔다",
    "들고 일어났다",
)

_RUBRIC_TO_DEFAULT_ACTION = {
    "task_1": "ADD_DETAIL",
    "content_1": "ADD_DETAIL",
    "content_2": "ADD_DETAIL",
    "content_3": "DELETE_OR_FOCUS",
    "organization_1": "RESTRUCTURE",
    "organization_2": "DELETE_OR_FOCUS",
    "expression_1": "STYLE_REFINE",
    "expression_2": "STYLE_REFINE",
}


def select_target_span(
    text: str,
    target_rubric: str,
    action_type: Optional[str] = None,
) -> str:
    """Return the best sentence-level target for a local candidate edit."""

    ranked = rank_target_spans(text, target_rubric, action_type=action_type, limit=1)
    return str(ranked[0]["span"]) if ranked else ""


def rank_target_spans(
    text: str,
    target_rubric: str,
    action_type: Optional[str] = None,
    limit: int = 3,
) -> list[dict[str, Any]]:
    """Rank sentence spans with transparent MVP heuristics.

    This is not a learned evidence extractor. It gives the proposer a better
    local anchor than the previous shortest-sentence fallback.
    """

    if target_rubric not in RUBRIC_KEYS:
        raise ValueError(f"Unknown target_rubric: {target_rubric}")
    if action_type == "STOP":
        return []

    sentences = split_sentences(text)
    if not sentences:
        return []

    action = action_type or _RUBRIC_TO_DEFAULT_ACTION.get(target_rubric, "ADD_DETAIL")
    scored = [
        {
            "span": sentence,
            "index": idx,
            "score": round(_score_sentence(sentence, idx, len(sentences), target_rubric, action), 6),
        }
        for idx, sentence in enumerate(sentences)
    ]
    scored.sort(key=lambda item: (-float(item["score"]), int(item["index"])))
    return scored[: max(1, limit)]


def _score_sentence(
    sentence: str,
    index: int,
    sentence_count: int,
    target_rubric: str,
    action_type: str,
) -> float:
    tokens = tokenize(sentence)
    word_count = len(tokens)
    topic_hits = _count_markers(sentence, _TOPIC_MARKERS)
    example_hits = _count_markers(sentence, _EXAMPLE_MARKERS)
    connector_hits = _count_markers(sentence, _CONNECTORS)
    style_hits = _count_markers(sentence, _STYLE_RED_FLAGS)
    score = 0.0

    if action_type == "ADD_DETAIL":
        score += 2.0 if topic_hits else 0.4
        score += 1.2 if example_hits == 0 else -0.4
        score += _short_enough_bonus(word_count)
    elif action_type == "DELETE_OR_FOCUS":
        score += 2.0 if topic_hits == 0 else 0.3
        score += 0.7 if word_count >= 12 else 0.2
    elif action_type == "COMPRESS":
        score += min(3.0, word_count / 6.0)
    elif action_type == "RESTRUCTURE":
        score += 1.6 if index > 0 and connector_hits == 0 else 0.4
        score += 0.5 if sentence_count > 1 else 0.0
    elif action_type == "STYLE_REFINE":
        score += 2.0 if style_hits else 0.3
        score += 0.5 if word_count >= 10 else 0.0

    if target_rubric in {"task_1", "content_1", "content_2"}:
        score += 0.8 if topic_hits else 0.0
        score += 0.5 if example_hits == 0 else -0.2
    elif target_rubric in {"content_3", "organization_2"}:
        score += 0.9 if topic_hits == 0 else 0.1
    elif target_rubric == "organization_1":
        score += 0.9 if index > 0 and connector_hits == 0 else 0.0
    elif target_rubric.startswith("expression"):
        score += 1.0 if style_hits else 0.0

    return score


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _short_enough_bonus(word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    if word_count <= 12:
        return 0.8
    if word_count <= 20:
        return 0.3
    return -0.2
