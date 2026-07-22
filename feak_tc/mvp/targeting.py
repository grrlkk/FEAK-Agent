"""Rubric-aware local target span selection for MVP proposals."""

from __future__ import annotations

from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from feak_tc.diagnose.constants import RUBRIC_KEYS
from feak_tc.diagnose.stub import split_sentences, tokenize


TARGETING_CONFIG_PATH = Path(__file__).resolve().parents[2] / "configs" / "targeting.yaml"
_DEFAULT_CONNECTORS = ("그리고", "그러나", "따라서", "또한", "즉", "그래서", "반면", "하지만", "그러므로")
_DEFAULT_EXAMPLE_MARKERS = ("예를 들어", "예컨대", "가령", "사례", "예시", "구체적으로")
_DEFAULT_STYLE_RED_FLAGS = ("  ", ",,", "??", "!!", "것이다 것이다")
_DEFAULT_DEFINITION_MARKERS = (
    "의미한다",
    "의미하는",
    "뜻한다",
    "뜻하는",
    "말한다",
    "가리킨다",
    "정의한다",
    "정의된다",
    "개념이다",
    "개념이라",
    "개념으로",
)
_DEFAULT_DEFINITION_PENALTY = 4.0
_DEFAULT_DEFINITION_MIN_REPEATS = 2
_DEFAULT_STOPWORDS = (
    "다음",
    "대해",
    "서술하세요",
    "설명하세요",
    "자신의",
    "생각",
    "의견",
    "이유",
    "그리고",
    "그러나",
    "따라서",
    "또한",
    "있다",
    "한다",
    "것이다",
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
    question: Optional[str] = None,
) -> str:
    """Return the best sentence-level target for a local candidate edit."""

    ranked = rank_target_spans(text, target_rubric, action_type=action_type, limit=1, question=question)
    return str(ranked[0]["span"]) if ranked else ""


def rank_target_spans(
    text: str,
    target_rubric: str,
    action_type: Optional[str] = None,
    limit: int = 3,
    question: Optional[str] = None,
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
    cfg = _targeting_config()
    context = _targeting_context(text, question, cfg)
    context["protected_indices"] = _protected_definition_indices(sentences, question, cfg)
    scored = [
        {
            "span": sentence,
            "index": idx,
            "score": round(_score_sentence(sentence, idx, len(sentences), target_rubric, action, context), 6),
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
    context: Mapping[str, Any],
) -> float:
    tokens = tokenize(sentence)
    word_count = len(tokens)
    topic_overlap = _topic_overlap(tokens, context["topic_terms"])
    example_hits = _count_markers(sentence, context["example_markers"])
    connector_hits = _count_markers(sentence, context["connectors"])
    style_hits = _count_markers(sentence, context["style_red_flags"])
    repetition_hits = _repetition_hits(tokens)
    score = 0.0

    if action_type == "ADD_DETAIL":
        score += 0.4 + 1.8 * topic_overlap
        score += 1.2 if example_hits == 0 else -0.4
        score += _short_enough_bonus(word_count)
    elif action_type == "DELETE_OR_FOCUS":
        score += 2.0 * (1.0 - topic_overlap)
        score += 0.7 if word_count >= 12 else 0.2
    elif action_type == "COMPRESS":
        score += min(3.0, word_count / 6.0)
        score += min(1.0, repetition_hits * 0.5)
    elif action_type == "RESTRUCTURE":
        score += 1.6 if index > 0 and connector_hits == 0 else 0.4
        score += 0.5 if sentence_count > 1 else 0.0
    elif action_type == "STYLE_REFINE":
        score += 2.0 if style_hits or repetition_hits else 0.3
        score += 0.5 if word_count >= 10 else 0.0

    if action_type == "DELETE_OR_FOCUS" and index in context.get("protected_indices", set()):
        score -= float(context.get("definition_penalty", _DEFAULT_DEFINITION_PENALTY))

    if target_rubric in {"task_1", "content_1", "content_2"}:
        score += 0.8 * topic_overlap
        score += 0.5 if example_hits == 0 else -0.2
    elif target_rubric in {"content_3", "organization_2"}:
        score += 0.9 * (1.0 - topic_overlap)
    elif target_rubric == "organization_1":
        score += 0.9 if index > 0 and connector_hits == 0 else 0.0
    elif target_rubric.startswith("expression"):
        score += 1.0 if style_hits else 0.0

    return score


def is_protected_deletion_span(text: str, span: str, question: Optional[str] = None) -> bool:
    """Check whether a deletion span overlaps a core-term definition sentence.

    Used to reject DELETE_OR_FOCUS candidates that would remove the sentence
    defining a term central to the question or the essay itself.
    """

    span = span.strip()
    if not span:
        return False
    sentences = split_sentences(text)
    protected = _protected_definition_indices(sentences, question, _targeting_config())
    return any(sentences[idx] in span or span in sentences[idx] for idx in protected)


def _protected_definition_indices(
    sentences: list[str],
    question: Optional[str],
    cfg: Mapping[str, Any],
) -> set[int]:
    """Indices of sentences that define a core term of the question or essay.

    A sentence is protected when it introduces a term with the Korean
    definition pattern "X(이)란 …" or a definition marker, and that term is a
    core term: it appears in the question, or recurs across enough other
    sentences of the essay.
    """

    markers = cfg.get("definition_markers", _DEFAULT_DEFINITION_MARKERS)
    min_repeats = int(cfg.get("definition_min_repeats", _DEFAULT_DEFINITION_MIN_REPEATS))
    question_text = question or ""
    protected: set[int] = set()
    for idx, sentence in enumerate(sentences):
        heads = _definition_heads(sentence, markers)
        if not heads:
            continue
        for head in heads:
            if question_text and head in question_text:
                protected.add(idx)
                break
            other_hits = sum(1 for j, other in enumerate(sentences) if j != idx and head in other)
            if other_hits >= min_repeats:
                protected.add(idx)
                break
    return protected


def _definition_heads(sentence: str, markers: tuple[str, ...]) -> list[str]:
    """Terms this sentence defines via "X(이)란" or "X는 …<definition marker>".

    An "X(이)란" head alone is not enough: the sentence must also look like a
    definition (marker or copular shape), so noun-internal "란" (수정란) and
    idioms ("…이란 이유로/말은") do not count. The particle fallback
    ("X는 …") requires a lexical marker: copular shape alone would match
    nearly every declarative sentence, and "…라고 할 수 있다" alone marks
    conclusions, not definitions.
    """

    has_marker = any(marker in sentence for marker in markers)
    heads: list[str] = []
    tokens = tokenize(sentence)
    if has_marker or _has_copular_shape(sentence):
        for token in tokens:
            for suffix in ("이란", "란"):
                stem = token[: -len(suffix)] if token.endswith(suffix) else ""
                if len(stem) >= 2:
                    heads.append(stem)
                    break
    if not heads and has_marker:
        for token in tokens:
            for suffix in ("이라는", "라는", "은", "는"):
                stem = token[: -len(suffix)] if token.endswith(suffix) else ""
                if len(stem) >= 2:
                    heads.append(stem)
                    break
    return heads


def _has_copular_shape(sentence: str) -> bool:
    if "라고 할 수" in sentence or "라고 할수" in sentence:
        return True
    core = sentence.strip().rstrip(".!?。？！…\"'”’」』)").rstrip()
    return core.endswith("이다")


def _count_markers(text: str, markers: tuple[str, ...]) -> int:
    return sum(text.count(marker) for marker in markers)


def _topic_overlap(tokens: list[str], topic_terms: set[str]) -> float:
    if not tokens or not topic_terms:
        return 0.0
    sentence_terms = {_normalize_token(token) for token in tokens}
    overlap = len(sentence_terms & topic_terms)
    return min(1.0, overlap / max(1, min(3, len(topic_terms))))


def _targeting_context(text: str, question: Optional[str], cfg: Mapping[str, tuple[str, ...]]) -> dict[str, Any]:
    return {
        "connectors": cfg["connectors"],
        "example_markers": cfg["example_markers"],
        "style_red_flags": cfg["style_red_flags"],
        "definition_penalty": cfg["definition_penalty"],
        "topic_terms": _topic_terms(text, question, set(cfg["stopwords"])),
    }


def _topic_terms(text: str, question: Optional[str], stopwords: set[str]) -> set[str]:
    question_terms = _content_terms(question or "", stopwords)
    if question_terms:
        return set(question_terms)

    terms = _content_terms(text, stopwords)
    counts = Counter(terms)
    repeated = [term for term, count in counts.most_common(12) if count > 1]
    if repeated:
        return set(repeated)
    return {term for term, _ in counts.most_common(8)}


def _content_terms(text: str, stopwords: set[str]) -> list[str]:
    terms = []
    for token in tokenize(text):
        normalized = _normalize_token(token)
        if len(normalized) < 2:
            continue
        if normalized in stopwords:
            continue
        terms.append(normalized)
    return terms


def _normalize_token(token: str) -> str:
    return token.strip().lower()


def _repetition_hits(tokens: list[str]) -> int:
    counts = Counter(_normalize_token(token) for token in tokens)
    return sum(count - 1 for count in counts.values() if count > 1)


@lru_cache(maxsize=1)
def _targeting_config() -> dict[str, tuple[str, ...]]:
    payload: Mapping[str, Any] = {}
    if TARGETING_CONFIG_PATH.exists():
        import yaml

        with TARGETING_CONFIG_PATH.open(encoding="utf-8") as f:
            loaded = yaml.safe_load(f) or {}
        if isinstance(loaded, Mapping):
            payload = loaded
    return {
        "connectors": _tuple_config(payload, "connectors", _DEFAULT_CONNECTORS),
        "example_markers": _tuple_config(payload, "example_markers", _DEFAULT_EXAMPLE_MARKERS),
        "style_red_flags": _tuple_config(payload, "style_red_flags", _DEFAULT_STYLE_RED_FLAGS),
        "stopwords": _tuple_config(payload, "stopwords", _DEFAULT_STOPWORDS),
        "definition_markers": _tuple_config(payload, "definition_markers", _DEFAULT_DEFINITION_MARKERS),
        "definition_penalty": _float_config(payload, "definition_penalty", _DEFAULT_DEFINITION_PENALTY),
        "definition_min_repeats": _float_config(
            payload, "definition_min_repeats", _DEFAULT_DEFINITION_MIN_REPEATS
        ),
    }


def _float_config(payload: Mapping[str, Any], key: str, default: float) -> float:
    value = payload.get(key)
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _tuple_config(payload: Mapping[str, Any], key: str, default: tuple[str, ...]) -> tuple[str, ...]:
    value = payload.get(key)
    if not isinstance(value, list):
        return default
    items = [str(item) for item in value if str(item)]
    return tuple(items) if items else default


def _short_enough_bonus(word_count: int) -> float:
    if word_count <= 0:
        return 0.0
    if word_count <= 12:
        return 0.8
    if word_count <= 20:
        return 0.3
    return -0.2
