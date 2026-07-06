"""Deterministic local diagnoser for MVP tests and offline development."""

from __future__ import annotations

import math
import re
from collections import Counter

from .base import Diagnosis, select_weak_rubrics
from .constants import FEAK_FEATURE_NAMES, MAX_SCORE, MIN_SCORE


_TOKEN_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_SENTENCE_RE = re.compile(r"[^.!?\n。？！]+[.!?。？！]?")
_CONNECTORS = ("그리고", "그러나", "따라서", "또한", "예를 들어", "즉", "그래서")
_EXAMPLE_MARKERS = ("예를 들어", "사례", "예시", "구체적으로", "5.18", "헌법")
_KEYWORD_MARKERS = ("권리", "존중", "인간", "사람", "침해", "당연")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text)


def split_sentences(text: str) -> list[str]:
    sentences = [match.group(0).strip() for match in _SENTENCE_RE.finditer(text)]
    return [sentence for sentence in sentences if sentence]


def _clip_score(value: float) -> float:
    return float(max(MIN_SCORE, min(MAX_SCORE, round(value, 2))))


def _ratio(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


class StubDiagnoser:
    """Cheap scorer shaped like the Kanana diagnoser output.

    It is not a grading model. It exists so the controller can be developed and
    tested before loading the 8B Kanana scorer.
    """

    def __init__(self, weak_top_n: int = 3) -> None:
        self.weak_top_n = weak_top_n

    def diagnose(self, text: str) -> Diagnosis:
        features = _extract_stub_features(text)
        rubrics = _score_stub_rubrics(text, features)
        weak = select_weak_rubrics(rubrics, top_n=self.weak_top_n)
        return Diagnosis(
            text=text,
            rubrics=rubrics,
            features=features,
            weak_rubrics=weak,
            metadata={"diagnoser": "stub"},
        )


def _extract_stub_features(text: str) -> dict[str, float]:
    tokens = tokenize(text)
    sentences = split_sentences(text)
    token_counts = Counter(tokens)
    unique = set(tokens)
    content_tokens = [tok for tok in tokens if len(tok) > 1]
    endings = [tok for tok in tokens if tok.endswith(("다", "요", "함", "음"))]
    connectors = sum(text.count(connector) for connector in _CONNECTORS)
    examples = sum(text.count(marker) for marker in _EXAMPLE_MARKERS)
    keyword_hits = sum(text.count(marker) for marker in _KEYWORD_MARKERS)
    repeated = sum(count - 1 for count in token_counts.values() if count > 1)
    word_lengths = [len(tok) for tok in tokens] or [0]
    sent_lengths = [len(tokenize(sentence)) for sentence in sentences] or [0]
    avg_word_len = sum(word_lengths) / len(word_lengths)
    word_len_std = math.sqrt(sum((x - avg_word_len) ** 2 for x in word_lengths) / len(word_lengths))
    avg_sent_len = sum(sent_lengths) / len(sent_lengths)

    feature_values = {
        "C_Cnt": float(len(content_tokens)),
        "E_Cnt": float(len(endings)),
        "F_Cnt": float(max(0, len(tokens) - len(content_tokens))),
        "J_Cnt": float(sum(1 for tok in tokens if tok.endswith(("은", "는", "이", "가", "을", "를")))),
        "X_Cnt": float(examples),
        "char_Cnt": float(len(text.replace(" ", ""))),
        "word_Cnt": float(len(tokens)),
        "E_NDW": float(len(set(endings))),
        "morph_NDW": float(len(unique)),
        "morph_LenAvg": float(avg_word_len),
        "morph_LenStd": float(word_len_std),
        "word_LenStd": float(word_len_std),
        "grade_2_ratio": _ratio(sum(1 for tok in tokens if len(tok) <= 2), len(tokens)),
        "grade_3_ratio": _ratio(sum(1 for tok in tokens if len(tok) == 3), len(tokens)),
        "grade_4_ratio": _ratio(sum(1 for tok in tokens if len(tok) >= 4), len(tokens)),
        "grade_m1_ratio": _ratio(sum(1 for tok in tokens if len(tok) >= 6), len(tokens)),
        "N_MSTTR": _ratio(len(unique), len(tokens)),
        "V_HDD": _ratio(len([tok for tok in tokens if tok.endswith("다")]), len(tokens)),
        "lemma_MATTR": _ratio(len(unique), len(tokens)),
        "2-gram_NDW": float(max(0, len(tokens) - 1)),
        "NN_repRatio": _ratio(repeated, len(tokens)),
        "word_sentLenAvg": float(avg_sent_len),
        "2-gram_RTTR": math.sqrt(float(max(0, len(tokens) - 1))),
        "adjacent_sentence_overlap_function_lemmas": float(connectors),
        "char_paraLenAvg": float(len(text.replace("\n", ""))),
        "avgSentSimilarity": min(1.0, 0.15 + 0.1 * connectors + 0.04 * max(0, len(sentences) - 1)),
        "topicConsistency": min(1.0, 0.2 + 0.12 * keyword_hits + 0.08 * examples),
        "text_dalechall": float(4.0 + _ratio(sum(1 for tok in tokens if len(tok) >= 5), len(tokens)) * 4.0),
        "text_oridx": float(100.0 + len(unique) * 125.0),
    }
    return {name: float(feature_values.get(name, 0.0)) for name in FEAK_FEATURE_NAMES}


def _score_stub_rubrics(text: str, features: dict[str, float]) -> dict[str, float]:
    tokens = tokenize(text)
    sentences = split_sentences(text)
    connectors = sum(text.count(connector) for connector in _CONNECTORS)
    examples = sum(text.count(marker) for marker in _EXAMPLE_MARKERS)
    keyword_hits = sum(text.count(marker) for marker in _KEYWORD_MARKERS)
    unique_ratio = features["lemma_MATTR"]
    word_count = features["word_Cnt"]
    sentence_count = max(1, len(sentences))
    length_score = min(4.0, word_count / 20.0)

    return {
        "task_1": _clip_score(1.5 + length_score + min(3.0, keyword_hits * 0.7)),
        "content_1": _clip_score(2.0 + min(3.0, sentence_count * 0.8) + min(2.0, keyword_hits * 0.35)),
        "content_2": _clip_score(1.8 + length_score + min(3.0, examples * 1.1)),
        "content_3": _clip_score(2.0 + min(4.5, keyword_hits * 0.8)),
        "organization_1": _clip_score(2.2 + min(3.0, connectors * 0.9) + min(2.0, sentence_count * 0.35)),
        "organization_2": _clip_score(2.0 + min(4.0, features["topicConsistency"] * 4.0)),
        "expression_1": _clip_score(2.0 + min(4.0, unique_ratio * 5.0) + min(1.0, word_count / 80.0)),
        "expression_2": _clip_score(6.0 - text.count("  ") - text.count(",,") + min(1.5, connectors * 0.2)),
    }
