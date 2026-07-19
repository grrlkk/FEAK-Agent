"""Cheap structural patch validity checks run before re-diagnosis.

These are rule-based guards against obviously broken patches (sentence
fragments, span collapse, whole-essay shrinkage). Semantic judgements stay in
transition features and the selector; this module must not grow into a second
heuristic.
"""

from __future__ import annotations

import re
from typing import Any, Mapping, Optional

from .schemas import Candidate


DEFAULT_VALIDITY_CFG = {
    "enabled": True,
    "min_after_chars": 5,
    "replace_min_ratio": 0.3,
    "essay_min_ratio": 0.6,
    "require_sentence_ending": True,
    "reject_new_numbers": True,
    "reject_new_claim_markers": True,
    # 0 disables the near-duplicate sentence check. Both thresholds must be
    # met: min-side overlap 0.6 catches compress-without-delete duplication
    # (observed at ~0.66), and the Jaccard floor filters out short sentences
    # that merely share topic words with a longer neighbor (observed FP at
    # Jaccard 0.29 vs genuine duplicates at 0.36+).
    "near_dup_overlap": 0.6,
    "near_dup_jaccard": 0.35,
}

_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
# Phrases that present invented evidence as fact; only flagged when the patch
# introduces them, since original essays may legitimately cite sources.
_CLAIM_MARKERS = (
    "조사에 따르면",
    "통계에 따르면",
    "연구에 따르면",
    "조사 결과",
    "연구 결과",
    "설문",
    "통계청",
    "보고서",
)

_SENTENCE_FINAL_PUNCT = (".", "!", "?", "…", "。", "？", "！")
_CLOSING_QUOTES = "\"'”’」』)】]"
# Final syllables of common Korean sentence-final endings. A patched span that
# should be a full sentence must end in one of these (before punctuation);
# connective endings (-아/-어/-고/-며) and bare particles (-는/-이) fail here.
_FINAL_SYLLABLES = set("다까요죠네군오라자함음임됨냐지가")


def patch_validity_violations(
    text: str,
    cand: Candidate,
    cfg: Optional[Mapping[str, Any]] = None,
) -> list[str]:
    """Return reasons the patched candidate is structurally invalid."""

    merged = dict(DEFAULT_VALIDITY_CFG)
    if cfg:
        section = cfg.get("validity", cfg)
        if isinstance(section, Mapping):
            merged.update({k: section[k] for k in DEFAULT_VALIDITY_CFG if k in section})
    if not merged["enabled"]:
        return []

    patch = cand.patch
    if cand.action_type == "STOP" or patch is None or patch.operation == "noop":
        return []
    if cand.new_text is None:
        return ["validity:no_new_text"]

    reasons: list[str] = []
    if _normalize(cand.new_text) == _normalize(text):
        reasons.append("validity:no_effect_patch")
    if len(cand.new_text) < float(merged["essay_min_ratio"]) * len(text):
        reasons.append("validity:essay_collapse")

    before = patch.before.strip()
    after = patch.after.strip()
    if patch.operation in {"replace", "insert_after"}:
        if len(after) < int(merged["min_after_chars"]):
            reasons.append("validity:after_too_short")
        if (
            patch.operation == "replace"
            and cand.action_type != "DELETE_OR_FOCUS"
            and before
            and len(after) < float(merged["replace_min_ratio"]) * len(before)
        ):
            reasons.append("validity:span_collapse")
        if bool(merged["require_sentence_ending"]) and _must_be_full_sentence(patch.operation, before):
            if not is_complete_sentence(after):
                reasons.append("validity:sentence_fragment")

    if bool(merged["reject_new_numbers"]) and _new_numbers(text, cand.new_text):
        reasons.append("validity:fabricated_numbers")
    if bool(merged["reject_new_claim_markers"]) and _new_claim_markers(text, cand.new_text):
        reasons.append("validity:fabricated_claim_marker")
    overlap = float(merged["near_dup_overlap"])
    jaccard = float(merged["near_dup_jaccard"])
    if overlap > 0 and _introduced_near_duplicate(text, cand.new_text, overlap, jaccard):
        reasons.append("validity:near_duplicate_sentence")
    return reasons


def _new_numbers(text: str, new_text: str) -> set[str]:
    return set(_NUMBER_RE.findall(new_text)) - set(_NUMBER_RE.findall(text))


def _new_claim_markers(text: str, new_text: str) -> list[str]:
    return [m for m in _CLAIM_MARKERS if m in new_text and m not in text]


def _introduced_near_duplicate(
    text: str, new_text: str, overlap: float, jaccard: float
) -> bool:
    """Detect near-duplicate sentence pairs the patch introduced.

    Pairs where both sentences already existed in the original are ignored so
    naturally repetitive essays are not punished for pre-existing repetition.
    """

    original = {re.sub(r"\s+", "", s) for s in _split_sentences(text)}
    sentences = [re.sub(r"\s+", "", s) for s in _split_sentences(new_text)]
    sentences = [s for s in sentences if len(s) > 10]
    for i in range(len(sentences)):
        for j in range(i + 1, len(sentences)):
            if sentences[i] in original and sentences[j] in original:
                continue
            min_side, symmetric = _bigram_similarity(sentences[i], sentences[j])
            if min_side >= overlap and symmetric >= jaccard:
                return True
    return False


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?。？！])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def _bigram_similarity(a: str, b: str) -> tuple[float, float]:
    """Return (min-side overlap, Jaccard) of character bigram sets."""

    bigrams_a = {a[i : i + 2] for i in range(len(a) - 1)}
    bigrams_b = {b[i : i + 2] for i in range(len(b) - 1)}
    if not bigrams_a or not bigrams_b:
        return 0.0, 0.0
    intersection = len(bigrams_a & bigrams_b)
    return (
        intersection / min(len(bigrams_a), len(bigrams_b)),
        intersection / len(bigrams_a | bigrams_b),
    )


def is_complete_sentence(span: str) -> bool:
    """Check that a span ends like a complete Korean sentence."""

    stripped = span.strip().rstrip(_CLOSING_QUOTES).rstrip()
    if not stripped:
        return False
    if not stripped.endswith(_SENTENCE_FINAL_PUNCT):
        return False
    core = stripped.rstrip("".join(_SENTENCE_FINAL_PUNCT)).rstrip(_CLOSING_QUOTES).rstrip()
    return bool(core) and core[-1] in _FINAL_SYLLABLES


def _must_be_full_sentence(operation: str, before: str) -> bool:
    # insert_after adds a standalone sentence; replace inherits the shape of
    # what it replaced (phrase-level replaces are exempt from the check).
    if operation == "insert_after":
        return True
    return before.endswith(_SENTENCE_FINAL_PUNCT)


def _normalize(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()
