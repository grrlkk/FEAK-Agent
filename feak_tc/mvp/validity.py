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
}

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
    return reasons


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
