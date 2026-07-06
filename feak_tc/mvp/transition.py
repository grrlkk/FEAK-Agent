"""Transition feature computation for the FEAK-TC MVP."""

from __future__ import annotations

from difflib import SequenceMatcher

from feak_tc.diagnose import Diagnosis
from feak_tc.diagnose.constants import RUBRIC_FEATURE_MAP, RUBRIC_KEYS
from feak_tc.diagnose.stub import tokenize

from .schemas import Candidate, Transition


def compute_transition(before: Diagnosis, after: Diagnosis, cand: Candidate) -> Transition:
    target = cand.target_rubric
    target_gain = after.rubrics[target] - before.rubrics[target]
    non_target_drop = max(
        before.rubrics[key] - after.rubrics[key]
        for key in RUBRIC_KEYS
        if key != target
    )
    non_target_drop = max(0.0, non_target_drop)
    target_gap_reduction = _feature_gap_reduction(before, after, target)
    goal_preservation = source_token_retention(before.text, after.text)
    emb_sim = lexical_similarity(before.text, after.text)
    return Transition(
        action_type=cand.action_type,
        target_rubric=target,
        target_gain=float(target_gain),
        non_target_drop=float(non_target_drop),
        target_gap_reduction=float(target_gap_reduction),
        evidence_match=float(_evidence_match(before, cand)),
        edit_ratio=float(edit_ratio(before.text, after.text)),
        goal_preservation=float(goal_preservation),
        emb_sim=float(emb_sim),
    )


def edit_ratio(before_text: str, after_text: str) -> float:
    before_tokens = tokenize(before_text)
    after_tokens = tokenize(after_text)
    total = max(1, len(before_tokens), len(after_tokens))
    matcher = SequenceMatcher(a=before_tokens, b=after_tokens)
    changed = 0
    for tag, i1, i2, j1, j2 in matcher.get_opcodes():
        if tag == "equal":
            continue
        changed += max(i2 - i1, j2 - j1)
    return min(1.0, changed / total)


def source_token_retention(before_text: str, after_text: str) -> float:
    before_tokens = tokenize(before_text)
    after_tokens = tokenize(after_text)
    if not before_tokens and not after_tokens:
        return 1.0
    if not before_tokens or not after_tokens:
        return 0.0
    matcher = SequenceMatcher(a=before_tokens, b=after_tokens)
    retained = sum(size for _, _, size in matcher.get_matching_blocks())
    return min(1.0, retained / len(before_tokens))


def lexical_similarity(a: str, b: str) -> float:
    a_tokens = set(tokenize(a))
    b_tokens = set(tokenize(b))
    if not a_tokens and not b_tokens:
        return 1.0
    if not a_tokens or not b_tokens:
        return 0.0
    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def _feature_gap_reduction(before: Diagnosis, after: Diagnosis, target_rubric: str) -> float:
    features = RUBRIC_FEATURE_MAP.get(target_rubric, [])
    if not features:
        return 0.0
    deltas = []
    for feature in features:
        before_value = before.features.get(feature, 0.0)
        after_value = after.features.get(feature, 0.0)
        scale = max(1.0, abs(before_value))
        deltas.append((after_value - before_value) / scale)
    return sum(deltas) / len(deltas)


def _evidence_match(before: Diagnosis, cand: Candidate) -> float:
    score = 0.0
    if cand.target_rubric in before.weak_rubrics:
        score += 0.6
    if cand.target_span and cand.target_span in before.text:
        score += 0.3
    if cand.action_type == "STOP":
        score = 1.0 if not before.weak_rubrics else 0.2
    return min(1.0, score)
