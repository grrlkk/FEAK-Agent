"""Transition feature computation for the FEAK-TC MVP."""

from __future__ import annotations

import os
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping, Optional

from feak_tc.diagnose import Diagnosis
from feak_tc.diagnose.constants import RUBRIC_FEATURE_MAP, RUBRIC_KEYS
from feak_tc.diagnose.stub import tokenize

from .schemas import Candidate, Transition


ELITE_STATS_PATH = Path(__file__).resolve().parents[2] / "configs" / "elite_features.yaml"
EMBEDDING_MODEL_CANDIDATES = (
    "jhgan/ko-sbert-sts",
    "snunlp/KR-SBERT-V40K-klueNLI-augSTS",
)


def compute_transition(
    before: Diagnosis,
    after: Diagnosis,
    cand: Candidate,
    elite_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> Transition:
    target = cand.target_rubric
    target_gain = after.rubrics[target] - before.rubrics[target]
    non_target_drop = max(
        before.rubrics[key] - after.rubrics[key]
        for key in RUBRIC_KEYS
        if key != target
    )
    non_target_drop = max(0.0, non_target_drop)
    target_gap_reduction = _feature_gap_reduction(before, after, target, elite_stats)
    semantic_similarity, similarity_info = semantic_text_similarity(before.text, after.text)
    cand.metadata["similarity"] = similarity_info
    return Transition(
        action_type=cand.action_type,
        target_rubric=target,
        target_gain=float(target_gain),
        non_target_drop=float(non_target_drop),
        target_gap_reduction=float(target_gap_reduction),
        evidence_match=float(_evidence_match(before, cand)),
        edit_ratio=float(edit_ratio(before.text, after.text)),
        goal_preservation=float(semantic_similarity),
        emb_sim=float(semantic_similarity),
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


def semantic_text_similarity(a: str, b: str) -> tuple[float, dict[str, Any]]:
    """Return Korean SBERT cosine similarity, falling back to legacy lexical signals.

    The transition schema stays fixed: both goal_preservation and emb_sim use
    this scalar. The chosen method is logged on the candidate metadata by
    compute_transition so batch logs can be audited.
    """

    if not a.strip() and not b.strip():
        return 1.0, {"method": "empty_text"}
    if not a.strip() or not b.strip():
        return 0.0, {"method": "empty_text"}

    model, model_info = _embedding_model()
    if model is None:
        return _legacy_similarity_fallback(a, b, model_info)

    try:
        embeddings = model.encode([a, b], normalize_embeddings=True)
        sim = _dot(embeddings[0], embeddings[1])
        return _clip_cosine(sim), {
            "method": "korean_sbert",
            "model": model_info["model"],
        }
    except Exception as exc:  # pragma: no cover - depends on optional runtime model
        info = dict(model_info)
        info["error"] = f"{exc.__class__.__name__}: {exc}"
        return _legacy_similarity_fallback(a, b, info)


def _legacy_similarity_fallback(a: str, b: str, reason: Mapping[str, Any]) -> tuple[float, dict[str, Any]]:
    return source_token_retention(a, b), {
        "method": "token_fallback",
        "goal_preservation": "source_token_retention",
        "emb_sim": "source_token_retention",
        "legacy_lexical_similarity": lexical_similarity(a, b),
        "reason": dict(reason),
    }


@lru_cache(maxsize=1)
def _embedding_model() -> tuple[Any, dict[str, Any]]:
    try:
        from sentence_transformers import SentenceTransformer
    except ModuleNotFoundError as exc:
        return None, {"error": f"{exc.__class__.__name__}: {exc}"}

    errors = []
    model_names = _embedding_model_candidates()
    for model_name in model_names:
        try:
            try:
                return SentenceTransformer(model_name, local_files_only=True), {"model": model_name}
            except TypeError:
                return SentenceTransformer(model_name), {"model": model_name}
        except Exception as exc:  # pragma: no cover - optional model/cache/network dependent
            errors.append(f"{model_name}: {exc.__class__.__name__}: {exc}")
    return None, {"error": "; ".join(errors), "candidates": list(model_names)}


def _embedding_model_candidates() -> tuple[str, ...]:
    configured = os.getenv("FEAK_EMBEDDING_MODEL")
    if configured:
        return (configured,)
    return EMBEDDING_MODEL_CANDIDATES


def _dot(a: Any, b: Any) -> float:
    return float(sum(float(x) * float(y) for x, y in zip(a, b)))


def _clip_cosine(value: float) -> float:
    return max(-1.0, min(1.0, value))


def _feature_gap_reduction(
    before: Diagnosis,
    after: Diagnosis,
    target_rubric: str,
    elite_stats: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    """How much closer the target features moved toward the elite band.

    Per feature, the gap is the normalized distance outside the elite band
    [low, high] measured on high-scoring essays; the reduction is
    gap(before) - gap(after), so movement past the band in either direction is
    penalized. Returns 0.0 when no elite stats are available (the stats file is
    produced by scripts/build_elite_band.py).
    """

    stats = elite_stats if elite_stats is not None else _load_elite_stats()
    features = RUBRIC_FEATURE_MAP.get(target_rubric, [])
    if not features or not stats:
        return 0.0
    reductions = []
    for feature in features:
        band = stats.get(feature)
        if not band:
            continue
        low = float(band["low"])
        high = float(band["high"])
        scale = max(high - low, 1e-6)
        gap_before = _band_gap(before.features.get(feature, 0.0), low, high, scale)
        gap_after = _band_gap(after.features.get(feature, 0.0), low, high, scale)
        reductions.append(gap_before - gap_after)
    if not reductions:
        return 0.0
    value = sum(reductions) / len(reductions)
    return max(-1.0, min(1.0, value))


def _band_gap(value: float, low: float, high: float, scale: float) -> float:
    if value < low:
        return (low - value) / scale
    if value > high:
        return (value - high) / scale
    return 0.0


@lru_cache(maxsize=1)
def _load_elite_stats() -> dict[str, dict[str, float]]:
    if not ELITE_STATS_PATH.exists():
        return {}
    import yaml

    with ELITE_STATS_PATH.open(encoding="utf-8") as f:
        payload: Any = yaml.safe_load(f) or {}
    features = payload.get("features", payload)
    stats: dict[str, dict[str, float]] = {}
    if isinstance(features, Mapping):
        for name, band in features.items():
            if isinstance(band, Mapping) and "low" in band and "high" in band:
                stats[str(name)] = {"low": float(band["low"]), "high": float(band["high"])}
    return stats


def _evidence_match(before: Diagnosis, cand: Candidate) -> float:
    score = 0.0
    if cand.target_rubric in before.weak_rubrics:
        score += 0.6
    if cand.target_span and cand.target_span in before.text:
        score += 0.3
    if cand.action_type == "STOP":
        score = 1.0 if not before.weak_rubrics else 0.2
    return min(1.0, score)
