"""Kanana scorer rubric and FEAK feature constants."""

from __future__ import annotations

from typing import Iterable, Mapping, Sequence


RUBRIC_KEYS = [
    "task_1",
    "content_1",
    "content_2",
    "content_3",
    "organization_1",
    "organization_2",
    "expression_1",
    "expression_2",
]

RUBRIC_NAMES_KO = {
    "task_1": "과제충실성",
    "content_1": "설명명료성",
    "content_2": "설명구체성",
    "content_3": "설명적절성",
    "organization_1": "문장연결성",
    "organization_2": "글통일성",
    "expression_1": "어휘적절성",
    "expression_2": "어법적절성",
}

FEAK_FEATURE_NAMES = [
    "C_Cnt",
    "E_Cnt",
    "F_Cnt",
    "J_Cnt",
    "X_Cnt",
    "char_Cnt",
    "word_Cnt",
    "E_NDW",
    "morph_NDW",
    "morph_LenAvg",
    "morph_LenStd",
    "word_LenStd",
    "grade_2_ratio",
    "grade_3_ratio",
    "grade_4_ratio",
    "grade_m1_ratio",
    "N_MSTTR",
    "V_HDD",
    "lemma_MATTR",
    "2-gram_NDW",
    "NN_repRatio",
    "word_sentLenAvg",
    "2-gram_RTTR",
    "adjacent_sentence_overlap_function_lemmas",
    "char_paraLenAvg",
    "avgSentSimilarity",
    "topicConsistency",
    "text_dalechall",
    "text_oridx",
]

ACTION_TYPES = [
    "ADD_DETAIL",
    "DELETE_OR_FOCUS",
    "COMPRESS",
    "RESTRUCTURE",
    "STYLE_REFINE",
    "STOP",
]

MIN_SCORE = 1
MAX_SCORE = 9

RUBRIC_FEATURE_MAP = {
    "task_1": ["topicConsistency", "word_Cnt", "char_Cnt"],
    "content_1": ["topicConsistency", "avgSentSimilarity", "C_Cnt"],
    "content_2": ["word_Cnt", "char_Cnt", "2-gram_NDW"],
    "content_3": ["topicConsistency", "grade_4_ratio", "C_Cnt"],
    "organization_1": ["adjacent_sentence_overlap_function_lemmas", "avgSentSimilarity"],
    "organization_2": ["topicConsistency", "avgSentSimilarity", "char_paraLenAvg"],
    "expression_1": ["lemma_MATTR", "morph_NDW", "N_MSTTR"],
    "expression_2": ["E_NDW", "morph_LenAvg", "morph_LenStd"],
}


def scores_to_rubric_dict(scores: Sequence[float]) -> dict[str, float]:
    """Map an 8-score sequence to the fixed Kanana rubric-key order."""

    if len(scores) != len(RUBRIC_KEYS):
        raise ValueError(f"Expected {len(RUBRIC_KEYS)} rubric scores, got {len(scores)}")
    return {key: float(value) for key, value in zip(RUBRIC_KEYS, scores)}


def require_rubric_scores(rubrics: Mapping[str, float]) -> dict[str, float]:
    """Return a normalized rubric dict or raise if any Kanana key is missing."""

    missing = [key for key in RUBRIC_KEYS if key not in rubrics]
    if missing:
        raise ValueError(f"Missing rubric scores: {missing}")
    return {key: float(rubrics[key]) for key in RUBRIC_KEYS}


def require_feak_features(features: Mapping[str, float]) -> dict[str, float]:
    """Return a normalized FEAK feature dict, filling absent MVP stub values with 0."""

    return {key: float(features.get(key, 0.0)) for key in FEAK_FEATURE_NAMES}


def validate_action_type(action_type: str, allowed: Iterable[str] = ACTION_TYPES) -> str:
    if action_type not in set(allowed):
        raise ValueError(f"Unknown action_type: {action_type}")
    return action_type
