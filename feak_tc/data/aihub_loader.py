"""AI-Hub Korean writing JSON normalization utilities."""

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, DefaultDict, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from feak_tc.schemas.essay import EssayRecord


PathLike = Union[str, Path]

_ID_KEYS = (
    "essay_id",
    "essayId",
    "writing_id",
    "writingId",
    "document_id",
    "documentId",
    "doc_id",
    "docId",
    "sample_id",
    "sampleId",
    "id",
)
_PROMPT_KEYS = ("prompt", "question", "assignment", "task_prompt", "topic_prompt", "제시문")
_TOPIC_KEYS = ("topic", "subject", "title", "essay_topic", "주제")
_GRADE_KEYS = ("grade", "student_grade", "school_grade", "학년")
_PURPOSE_KEYS = ("purpose", "writing_purpose", "genre", "task_type", "글의목적")
_TEXT_KEYS = (
    "text",
    "essay_text",
    "essayText",
    "essay",
    "content",
    "writing",
    "answer",
    "body",
    "paragraph",
    "paragraphs",
    "sentences",
    "원문",
)
_FEATURE_KEYS = (
    "features",
    "feature",
    "linguistic_features",
    "morpheme_features",
    "morpheme",
    "metrics",
)
_SCORE_KEYS = (
    "rubric_scores",
    "rubricScores",
    "rubric_score",
    "scores",
    "scoring",
    "evaluation",
    "evaluations",
    "assessment",
    "assessments",
    "ratings",
)
_MEAN_SCORE_KEYS = ("rubric_scores_mean", "rubricScoresMean", "score_means", "mean_scores")
_FEEDBACK_KEYS = (
    "expert_feedback",
    "expertFeedback",
    "feedback",
    "comments",
    "comment",
    "teacher_feedback",
    "annotations",
    "첨삭",
)
_RUBRIC_DEFINITION_KEYS = (
    "rubric_definitions",
    "rubricDefinitions",
    "rubric_criteria",
    "criteria",
    "scoring_criteria",
    "rubric",
)
_RECORD_CONTAINER_KEYS = ("records", "essays", "documents", "data", "items")
_GENERIC_SCORE_KEYS = {"score", "scores", "point", "points", "value", "rawscore", "점수"}
_SUMMARY_SCORE_KEYS = {"total", "sum", "mean", "average", "avg"}
_LABEL_KEYS = {"rubric", "criterion", "criteria", "category", "item", "name", "label", "항목"}


def iter_json_files(input_path: PathLike, missing_ok: bool = False) -> Iterator[Path]:
    """Yield JSON files from a file or directory in deterministic order."""

    path = Path(input_path)
    if not path.exists():
        if missing_ok:
            return
        raise FileNotFoundError(path)
    if path.is_file():
        if path.suffix.lower() == ".json":
            yield path
        return
    yield from sorted(p for p in path.rglob("*.json") if p.is_file())


def load_json_file(path: PathLike) -> Any:
    """Load a JSON file using utf-8-sig to tolerate BOM-prefixed AI-Hub files."""

    path = Path(path)
    try:
        with path.open("r", encoding="utf-8-sig") as f:
            return json.load(f)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid JSON file: {path}") from exc


def iter_aihub_records(
    input_path: PathLike,
    limit: Optional[int] = None,
    missing_ok: bool = False,
) -> Iterator[EssayRecord]:
    """Yield normalized records from one JSON file or a directory of JSON files."""

    emitted = 0
    for json_path in iter_json_files(input_path, missing_ok=missing_ok):
        payload = load_json_file(json_path)
        for record_index, raw_record in _split_payload_records(payload):
            if not isinstance(raw_record, dict):
                continue
            yield normalize_aihub_record(raw_record, json_path, record_index=record_index)
            emitted += 1
            if limit is not None and emitted >= limit:
                return


def load_aihub_records(
    input_path: PathLike,
    limit: Optional[int] = None,
    missing_ok: bool = False,
) -> List[EssayRecord]:
    """Return normalized records as a list."""

    return list(iter_aihub_records(input_path, limit=limit, missing_ok=missing_ok))


def normalize_aihub_record(
    raw: Dict[str, Any],
    raw_path: PathLike,
    record_index: Optional[int] = None,
) -> EssayRecord:
    """Normalize a raw AI-Hub JSON object into the canonical EssayRecord schema."""

    raw_path_value = str(raw_path)
    if record_index is not None:
        raw_path_value = f"{raw_path_value}#{record_index}"

    essay_id = _coerce_scalar(_find_first_by_keys(raw, _ID_KEYS))
    if not essay_id:
        stem = Path(raw_path).stem
        essay_id = f"{stem}_{record_index}" if record_index is not None else stem

    score_source = _coerce_mapping(_find_first_by_keys(raw, _SCORE_KEYS))
    provided_means = _coerce_numeric_mapping(_find_first_by_keys(raw, _MEAN_SCORE_KEYS))

    return EssayRecord(
        essay_id=str(essay_id),
        prompt=_coerce_optional_text(_find_first_by_keys(raw, _PROMPT_KEYS)),
        topic=_coerce_optional_text(_find_first_by_keys(raw, _TOPIC_KEYS)),
        grade=_coerce_grade(_find_first_by_keys(raw, _GRADE_KEYS)),
        purpose=_coerce_optional_text(_find_first_by_keys(raw, _PURPOSE_KEYS)),
        text=_find_text(raw),
        features=_coerce_mapping(_find_first_by_keys(raw, _FEATURE_KEYS)),
        rubric_scores_raw=score_source,
        rubric_scores_mean=provided_means or compute_rubric_score_means(score_source),
        expert_feedback=_coerce_feedback(_find_first_by_keys(raw, _FEEDBACK_KEYS)),
        rubric_definitions=_coerce_mapping_or_value(_find_first_by_keys(raw, _RUBRIC_DEFINITION_KEYS)),
        raw_path=raw_path_value,
    )


def compute_rubric_score_means(score_source: Dict[str, Any]) -> Dict[str, float]:
    """Compute rubric-level means from nested scorer/rater structures."""

    buckets: DefaultDict[str, List[float]] = defaultdict(list)
    _collect_score_values(score_source, buckets)
    return {key: sum(values) / len(values) for key, values in buckets.items() if values}


def _split_payload_records(payload: Any) -> Iterator[Tuple[Optional[int], Any]]:
    if isinstance(payload, list):
        for idx, item in enumerate(payload):
            yield idx, item
        return

    if isinstance(payload, dict):
        for key, value in payload.items():
            if _normalize_key(key) in {_normalize_key(k) for k in _RECORD_CONTAINER_KEYS}:
                if isinstance(value, list) and all(isinstance(item, dict) for item in value):
                    if not _has_direct_key(payload, _TEXT_KEYS):
                        for idx, item in enumerate(value):
                            yield idx, item
                        return
        yield None, payload


def _find_text(raw: Dict[str, Any]) -> str:
    value = _find_first_by_keys(raw, _TEXT_KEYS)
    return _coerce_text(value)


def _find_first_by_keys(obj: Any, keys: Sequence[str]) -> Any:
    normalized = {_normalize_key(key) for key in keys}
    queue: List[Any] = [obj]
    while queue:
        current = queue.pop(0)
        if isinstance(current, dict):
            for key, value in current.items():
                if _normalize_key(key) in normalized and not _is_empty(value):
                    return value
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None


def _has_direct_key(obj: Dict[str, Any], keys: Sequence[str]) -> bool:
    normalized = {_normalize_key(key) for key in keys}
    return any(_normalize_key(key) in normalized for key in obj.keys())


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())


def _is_empty(value: Any) -> bool:
    return value is None or value == "" or value == [] or value == {}


def _coerce_scalar(value: Any) -> Optional[Union[str, int, float]]:
    if isinstance(value, (str, int, float)) and not isinstance(value, bool):
        return value
    return None


def _coerce_grade(value: Any) -> Optional[Union[str, int]]:
    scalar = _coerce_scalar(value)
    if isinstance(scalar, float) and scalar.is_integer():
        return int(scalar)
    if isinstance(scalar, (str, int)):
        return scalar
    return None


def _coerce_optional_text(value: Any) -> Optional[str]:
    text = _coerce_text(value)
    return text or None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return str(value)
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, dict):
        for key in _TEXT_KEYS + ("value", "sentence"):
            for raw_key, raw_value in value.items():
                if _normalize_key(raw_key) == _normalize_key(key):
                    text = _coerce_text(raw_value)
                    if text:
                        return text
    return ""


def _coerce_mapping(value: Any) -> Dict[str, Any]:
    if isinstance(value, dict):
        return value
    if isinstance(value, list):
        return {"items": value}
    return {}


def _coerce_mapping_or_value(value: Any) -> Any:
    if value is None:
        return {}
    return value


def _coerce_feedback(value: Any) -> Any:
    if value is None:
        return []
    return value


def _coerce_numeric_mapping(value: Any) -> Dict[str, float]:
    if not isinstance(value, dict):
        return {}
    result: Dict[str, float] = {}
    for key, raw_value in value.items():
        numeric = _as_float(raw_value)
        if numeric is not None:
            result[str(key)] = numeric
    return result


def _as_float(value: Any) -> Optional[float]:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value.strip())
        except ValueError:
            return None
    return None


def _collect_score_values(
    obj: Any,
    buckets: DefaultDict[str, List[float]],
    parent_key: Optional[str] = None,
) -> None:
    if isinstance(obj, list):
        for item in obj:
            _collect_score_values(item, buckets, parent_key=parent_key)
        return

    if not isinstance(obj, dict):
        return

    label = _current_label(obj)
    score = _current_score(obj)
    if label and score is not None:
        buckets[label].append(score)

    for key, value in obj.items():
        key_name = str(key)
        norm_key = _normalize_key(key)

        numeric = _as_float(value)
        if numeric is not None:
            if norm_key not in _GENERIC_SCORE_KEYS and norm_key not in _SUMMARY_SCORE_KEYS:
                buckets[key_name].append(numeric)
            elif (
                parent_key
                and not label
                and _normalize_key(parent_key) not in _GENERIC_SCORE_KEYS
            ):
                buckets[str(parent_key)].append(numeric)
            continue

        if isinstance(value, dict):
            direct_score = _current_score(value)
            if direct_score is not None and norm_key not in _GENERIC_SCORE_KEYS:
                buckets[key_name].append(direct_score)
                if _is_leaf_score_dict(value):
                    continue
            _collect_score_values(value, buckets, parent_key=key_name)
        elif isinstance(value, list):
            _collect_score_values(value, buckets, parent_key=key_name)


def _current_label(obj: Dict[str, Any]) -> Optional[str]:
    for key, value in obj.items():
        if _normalize_key(key) in _LABEL_KEYS and isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _current_score(obj: Dict[str, Any]) -> Optional[float]:
    for key, value in obj.items():
        if _normalize_key(key) in _GENERIC_SCORE_KEYS:
            numeric = _as_float(value)
            if numeric is not None:
                return numeric
    return None


def _is_leaf_score_dict(obj: Dict[str, Any]) -> bool:
    for key, value in obj.items():
        norm_key = _normalize_key(key)
        if norm_key in _LABEL_KEYS or norm_key in _GENERIC_SCORE_KEYS:
            continue
        if isinstance(value, (dict, list)):
            return False
        if _as_float(value) is not None and norm_key not in _SUMMARY_SCORE_KEYS:
            return False
    return True
