"""Batch execution helpers for collecting MVP transition logs."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Iterator, Mapping, Optional, Union

from feak_tc.diagnose import Diagnoser

from .loop import serializable_one_step


PathLike = Union[str, Path]

_TEXT_KEYS = (
    "text",
    "essay_text",
    "essayText",
    "essay",
    "content",
    "body",
    "answer",
    "writing",
    "paragraph",
    "paragraphs",
)
_ID_KEYS = ("record_id", "essay_id", "essayId", "id", "sample_id", "sampleId", "document_id", "docId")
_CONTAINER_KEYS = ("records", "essays", "documents", "data", "items")


@dataclass
class TextRecord:
    record_id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def iter_text_records(input_path: PathLike, limit: Optional[int] = None) -> Iterator[TextRecord]:
    """Yield text records from txt, jsonl, json, or a directory of txt files."""

    path = Path(input_path)
    if not path.exists():
        raise FileNotFoundError(path)

    emitted = 0
    for record in _iter_text_records(path):
        if not record.text.strip():
            continue
        yield record
        emitted += 1
        if limit is not None and emitted >= limit:
            return


def run_batch(
    *,
    input_path: PathLike,
    output_path: PathLike,
    diagnoser: Diagnoser,
    cfg: Optional[Mapping[str, Any]] = None,
    limit: Optional[int] = None,
    append: bool = False,
    fail_fast: bool = False,
) -> dict[str, Any]:
    """Run one-step MVP over many records and write JSONL logs."""

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if append else "w"
    summary = {"total": 0, "ok": 0, "error": 0, "output_path": str(output)}

    with output.open(mode, encoding="utf-8") as f:
        for record in iter_text_records(input_path, limit=limit):
            summary["total"] += 1
            try:
                result = serializable_one_step(text=record.text, diagnoser=diagnoser, cfg=cfg)
                row = {
                    "record_id": record.record_id,
                    "status": "ok",
                    "input": record.to_dict(),
                    "output": result,
                }
                summary["ok"] += 1
            except Exception as exc:
                row = {
                    "record_id": record.record_id,
                    "status": "error",
                    "input": record.to_dict(),
                    "error": {
                        "type": exc.__class__.__name__,
                        "message": str(exc),
                    },
                }
                summary["error"] += 1
                if fail_fast:
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    f.flush()
                    raise

            f.write(json.dumps(row, ensure_ascii=False) + "\n")
            f.flush()

    return summary


def _iter_text_records(path: Path) -> Iterator[TextRecord]:
    if path.is_dir():
        supported = {".txt", ".jsonl", ".json"}
        for child in sorted(p for p in path.rglob("*") if p.is_file() and p.suffix.lower() in supported):
            yield from _iter_text_records(child)
        return

    suffix = path.suffix.lower()
    if suffix == ".txt":
        yield TextRecord(
            record_id=path.stem,
            text=path.read_text(encoding="utf-8"),
            metadata={"source_path": str(path)},
        )
        return

    if suffix == ".jsonl":
        yield from _iter_jsonl_records(path)
        return

    if suffix == ".json":
        yield from _iter_json_records(path)
        return

    raise ValueError(f"Unsupported batch input format: {path}")


def _iter_jsonl_records(path: Path) -> Iterator[TextRecord]:
    with path.open("r", encoding="utf-8-sig") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            raw = json.loads(line)
            yield _coerce_record(raw, path, idx)


def _iter_json_records(path: Path) -> Iterator[TextRecord]:
    with path.open("r", encoding="utf-8-sig") as f:
        payload = json.load(f)
    for idx, raw in enumerate(_split_json_payload(payload)):
        yield _coerce_record(raw, path, idx)


def _split_json_payload(payload: Any) -> list[Any]:
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        for key in _CONTAINER_KEYS:
            value = payload.get(key)
            if isinstance(value, list):
                return value
        return [payload]
    return []


def _coerce_record(raw: Any, path: Path, index: int) -> TextRecord:
    if isinstance(raw, str):
        return TextRecord(
            record_id=f"{path.stem}_{index}",
            text=raw,
            metadata={"source_path": str(path), "record_index": index},
        )
    if not isinstance(raw, Mapping):
        return TextRecord(
            record_id=f"{path.stem}_{index}",
            text="",
            metadata={"source_path": str(path), "record_index": index, "raw_type": type(raw).__name__},
        )

    record_id = _find_scalar(raw, _ID_KEYS) or f"{path.stem}_{index}"
    text = _find_text(raw)
    metadata = {
        "source_path": str(path),
        "record_index": index,
    }
    for key in ("prompt", "question", "topic", "grade", "purpose"):
        value = _find_scalar(raw, (key,))
        if value is not None:
            metadata[key] = value
    return TextRecord(record_id=str(record_id), text=text, metadata=metadata)


def _find_scalar(obj: Any, keys: tuple[str, ...]) -> Optional[Any]:
    normalized = {_normalize_key(key) for key in keys}
    queue = [obj]
    while queue:
        current = queue.pop(0)
        if isinstance(current, Mapping):
            for key, value in current.items():
                if _normalize_key(key) in normalized and isinstance(value, (str, int, float)):
                    return value
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None


def _find_text(obj: Any) -> str:
    value = _find_first(obj, _TEXT_KEYS)
    return _coerce_text(value)


def _find_first(obj: Any, keys: tuple[str, ...]) -> Any:
    normalized = {_normalize_key(key) for key in keys}
    queue = [obj]
    while queue:
        current = queue.pop(0)
        if isinstance(current, Mapping):
            for key, value in current.items():
                if _normalize_key(key) in normalized and value not in (None, "", [], {}):
                    return value
            queue.extend(current.values())
        elif isinstance(current, list):
            queue.extend(current)
    return None


def _coerce_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        parts = [_coerce_text(item) for item in value]
        return "\n".join(part for part in parts if part)
    if isinstance(value, Mapping):
        nested = _find_first(value, _TEXT_KEYS + ("value", "sentence"))
        if nested is not value:
            return _coerce_text(nested)
    return ""


def _normalize_key(key: Any) -> str:
    return "".join(ch for ch in str(key).lower() if ch.isalnum())
