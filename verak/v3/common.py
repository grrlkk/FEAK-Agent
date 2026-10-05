"""Small local-file utilities; no environment files or model side effects."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import tempfile

import yaml

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = Path(__file__).with_name("config.yaml")


def sha_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pair_key(question: str, text: str) -> str:
    return sha_text(json.dumps([question, text], ensure_ascii=False, separators=(",", ":")))


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        temp = Path(handle.name)
    temp.replace(path)


def load_config(path=DEFAULT_CONFIG):
    config = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    root = Path(config["paths"]["repo"])
    for name, value in config["paths"].items():
        candidate = Path(value)
        config["paths"][name] = candidate if candidate.is_absolute() else root / candidate
    return config


def extract_question_essay(row: dict) -> tuple[str, str]:
    """Same exact text boundary as v2; gold and keywords are not returned."""
    if isinstance(row.get("question"), str) and isinstance(row.get("draft"), str):
        question, essay = row["question"], row["draft"]
    else:
        value = row.get("user")
        if not isinstance(value, str) or not value.startswith("질문:") or "\n에세이:" not in value:
            raise ValueError("Expected question/draft or 질문:/에세이: input")
        question, essay = value[len("질문:"):].split("\n에세이:", 1)
        question, essay = question.removeprefix(" "), essay.removeprefix(" ")
        if "\n핵심 키워드:" in essay:
            essay, _ = essay.rsplit("\n핵심 키워드:", 1)
    if not question.strip() or not essay.strip():
        raise ValueError("Empty question or essay")
    return question, essay


@dataclass(frozen=True)
class Example:
    source_line: int
    question: str
    text: str
    question_hash: str
    essay_hash: str
    genre: str
    seen_by_scorer: bool

    @property
    def id(self):
        return f"valid:{self.source_line}"

    def metadata(self):
        return {"id": self.id, "source_line": self.source_line,
                "question_hash": self.question_hash, "essay_hash": self.essay_hash,
                "genre": self.genre, "seen_by_scorer": self.seen_by_scorer}
