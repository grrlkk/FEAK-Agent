"""Deterministic retrieval of natural off-topic sentences.

The assignment is built for the complete source bank, rather than one process,
so sharded generation cannot accidentally reuse the same distractor sentence.
Only sentences copied from a different question and source essay are eligible.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Mapping

from feak_tc.mvp.validity import is_complete_sentence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
_DECIMAL_DOT = "\ue000"
_WORD_RE = re.compile(r"[0-9A-Za-z가-힣]+")
_NUMBER_RE = re.compile(r"\d+(?:[.,]\d+)*%?")
_DEPENDENT_PREFIXES = (
    "그리고",
    "그래서",
    "그러나",
    "그런데",
    "그러므로",
    "따라서",
    "하지만",
    "반면",
    "또한",
    "게다가",
    "예를 들어",
    "예컨대",
    "가령",
    "먼저",
    "다음으로",
    "마지막으로",
    "결국",
    "이처럼",
    "이러한",
    "이것",
    "이를",
    "이는",
    "이때",
    "여기서",
    "그 결과",
    "이로 인해",
    "또,",
    "또 ",
    "물론",
    "반대로",
    "그러니",
    "그러면",
    "그렇다고",
    "이렇게",
    "이런",
    "예전에는",
    "첫째",
    "둘째",
    "셋째",
    "무엇보다",
    "그뒤",
    "그때",
    "그렇다면",
    "그의",
    "일례로",
    "앞으로",
    "뉴스에서",
    "이렇듯",
    "우선",
    "예를들어",
    "나의",
    "나는",
    "저는",
    "제가",
    "저도",
    "우리가",
    "우리의",
    "당신",
    "놀라운 것은",
    "사실",
    "예를 들면",
    "그러자",
    "특히",
    "실제로",
    "실제가",
    "일단",
    "즉",
    "첫 번째로",
    "두 번째로",
    "세 번째로",
    "나뿐만",
    "만약",
    "그럼에도",
    "그동안",
    "이와 같은",
    "이에 대한",
    "당시",
    "그날",
    "자신의",
    "결론적으로",
    "처음에는",
    "그렇기 때문에",
    "그러면서",
    "왜냐하면",
    "이와 같이",
    "이로 인한",
    "어제",
    "오늘",
    "주말",
    "요즘",
    "최근",
    "지난",
    "가끔",
    "점심",
    "아침",
    "내일",
    "두번째로",
)
_DEMONSTRATIVE_PREFIX_RE = re.compile(
    r"^(?:이|그|저)\s|^(?:이들|그들|저들|이것|그것|저것|이는|그는|그녀|그의|그때|그뒤|이러한|그러한|이런|그런)"
)
_FORMAL_POLITE_RE = re.compile(r"(?:습니다|습니까|ㅂ니다|ㅂ니까)[.!?。？！]$")
_QUOTE_CHARS = "\"'‘’“”「」『』"


@dataclass(frozen=True)
class DistractorSentence:
    text: str
    record_id: str
    question: str


@dataclass(frozen=True)
class SourceRecord:
    record_id: str
    question: str
    text: str


def generate_retrieval_payload(
    *,
    record_id: str,
    question: str,
    text: str,
    source_text: str,
    spec: Mapping[str, Any],
) -> dict[str, Any]:
    """Return exact anchors plus globally assigned distractor provenance."""

    count = int(spec.get("edits_per_step", 1))
    bank_path = _resolve_bank_path(str(spec["distractor_bank_path"]))
    minimum = int(spec.get("distractor_min_chars", 15))
    maximum = int(spec.get("distractor_max_chars", 80))
    max_jaccard = float(spec.get("distractor_max_lexical_jaccard", 0.12))
    pool_size = int(spec.get("distractor_candidate_pool_size", 1))
    match_style = bool(spec.get("distractor_match_ending_style", True))
    assignments = _build_assignments(
        str(bank_path),
        count,
        pool_size,
        minimum,
        maximum,
        max_jaccard,
        match_style,
    )
    assigned = assignments.get(record_id)
    if assigned is None:
        raise ValueError(f"record_id is absent from distractor bank: {record_id}")
    if len(assigned) != count * pool_size:
        raise ValueError(
            f"distractor assignment has {len(assigned)} sentences, "
            f"expected {count * pool_size}"
        )

    selected = _select_semantic_candidates(
        assigned,
        count=count,
        pool_size=pool_size,
        question=question,
        source_text=source_text,
        enabled=bool(spec.get("distractor_semantic_filter_enabled", False)),
        max_question_similarity=float(
            spec.get("distractor_max_question_similarity", 0.50)
        ),
        max_source_similarity=float(
            spec.get("distractor_max_source_similarity", 0.50)
        ),
    )

    anchors = _select_anchors(text, source_text, count, record_id)
    edits = []
    for anchor, selection in zip(anchors, selected):
        distractor = selection["candidate"]
        if distractor.record_id == record_id:
            raise ValueError("distractor must originate in a different essay")
        if _normalize(distractor.question) == _normalize(question):
            raise ValueError("distractor must originate under a different question")
        edits.append(
            {
                "anchor_span": anchor,
                "insertion": distractor.text,
                "distractor_record_id": distractor.record_id,
                "distractor_question": distractor.question,
                "distractor_question_similarity": selection.get("question_similarity"),
                "distractor_source_similarity": selection.get("source_similarity"),
                "distractor_semantic_model": selection.get("model"),
            }
        )
    return {"edits": edits}


def split_sentence_units(text: str) -> list[str]:
    """Split sentence units without breaking decimal expressions such as 10.1."""

    protected = re.sub(r"(?<=\d)\.(?=\d)", _DECIMAL_DOT, text.strip())
    units = []
    start = 0
    for match in re.finditer(r"[.!?。？！]+(?:[\"'”’」』)】\]]*)", protected):
        unit = protected[start : match.end()].strip()
        if unit:
            units.append(unit.replace(_DECIMAL_DOT, "."))
        start = match.end()
    tail = protected[start:].strip()
    if tail:
        units.append(tail.replace(_DECIMAL_DOT, "."))
    return units


def is_metadata_text(text: str) -> bool:
    return "핵심 키워드:" in _normalize(text)


def is_standalone_distractor_sentence(
    text: str,
    *,
    minimum: int = 15,
    maximum: int = 80,
) -> bool:
    """Return whether a retrieved sentence can stand without donor context."""

    return _eligible_sentence(_normalize(text), minimum, maximum)


def _resolve_bank_path(raw: str) -> Path:
    path = Path(raw).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    if not path.is_file():
        raise ValueError(f"distractor bank does not exist: {path}")
    return path.resolve()


@lru_cache(maxsize=8)
def _build_assignments(
    bank_path: str,
    count: int,
    pool_size: int,
    minimum: int,
    maximum: int,
    max_jaccard: float,
    match_style: bool,
) -> dict[str, tuple[DistractorSentence, ...]]:
    records = _load_records(bank_path)
    candidates = _candidate_bank(records, minimum, maximum)
    if pool_size < 1:
        raise ValueError("distractor candidate pool size must be positive")
    required = len(records) * count * pool_size
    if len(candidates) < required:
        raise ValueError(
            "distractor bank has fewer unique eligible sentences than required "
            f"({len(candidates)} < {required})"
        )

    used_texts: set[str] = set()
    assignments: dict[str, tuple[DistractorSentence, ...]] = {}
    for record in records:
        selected: list[DistractorSentence] = []
        selected_donors: set[str] = set()
        start = _stable_index(record.record_id, len(candidates))
        for slot in range(count):
            slot_start = (
                start + slot * max(1, len(candidates) // count)
            ) % len(candidates)
            for pool_index in range(pool_size):
                candidate_start = (
                    slot_start
                    + pool_index * max(1, len(candidates) // (count * pool_size))
                ) % len(candidates)
                for offset in range(len(candidates)):
                    candidate = candidates[(candidate_start + offset) % len(candidates)]
                    if candidate.text in used_texts:
                        continue
                    if candidate.record_id in selected_donors:
                        continue
                    if not _eligible_for_record(
                        candidate,
                        record,
                        max_jaccard,
                        match_style,
                    ):
                        continue
                    selected.append(candidate)
                    selected_donors.add(candidate.record_id)
                    used_texts.add(candidate.text)
                    break
                else:
                    raise ValueError(
                        "no unused distractor remains for "
                        f"{record.record_id} slot {slot} candidate {pool_index}"
                    )
        assignments[record.record_id] = tuple(selected)
    return assignments


@lru_cache(maxsize=8)
def _load_records(bank_path: str) -> tuple[SourceRecord, ...]:
    records = []
    seen_ids: set[str] = set()
    with Path(bank_path).open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            record_id = str(row.get("record_id", "")).strip()
            question = str(row.get("question", "")).strip()
            text = str(row.get("text", "")).strip()
            if not record_id or not question or not text:
                raise ValueError(f"invalid distractor bank row {line_number}")
            if record_id in seen_ids:
                raise ValueError(f"duplicate distractor bank record_id: {record_id}")
            seen_ids.add(record_id)
            records.append(SourceRecord(record_id, question, text))
    if not records:
        raise ValueError("distractor bank is empty")
    return tuple(records)


def _candidate_bank(
    records: tuple[SourceRecord, ...],
    minimum: int,
    maximum: int,
) -> tuple[DistractorSentence, ...]:
    candidates = []
    seen_texts: set[str] = set()
    for record in records:
        body = " ".join(
            line.strip()
            for line in record.text.splitlines()
            if line.strip() and not is_metadata_text(line)
        )
        for raw_sentence in split_sentence_units(body):
            sentence = _normalize(raw_sentence)
            if sentence in seen_texts or not _eligible_sentence(sentence, minimum, maximum):
                continue
            seen_texts.add(sentence)
            candidates.append(DistractorSentence(sentence, record.record_id, record.question))
    return tuple(candidates)


def _eligible_sentence(sentence: str, minimum: int, maximum: int) -> bool:
    if not minimum <= len(sentence) <= maximum:
        return False
    if not is_complete_sentence(sentence) or len(split_sentence_units(sentence)) != 1:
        return False
    if _NUMBER_RE.search(sentence) or is_metadata_text(sentence):
        return False
    if any(
        marker in sentence
        for marker in (
            "#@",
            "<문단>",
            "http://",
            "https://",
            ":",
            "：",
            "<",
            ">",
        )
    ):
        return False
    if any(character in sentence for character in _QUOTE_CHARS):
        return False
    stripped = sentence.lstrip("\"'‘“")
    if stripped.startswith("(") or stripped.startswith(_DEPENDENT_PREFIXES):
        return False
    first_clause = stripped.split(",", 1)[0]
    if first_clause.endswith(("다면", "라면", "으면", "면")):
        return False
    if any(marker in stripped for marker in (" 또한 ", " 이에 ")):
        return False
    if ", 우선 " in stripped:
        return False
    if "원래" in stripped and "지금은" in stripped:
        return False
    if stripped.rstrip(".!?。？！").endswith(("뜻이 된다", "의미가 된다")):
        return False
    first_words = _WORD_RE.findall(stripped)[:5]
    if not any(
        len(word) > 1 and word.endswith(("은", "는", "이", "가", "란"))
        for word in first_words
    ):
        return False
    return not _DEMONSTRATIVE_PREFIX_RE.match(stripped)


def _eligible_for_record(
    candidate: DistractorSentence,
    record: SourceRecord,
    max_jaccard: float,
    match_style: bool,
) -> bool:
    if candidate.record_id == record.record_id:
        return False
    if _normalize(candidate.question) == _normalize(record.question):
        return False
    if candidate.text in _normalize(record.text):
        return False
    record_style = _ending_style(record.text)
    if (
        match_style
        and record_style != "unclassified"
        and _ending_style(candidate.text) != record_style
    ):
        return False
    target_tokens = set(_WORD_RE.findall(f"{record.question} {record.text}".lower()))
    candidate_tokens = set(_WORD_RE.findall(candidate.text.lower()))
    if not target_tokens or not candidate_tokens:
        return False
    jaccard = len(target_tokens & candidate_tokens) / len(target_tokens | candidate_tokens)
    return jaccard <= max_jaccard


def _select_anchors(
    current_text: str,
    source_text: str,
    count: int,
    record_id: str,
) -> list[str]:
    candidates = [
        sentence
        for sentence in split_sentence_units(source_text)
        if sentence in current_text
        and is_complete_sentence(sentence)
        and not is_metadata_text(sentence)
    ]
    candidates = list(dict.fromkeys(candidates))
    if len(candidates) < count:
        raise ValueError(
            f"not enough source-origin retrieval anchors: {len(candidates)} < {count}"
        )
    start = _stable_index(f"anchor:{record_id}", len(candidates))
    stride = max(1, len(candidates) // count)
    anchors = []
    for slot in range(count):
        index = (start + slot * stride) % len(candidates)
        while candidates[index] in anchors:
            index = (index + 1) % len(candidates)
        anchors.append(candidates[index])
    return anchors


def _stable_index(value: str, size: int) -> int:
    digest = hashlib.sha256(value.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "big") % size


def _select_semantic_candidates(
    assigned: tuple[DistractorSentence, ...],
    *,
    count: int,
    pool_size: int,
    question: str,
    source_text: str,
    enabled: bool,
    max_question_similarity: float,
    max_source_similarity: float,
) -> list[dict[str, Any]]:
    if not enabled:
        return [
            {"candidate": assigned[slot * pool_size]}
            for slot in range(count)
        ]

    from .quality import _semantic_embedding_model

    model, model_info = _semantic_embedding_model("BAAI/bge-m3")
    candidate_texts = [candidate.text for candidate in assigned]
    embeddings = model.encode(
        candidate_texts + [question, source_text],
        batch_size=max(2, len(candidate_texts) + 2),
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    question_embedding = embeddings[-2]
    source_embedding = embeddings[-1]
    selected = []
    for slot in range(count):
        options = []
        for index in range(slot * pool_size, (slot + 1) * pool_size):
            question_similarity = _dot(embeddings[index], question_embedding)
            source_similarity = _dot(embeddings[index], source_embedding)
            options.append(
                (
                    max(question_similarity, source_similarity),
                    question_similarity,
                    source_similarity,
                    assigned[index],
                )
            )
        viable = [
            option
            for option in options
            if option[1] <= max_question_similarity
            and option[2] <= max_source_similarity
        ]
        if not viable:
            best = min(options, key=lambda option: option[:3])
            raise ValueError(
                "no semantically distant distractor candidate for slot "
                f"{slot}: best question={best[1]:.3f}, source={best[2]:.3f}"
            )
        _, question_similarity, source_similarity, candidate = min(
            viable,
            key=lambda option: option[:3],
        )
        selected.append(
            {
                "candidate": candidate,
                "question_similarity": question_similarity,
                "source_similarity": source_similarity,
                "model": model_info["model"],
            }
        )
    return selected


def _dot(left: Any, right: Any) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


def _normalize(text: str) -> str:
    return " ".join(text.split())


def _ending_style(text: str) -> str:
    complete = [
        sentence
        for sentence in split_sentence_units(text)
        if is_complete_sentence(sentence)
    ]
    if not complete:
        return "unclassified"
    styles = Counter(_sentence_ending_style(sentence) for sentence in complete)
    return styles.most_common(1)[0][0]


def _sentence_ending_style(sentence: str) -> str:
    stripped = sentence.strip().rstrip(".!?。？！").rstrip()
    if _FORMAL_POLITE_RE.search(sentence.strip()):
        return "formal_polite"
    if stripped.endswith(("요", "죠")):
        return "haeyo"
    if stripped.endswith("다"):
        return "plain_da"
    return "other"
