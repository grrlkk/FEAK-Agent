"""Corpus-level quality gates for generated corruption transitions."""

from __future__ import annotations

import math
import os
import re
from collections import Counter, defaultdict
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


_WORD_RE = re.compile(r"[가-힣A-Za-z0-9]+")
_SPACE_RE = re.compile(r"\s+")


def audit_edit_artifacts(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Detect recurring generated-edit templates across distinct essays.

    Only inserted/replacement text is inspected. Source essay text and move/delete
    spans are excluded so naturally repeated source material cannot trip the gate.
    """

    enabled = bool(cfg.get("enabled", True))
    records = list(_generated_edit_records(rows))
    essay_ids = {record["essay_id"] for record in records}
    minimum_essays = max(
        int(cfg.get("min_distinct_essays", 3)),
        math.ceil(float(cfg.get("max_distinct_essay_fraction", 0.02)) * len(essay_ids)),
    )
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "rows": len(rows),
            "generated_edits": len(records),
            "distinct_essays": len(essay_ids),
            "violations": [],
        }
    if not records:
        return {
            "enabled": True,
            "passed": True,
            "rows": len(rows),
            "generated_edits": 0,
            "distinct_essays": 0,
            "minimum_distinct_essays": minimum_essays,
            "violation_count": 0,
            "violations": [],
        }

    patterns: dict[tuple[str, str], dict[str, Any]] = {}
    for record in records:
        signature = canonical_edit_signature(record["text"])
        _record_pattern(
            patterns,
            kind="exact_template",
            pattern=signature,
            record=record,
        )

        words = signature.split()
        for size in tuple(int(value) for value in cfg.get("word_ngram_sizes", [5, 6, 7, 8])):
            for ngram in _ngrams(words, size):
                _record_pattern(
                    patterns,
                    kind=f"word_{size}gram",
                    pattern=" ".join(ngram),
                    record=record,
                )

        compact = _SPACE_RE.sub("", signature)
        char_size = int(cfg.get("char_ngram_size", 20))
        for offset in range(max(0, len(compact) - char_size + 1)):
            _record_pattern(
                patterns,
                kind=f"char_{char_size}gram",
                pattern=compact[offset : offset + char_size],
                record=record,
            )

    violations = []
    for (kind, pattern), detail in patterns.items():
        distinct = len(detail["essay_ids"])
        if distinct < minimum_essays:
            continue
        violations.append(
            {
                "kind": kind,
                "pattern": pattern,
                "distinct_essays": distinct,
                "essay_fraction": distinct / max(1, len(essay_ids)),
                "occurrences": detail["occurrences"],
                "operators": dict(sorted(detail["operators"].items())),
                "affected_pair_ids": sorted(detail["pair_ids"]),
                "example_pair_ids": sorted(detail["pair_ids"])[:5],
            }
        )

    violations.sort(
        key=lambda item: (
            -int(item["distinct_essays"]),
            -int(item["occurrences"]),
            str(item["kind"]),
            str(item["pattern"]),
        )
    )
    limit = int(cfg.get("report_limit", 50))
    return {
        "enabled": True,
        "passed": not violations,
        "rows": len(rows),
        "generated_edits": len(records),
        "distinct_essays": len(essay_ids),
        "minimum_distinct_essays": minimum_essays,
        "word_ngram_sizes": list(cfg.get("word_ngram_sizes", [5, 6, 7, 8])),
        "char_ngram_size": int(cfg.get("char_ngram_size", 20)),
        "violation_count": len(violations),
        "violations": violations[:limit],
    }


def audit_operator_balance(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Report whether one operator dominates the accepted training pool."""

    counts = Counter(str(row["corruption_op"]) for row in rows)
    total = sum(counts.values())
    maximum = float(cfg.get("max_operator_fraction", 0.40))
    fractions = {
        operator: count / total if total else 0.0
        for operator, count in sorted(counts.items())
    }
    dominant = max(fractions, key=fractions.get) if fractions else None
    dominant_fraction = fractions.get(dominant, 0.0) if dominant is not None else 0.0
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "passed": not bool(cfg.get("enabled", True)) or dominant_fraction <= maximum,
        "rows": total,
        "max_operator_fraction": maximum,
        "counts": dict(sorted(counts.items())),
        "fractions": fractions,
        "dominant_operator": dominant,
        "dominant_fraction": dominant_fraction,
    }


def audit_semantic_edit_artifacts(
    rows: Sequence[Mapping[str, Any]],
    cfg: Mapping[str, Any],
    *,
    encode_fn: Any = None,
) -> dict[str, Any]:
    """Detect paraphrased edit templates and retrieval provenance defects."""

    enabled = bool(cfg.get("enabled", False))
    operators = {str(value) for value in cfg.get("operators", ["INSERT_OFFTOPIC"])}
    records = [
        record
        for record in _generated_edit_records(rows)
        if record["operator"] in operators
    ]
    if not enabled:
        return {
            "enabled": False,
            "passed": True,
            "generated_edits": len(records),
            "violations": [],
        }
    if not records:
        return {
            "enabled": True,
            "passed": True,
            "operators": sorted(operators),
            "generated_edits": 0,
            "distinct_essays": 0,
            "violations": [],
        }

    essay_ids = {record["essay_id"] for record in records}
    minimum_essays = max(
        int(cfg.get("min_distinct_essays", 3)),
        math.ceil(float(cfg.get("max_distinct_essay_fraction", 0.02)) * len(essay_ids)),
    )
    duplicate_texts = [
        text
        for text, count in Counter(record["text"] for record in records).items()
        if count > 1
    ]
    provenance_failures = []
    for record in records:
        donor_id = record.get("distractor_record_id", "")
        donor_question = record.get("distractor_question", "")
        target_question = record.get("question", "")
        reasons = []
        if not donor_id or not donor_question:
            reasons.append("missing_retrieval_provenance")
        if donor_id and donor_id == record["essay_id"]:
            reasons.append("same_source_essay")
        if (
            donor_question
            and target_question
            and _normalize_text(donor_question) == _normalize_text(target_question)
        ):
            reasons.append("same_source_question")
        if reasons:
            provenance_failures.append(
                {
                    "pair_id": record["pair_id"],
                    "reasons": reasons,
                }
            )

    first_tokens = Counter(_first_token(record["text"]) for record in records)
    first_tokens.pop("", None)
    dominant_first_token, dominant_first_count = (
        first_tokens.most_common(1)[0] if first_tokens else (None, 0)
    )
    dominant_first_fraction = dominant_first_count / max(1, len(records))
    cue_terms = tuple(str(value) for value in cfg.get("cue_terms", []))
    cue_records = [
        record for record in records if any(term in record["text"] for term in cue_terms)
    ]
    cue_fraction = len(cue_records) / max(1, len(records))

    relevance_enabled = bool(cfg.get("relevance_enabled", False))
    logged_relevance = relevance_enabled and all(
        isinstance(record.get("distractor_question_similarity"), (int, float))
        and isinstance(record.get("distractor_source_similarity"), (int, float))
        for record in records
    )
    model_info: dict[str, Any]
    try:
        if encode_fn is None:
            requested_model = str(cfg.get("model", "BAAI/bge-m3"))
            model, model_info = _semantic_embedding_model(requested_model)
            embedding_texts = [record["text"] for record in records]
            if relevance_enabled and not logged_relevance:
                embedding_texts.extend(record["question"] for record in records)
                embedding_texts.extend(record["text_before"] for record in records)
            all_embeddings = model.encode(
                embedding_texts,
                batch_size=int(cfg.get("batch_size", 32)),
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            embeddings = all_embeddings[: len(records)]
        else:
            embedding_texts = [record["text"] for record in records]
            if relevance_enabled and not logged_relevance:
                embedding_texts.extend(record["question"] for record in records)
                embedding_texts.extend(record["text_before"] for record in records)
            all_embeddings = encode_fn(embedding_texts)
            embeddings = all_embeddings[: len(records)]
            model_info = {"model": "injected_test_encoder"}
    except Exception as exc:
        return {
            "enabled": True,
            "passed": False,
            "generated_edits": len(records),
            "distinct_essays": len(essay_ids),
            "model_error": f"{exc.__class__.__name__}: {exc}",
            "duplicate_texts": duplicate_texts,
            "provenance_failures": provenance_failures,
            "violations": [],
        }

    threshold = float(cfg.get("similarity_threshold", 0.88))
    clusters = _semantic_clusters(records, embeddings, threshold)
    relevance_violations = []
    if relevance_enabled:
        question_limit = float(cfg.get("max_question_similarity", 0.50))
        source_limit = float(cfg.get("max_source_similarity", 0.50))
        count = len(records)
        for index, record in enumerate(records):
            if logged_relevance:
                question_similarity = float(
                    record["distractor_question_similarity"]
                )
                source_similarity = float(record["distractor_source_similarity"])
            else:
                question_similarity = _embedding_dot(
                    all_embeddings[index], all_embeddings[count + index]
                )
                source_similarity = _embedding_dot(
                    all_embeddings[index], all_embeddings[2 * count + index]
                )
            if question_similarity > question_limit or source_similarity > source_limit:
                relevance_violations.append(
                    {
                        "pair_id": record["pair_id"],
                        "text": record["text"],
                        "question_similarity": question_similarity,
                        "source_similarity": source_similarity,
                    }
                )
    violations = []
    for indices in clusters:
        cluster_essays = {records[index]["essay_id"] for index in indices}
        if len(cluster_essays) < minimum_essays:
            continue
        pairs = [records[index]["pair_id"] for index in indices]
        violations.append(
            {
                "distinct_essays": len(cluster_essays),
                "edit_count": len(indices),
                "pair_ids": sorted(pairs),
                "examples": [records[index]["text"] for index in indices[:5]],
            }
        )
    violations.sort(key=lambda item: (-item["distinct_essays"], -item["edit_count"]))

    first_token_limit = float(cfg.get("max_first_token_fraction", 1.0))
    cue_limit = float(cfg.get("max_cue_fraction", 1.0))
    distribution_failures = []
    if dominant_first_fraction > first_token_limit:
        distribution_failures.append("dominant_first_token")
    if cue_fraction > cue_limit:
        distribution_failures.append("cue_term_concentration")
    passed = not (
        duplicate_texts
        or provenance_failures
        or violations
        or relevance_violations
        or distribution_failures
    )
    return {
        "enabled": True,
        "passed": passed,
        "operators": sorted(operators),
        "generated_edits": len(records),
        "distinct_essays": len(essay_ids),
        "model": model_info,
        "similarity_threshold": threshold,
        "minimum_distinct_essays": minimum_essays,
        "duplicate_texts": duplicate_texts,
        "provenance_failures": provenance_failures,
        "semantic_cluster_count": len(clusters),
        "violation_count": len(violations),
        "violations": violations[: int(cfg.get("report_limit", 30))],
        "relevance_enabled": relevance_enabled,
        "relevance_similarity_source": (
            "generation_log" if logged_relevance else "audit_recomputed"
        ),
        "max_question_similarity": float(cfg.get("max_question_similarity", 0.50)),
        "max_source_similarity": float(cfg.get("max_source_similarity", 0.50)),
        "relevance_violation_count": len(relevance_violations),
        "relevance_violations": relevance_violations[: int(cfg.get("report_limit", 30))],
        "first_token_counts": dict(first_tokens.most_common(20)),
        "dominant_first_token": dominant_first_token,
        "dominant_first_token_fraction": dominant_first_fraction,
        "max_first_token_fraction": first_token_limit,
        "cue_terms": list(cue_terms),
        "cue_edit_count": len(cue_records),
        "cue_fraction": cue_fraction,
        "max_cue_fraction": cue_limit,
        "distribution_failures": distribution_failures,
    }


def canonical_edit_signature(text: str) -> str:
    """Normalize edit text and collapse variable repeated-token templates."""

    words = _WORD_RE.findall(text.lower())
    canonical: list[str] = []
    index = 0
    while index < len(words):
        if (
            index + 2 < len(words)
            and words[index] == words[index + 1]
            and _same_repeated_stem(words[index], words[index + 2])
        ):
            canonical.append("<REP>")
            index += 3
            continue
        canonical.append(words[index])
        index += 1
    return " ".join(canonical)


def _generated_edit_records(
    rows: Sequence[Mapping[str, Any]],
) -> Iterable[dict[str, Any]]:
    for row in rows:
        essay_id = str(row.get("essay_id") or row.get("record_id") or "")
        pair_id = f"{essay_id}:stage{row.get('stage_k', '?')}"
        operator = str(row.get("corruption_op") or "")
        for edit in row.get("edits", []):
            operation = str(edit.get("operation") or "")
            text = str(edit.get("text") or "").strip()
            if operation not in {"insert_after", "replace"} or not text:
                continue
            yield {
                "essay_id": essay_id,
                "pair_id": pair_id,
                "operator": operator,
                "text": text,
                "question": str(row.get("question") or ""),
                "distractor_record_id": str(edit.get("distractor_record_id") or ""),
                "distractor_question": str(edit.get("distractor_question") or ""),
                "text_before": str(row.get("text_before") or ""),
                "distractor_question_similarity": edit.get(
                    "distractor_question_similarity"
                ),
                "distractor_source_similarity": edit.get(
                    "distractor_source_similarity"
                ),
            }


def _record_pattern(
    patterns: dict[tuple[str, str], dict[str, Any]],
    *,
    kind: str,
    pattern: str,
    record: Mapping[str, str],
) -> None:
    if not pattern:
        return
    detail = patterns.setdefault(
        (kind, pattern),
        {
            "essay_ids": set(),
            "pair_ids": set(),
            "operators": Counter(),
            "occurrences": 0,
        },
    )
    detail["essay_ids"].add(record["essay_id"])
    detail["pair_ids"].add(record["pair_id"])
    detail["operators"][record["operator"]] += 1
    detail["occurrences"] += 1


def _ngrams(words: Sequence[str], size: int) -> Iterable[tuple[str, ...]]:
    if size <= 0:
        return
    seen = set()
    for index in range(max(0, len(words) - size + 1)):
        value = tuple(words[index : index + size])
        if value not in seen:
            seen.add(value)
            yield value


def _same_repeated_stem(first: str, third: str) -> bool:
    if first == third:
        return True
    particles = ("을", "를", "이", "가", "은", "는", "도", "만")
    return any(third == first + particle for particle in particles)


def _semantic_clusters(
    records: Sequence[Mapping[str, str]],
    embeddings: Sequence[Any],
    threshold: float,
) -> list[list[int]]:
    if len(records) != len(embeddings):
        raise ValueError("embedding count does not match generated edit count")
    parents = list(range(len(records)))

    def find(index: int) -> int:
        while parents[index] != index:
            parents[index] = parents[parents[index]]
            index = parents[index]
        return index

    def union(left: int, right: int) -> None:
        root_left = find(left)
        root_right = find(right)
        if root_left != root_right:
            parents[root_right] = root_left

    for left in range(len(records)):
        for right in range(left + 1, len(records)):
            similarity = sum(
                float(a) * float(b)
                for a, b in zip(embeddings[left], embeddings[right])
            )
            if similarity >= threshold:
                union(left, right)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(records)):
        grouped[find(index)].append(index)
    return [indices for indices in grouped.values() if len(indices) > 1]


def _first_token(text: str) -> str:
    words = _WORD_RE.findall(text.lower())
    return words[0] if words else ""


def _normalize_text(text: str) -> str:
    return _SPACE_RE.sub(" ", text).strip()


def _embedding_dot(left: Any, right: Any) -> float:
    return float(sum(float(a) * float(b) for a, b in zip(left, right)))


@lru_cache(maxsize=2)
def _semantic_embedding_model(model_name: str) -> tuple[Any, dict[str, Any]]:
    """Load the configured sentence encoder from a complete local snapshot."""

    from sentence_transformers import SentenceTransformer

    snapshot = _local_huggingface_snapshot(model_name)
    if snapshot is None:
        raise RuntimeError(f"no complete local Hugging Face snapshot for {model_name}")
    device = os.getenv("FEAK_EMBEDDING_DEVICE") or None
    model = SentenceTransformer(
        str(snapshot),
        device=device,
        local_files_only=True,
    )
    return model, {
        "model": model_name,
        "snapshot": str(snapshot),
        "device": device or "auto",
    }


def _local_huggingface_snapshot(model_name: str) -> Path | None:
    configured_home = os.getenv("HF_HOME")
    cache_root = (
        Path(configured_home).expanduser() / "hub"
        if configured_home
        else Path.home() / ".cache" / "huggingface" / "hub"
    )
    model_dir = cache_root / f"models--{model_name.replace('/', '--')}"
    ref = model_dir / "refs" / "main"
    candidates = []
    if ref.is_file():
        candidates.append(model_dir / "snapshots" / ref.read_text(encoding="utf-8").strip())
    snapshots = model_dir / "snapshots"
    if snapshots.is_dir():
        candidates.extend(sorted(snapshots.iterdir()))
    for candidate in candidates:
        if (
            (candidate / "config.json").is_file()
            and (candidate / "modules.json").is_file()
            and any(
                (candidate / filename).is_file()
                for filename in ("model.safetensors", "pytorch_model.bin")
            )
        ):
            return candidate.resolve()
    return None
