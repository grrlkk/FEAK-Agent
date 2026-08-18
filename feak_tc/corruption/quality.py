"""Corpus-level quality gates for generated corruption transitions."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
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
) -> Iterable[dict[str, str]]:
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
