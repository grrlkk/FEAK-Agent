"""Trajectory audit and dataset assembly for the RV data pilot."""

from __future__ import annotations

import hashlib
import json
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from .generation import (
    LLM_CANDIDATE_TYPES,
    validate_llm_candidate_texts,
    wrong_target_rubric,
)
from .labels import build_weak_labels
from .schema import CANDIDATE_TYPES, LABEL_FIELDS, validate_rv_sample


@dataclass(frozen=True)
class ResolvedTransition:
    """A measured transition joined back to its exact raw trajectory."""

    row: dict[str, Any]
    chain: dict[str, Any]
    source_path: str
    exact_match_count: int

    @property
    def essay_id(self) -> str:
        return str(self.row["essay_id"])

    @property
    def chain_id(self) -> str:
        return str(self.row["chain_id"])

    @property
    def stage_k(self) -> int:
        return int(self.row["stage_k"])

    @property
    def transition_id(self) -> str:
        return str(
            self.row.get("transition_id")
            or f"{self.chain_id}:stage{self.stage_k}"
        )

    @property
    def previous_text(self) -> str:
        return _state_text(self.chain["states"][self.stage_k - 1])

    @property
    def current_text(self) -> str:
        return _state_text(self.chain["states"][self.stage_k])

    @property
    def has_next(self) -> bool:
        return len(self.chain.get("states", [])) > self.stage_k + 1

    @property
    def next_text(self) -> str:
        if not self.has_next:
            raise ValueError(f"{self.transition_id} has no next trajectory state")
        return _state_text(self.chain["states"][self.stage_k + 1])

    @property
    def partial_text(self) -> str:
        return _replay_first_corruption_edit(self)

    @property
    def question(self) -> str:
        return str(self.row.get("question") or self.chain.get("question") or "").strip()

    @property
    def current_step(self) -> Mapping[str, Any]:
        return self.chain["steps"][self.stage_k - 1]

    @property
    def next_step(self) -> Mapping[str, Any] | None:
        steps = self.chain.get("steps", [])
        return steps[self.stage_k] if len(steps) > self.stage_k else None


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    with Path(path).open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def resolve_training_rows(
    rows: Sequence[Mapping[str, Any]],
    raw_chain_paths: Sequence[str | Path],
    *,
    strict: bool = True,
) -> tuple[list[ResolvedTransition], dict[str, Any]]:
    """Join measured rows to exact raw states using configured source priority."""

    by_essay: dict[str, list[tuple[str, dict[str, Any]]]] = defaultdict(list)
    raw_records = 0
    for raw_path in raw_chain_paths:
        path = Path(raw_path)
        for chain in read_jsonl(path):
            record_id = str(chain.get("record_id") or chain.get("essay_id") or "")
            by_essay[record_id].append((_portable_path(path), chain))
            raw_records += 1

    resolved: list[ResolvedTransition] = []
    source_counts: Counter[str] = Counter()
    unresolved: list[str] = []
    multiple_exact = 0
    metadata_mismatch_matches = 0
    divergent_next = 0
    rows_with_next = 0
    for raw_row in rows:
        row = dict(raw_row)
        essay_id = str(row["essay_id"])
        stage_k = int(row["stage_k"])
        matches: list[tuple[str, dict[str, Any]]] = []
        next_values: set[str] = set()
        for source_path, chain in by_essay.get(essay_id, []):
            states = chain.get("states") or []
            steps = chain.get("steps") or []
            if len(states) <= stage_k or len(steps) < stage_k:
                continue
            if (
                _state_text(states[stage_k - 1]) != str(row["text_before"])
                or _state_text(states[stage_k]) != str(row["text"])
            ):
                continue
            if not _step_matches(row, steps[stage_k - 1]):
                metadata_mismatch_matches += 1
                continue
            matches.append((source_path, chain))
            if len(states) > stage_k + 1:
                next_values.add(_state_text(states[stage_k + 1]))
        transition_id = str(
            row.get("transition_id") or f"{row['chain_id']}:stage{stage_k}"
        )
        if not matches:
            unresolved.append(transition_id)
            continue
        if len(matches) > 1:
            multiple_exact += 1
        if len(next_values) > 1:
            divergent_next += 1
        source_path, chain = matches[0]
        anchor = ResolvedTransition(
            row=row,
            chain=chain,
            source_path=source_path,
            exact_match_count=len(matches),
        )
        resolved.append(anchor)
        source_counts[Path(source_path).name] += 1
        rows_with_next += int(anchor.has_next)

    if strict and unresolved:
        raise ValueError(
            f"failed to resolve {len(unresolved)} training transitions; "
            f"first={unresolved[:5]}"
        )
    report = {
        "training_rows": len(rows),
        "raw_files": [_portable_path(Path(path)) for path in raw_chain_paths],
        "raw_records_scanned": raw_records,
        "resolved_exact_rows": len(resolved),
        "unresolved_rows": len(unresolved),
        "unresolved_transition_ids": unresolved[:50],
        "multiple_exact_match_rows": multiple_exact,
        "divergent_next_state_rows_across_versions": divergent_next,
        "exact_match_metadata_mismatches_skipped": metadata_mismatch_matches,
        "rows_with_next_state": rows_with_next,
        "source_priority_selection": dict(sorted(source_counts.items())),
    }
    return resolved, report


def audit_corruption_data(
    rows: Sequence[Mapping[str, Any]],
    resolved: Sequence[ResolvedTransition],
    resolution_report: Mapping[str, Any],
) -> dict[str, Any]:
    """Classify source fields as reusable, enrichable, or requiring regeneration."""

    total = len(rows)
    resolved_count = len(resolved)
    direct_counts = {
        field: sum(row.get(field) not in (None, "") for row in rows)
        for field in (
            "essay_id",
            "corruption_op",
            "target_rubric",
            "text_before",
            "text",
            "reverse_action",
        )
    }
    edit_rows = sum(
        bool(row.get("edits"))
        and all(str(edit.get("target_span") or "") for edit in row["edits"])
        for row in rows
    )
    rows_with_next = sum(anchor.has_next for anchor in resolved)
    question_direct = sum(bool(str(row.get("question") or "").strip()) for row in rows)
    question_joined = sum(bool(anchor.question) for anchor in resolved)
    field_audit = {
        "essay_id": _field_result(
            "directly_usable", direct_counts["essay_id"], resolved_count,
            "Stable ID is present in every measured transition.",
        ),
        "x0": _field_result(
            "metadata_enrichment_needed", 0, resolved_count,
            "Not copied into the training pool; recovered as states[0] by exact raw-chain join.",
        ),
        "x1_x2_sequence": _field_result(
            "metadata_enrichment_needed", 0, resolved_count,
            "The pool is transition-flat; ordered states are recovered from raw chains.",
        ),
        "corruption_type": _field_result(
            "directly_usable", direct_counts["corruption_op"], resolved_count,
            "Available as corruption_op.",
        ),
        "target_rubric": _field_result(
            "directly_usable", direct_counts["target_rubric"], resolved_count,
            "Measured transition and raw step metadata agree.",
        ),
        "changed_span": _field_result(
            "metadata_enrichment_needed", edit_rows, resolved_count,
            "Two textual edit records exist per row, but canonical changed_spans and "
            "offsets do not.",
        ),
        "before_text": _field_result(
            "directly_usable", direct_counts["text_before"], resolved_count,
            "Available as text_before and exactly matched to states[k-1].",
        ),
        "after_text": _field_result(
            "directly_usable", direct_counts["text"], resolved_count,
            "Available as text and exactly matched to states[k].",
        ),
        "previous_next_links": _field_result(
            "metadata_enrichment_needed", 0, rows_with_next,
            "Canonical state IDs are absent; raw state indices provide previous links for all "
            "resolved rows and next links where a later state exists.",
        ),
        "question": {
            "classification": "metadata_enrichment_needed",
            "training_pool_rows": question_direct,
            "after_trajectory_join_rows": question_joined,
            "notes": "Missing pool values are recovered from raw chain records.",
        },
    }
    return {
        "audit_version": "rv-corruption-audit-v1",
        "training_pool": {
            "rows": total,
            "essays": len({str(row["essay_id"]) for row in rows}),
            "operators": dict(sorted(Counter(str(row["corruption_op"]) for row in rows).items())),
            "stages": dict(sorted(Counter(str(row["stage_k"]) for row in rows).items())),
            "edit_count_per_transition": dict(
                sorted(Counter(str(len(row.get("edits") or [])) for row in rows).items())
            ),
        },
        "trajectory_resolution": dict(resolution_report),
        "field_audit": field_audit,
        "decision_summary": {
            "directly_usable": [
                name for name, item in field_audit.items()
                if item["classification"] == "directly_usable"
            ],
            "metadata_enrichment_needed": [
                name for name, item in field_audit.items()
                if item["classification"] == "metadata_enrichment_needed"
            ],
            "regeneration_needed": [],
            "source_transition_conclusion": (
                "All selected corruption transitions can be reused; no corruption regeneration "
                "is needed for this pilot."
            ),
        },
        "rv_candidate_strategy": {
            "trajectory_reuse": [
                "correct_repair",
                "further_corruption",
                "no_edit",
            ],
            "corruption_edit_replay": ["partial_repair"],
            "llm_generation_needed": list(LLM_CANDIDATE_TYPES),
        },
    }


def select_pilot_anchors(
    resolved: Sequence[ResolvedTransition],
    *,
    sample_size: int,
    seed: int,
) -> list[ResolvedTransition]:
    """Select one next-linked transition per essay, stratified by corruption type."""

    if sample_size < 1:
        raise ValueError("RV pilot sample_size must be positive")
    by_operator: dict[str, list[ResolvedTransition]] = defaultdict(list)
    for anchor in resolved:
        if anchor.has_next:
            by_operator[str(anchor.row["corruption_op"])].append(anchor)
    operators = sorted(by_operator)
    if len(operators) < 2:
        raise ValueError("RV pilot requires multiple corruption types")
    quotas = {
        operator: sample_size // len(operators) + int(index < sample_size % len(operators))
        for index, operator in enumerate(operators)
    }
    for operator in operators:
        by_operator[operator].sort(key=lambda item: _stable_key(item.transition_id, seed))

    selected: list[ResolvedTransition] = []
    used_essays: set[str] = set()
    for operator in operators:
        if quotas[operator] == 0:
            continue
        for anchor in by_operator[operator]:
            if anchor.essay_id in used_essays:
                continue
            selected.append(anchor)
            used_essays.add(anchor.essay_id)
            if sum(
                item.row["corruption_op"] == operator for item in selected
            ) >= quotas[operator]:
                break
    if len(selected) < sample_size:
        remaining = sorted(
            (
                anchor
                for anchors in by_operator.values()
                for anchor in anchors
                if anchor.essay_id not in used_essays
            ),
            key=lambda item: _stable_key(item.transition_id, seed),
        )
        for anchor in remaining:
            if anchor.essay_id in used_essays:
                continue
            selected.append(anchor)
            used_essays.add(anchor.essay_id)
            if len(selected) == sample_size:
                break
    if len(selected) != sample_size:
        raise ValueError(
            f"only {len(selected)} unique next-linked essays are available for "
            f"sample_size={sample_size}"
        )
    return sorted(selected, key=lambda item: item.transition_id)


def build_candidate_rows(
    anchor: ResolvedTransition,
    llm_texts: Mapping[str, str],
    labels_cfg: Mapping[str, Any],
    *,
    dataset_version: str,
    label_source: str,
    llm_model: str,
    llm_validation_cfg: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Combine trajectory, exact edit-replay, and LLM-generated candidates."""

    if not anchor.has_next:
        raise ValueError(f"{anchor.transition_id} cannot provide further_corruption")
    validate_llm_candidate_texts(anchor, llm_texts, llm_validation_cfg or {})
    texts = {
        "correct_repair": anchor.previous_text,
        "partial_repair": anchor.partial_text,
        **{name: str(llm_texts[name]).strip() for name in LLM_CANDIDATE_TYPES},
        "further_corruption": anchor.next_text,
        "no_edit": anchor.current_text,
    }
    sources = {
        "correct_repair": "trajectory_previous",
        "partial_repair": "corruption_edit_replay",
        "wrong_target": "llm",
        "over_edit": "llm",
        "further_corruption": "trajectory_next",
        "no_edit": "trajectory_current",
    }
    next_step = anchor.next_step or {}
    rows = []
    for candidate_type in CANDIDATE_TYPES:
        row = {
            "dataset_version": dataset_version,
            "sample_id": f"{anchor.transition_id}:{candidate_type}",
            "essay_id": anchor.essay_id,
            "chain_id": anchor.chain_id,
            "state_id": _state_id(anchor.chain_id, anchor.stage_k),
            "stage_k": anchor.stage_k,
            "previous_state_id": _state_id(anchor.chain_id, anchor.stage_k - 1),
            "next_state_id": _state_id(anchor.chain_id, anchor.stage_k + 1),
            "source_transition_id": anchor.transition_id,
            "question": anchor.question,
            "before_text": anchor.current_text,
            "after_text": texts[candidate_type],
            "target_rubric": str(anchor.row["target_rubric"]),
            "intended_action": str(anchor.row["reverse_action"]),
            "intent": str(anchor.row.get("intent") or ""),
            "corruption_type": str(anchor.row["corruption_op"]),
            "changed_spans": _normalize_changed_spans(anchor.row.get("edits") or []),
            "candidate_type": candidate_type,
            "candidate_source": sources[candidate_type],
            **build_weak_labels(
                candidate_type,
                labels_cfg,
                label_source=label_source,
            ),
            "provenance": {
                "raw_chain_file": anchor.source_path,
                "source_state_index": anchor.stage_k,
                "candidate_method": sources[candidate_type],
                "candidate_model": llm_model if sources[candidate_type] == "llm" else None,
                "requested_wrong_target_rubric": (
                    wrong_target_rubric(str(anchor.row["target_rubric"]))
                    if candidate_type == "wrong_target"
                    else None
                ),
                "further_corruption_type": (
                    str(next_step.get("corruption_op") or next_step.get("operator") or "")
                    if candidate_type == "further_corruption"
                    else None
                ),
                "further_target_rubric": (
                    str(next_step.get("target_rubric") or "")
                    if candidate_type == "further_corruption"
                    else None
                ),
            },
        }
        validate_rv_sample(row)
        rows.append(row)
    if len({row["after_text"] for row in rows}) != len(rows):
        raise ValueError(f"{anchor.transition_id} candidate texts are not all distinct")
    return rows


def build_pilot_report(
    rows: Sequence[Mapping[str, Any]],
    *,
    requested_essays: int,
    audit_path: str,
    schema_path: str,
    output_path: str,
) -> dict[str, Any]:
    """Summarize candidate, weak-label, source, and edit-size distributions."""

    failures: list[str] = []
    for row in rows:
        try:
            validate_rv_sample(row)
        except ValueError as exc:
            failures.append(f"{row.get('sample_id')}: {exc}")
    by_state: dict[str, list[Mapping[str, Any]]] = defaultdict(list)
    for row in rows:
        by_state[str(row["state_id"])].append(row)
    incomplete_states = {
        state_id: sorted(str(row["candidate_type"]) for row in state_rows)
        for state_id, state_rows in by_state.items()
        if set(str(row["candidate_type"]) for row in state_rows) != set(CANDIDATE_TYPES)
        or len(state_rows) != len(CANDIDATE_TYPES)
    }
    representatives = [state_rows[0] for state_rows in by_state.values()]
    edit_stats: dict[str, dict[str, float]] = {}
    for candidate_type in CANDIDATE_TYPES:
        values = [
            _edit_distance_ratio(str(row["before_text"]), str(row["after_text"]))
            for row in rows
            if row["candidate_type"] == candidate_type
        ]
        if values:
            edit_stats[candidate_type] = {
                "min": min(values),
                "median": statistics.median(values),
                "mean": statistics.fmean(values),
                "max": max(values),
            }
    essay_count = len({str(row["essay_id"]) for row in rows})
    expected_rows = requested_essays * len(CANDIDATE_TYPES)
    unique_samples = len({str(row["sample_id"]) for row in rows})
    comparative_checks = Counter()
    for state_rows in by_state.values():
        candidates = {str(row["candidate_type"]): row for row in state_rows}
        if set(candidates) != set(CANDIDATE_TYPES):
            continue
        before = str(state_rows[0]["before_text"])
        distances = {
            name: _edit_distance_ratio(before, str(candidates[name]["after_text"]))
            for name in CANDIDATE_TYPES
        }
        comparative_checks["partial_less_changed_than_correct"] += (
            distances["partial_repair"] < distances["correct_repair"]
        )
        comparative_checks["over_more_changed_than_partial"] += (
            distances["over_edit"] > distances["partial_repair"]
        )
        comparative_checks["over_more_changed_than_correct"] += (
            distances["over_edit"] > distances["correct_repair"]
        )
        comparative_checks["wrong_target_is_non_noop"] += distances["wrong_target"] > 0
    passed = (
        not failures
        and not incomplete_states
        and essay_count == requested_essays
        and len(by_state) == requested_essays
        and len(rows) == expected_rows
        and unique_samples == len(rows)
    )
    return {
        "report_version": "rv-data-pilot-v1",
        "passed": passed,
        "requested_essays": requested_essays,
        "essays": essay_count,
        "states": len(by_state),
        "rows": len(rows),
        "expected_rows": expected_rows,
        "unique_sample_ids": unique_samples,
        "candidate_types": dict(
            sorted(Counter(str(row["candidate_type"]) for row in rows).items())
        ),
        "candidate_sources": dict(
            sorted(Counter(str(row["candidate_source"]) for row in rows).items())
        ),
        "corruption_types": dict(
            sorted(Counter(str(row["corruption_type"]) for row in representatives).items())
        ),
        "target_rubrics": dict(
            sorted(Counter(str(row["target_rubric"]) for row in representatives).items())
        ),
        "stages": dict(sorted(Counter(str(row["stage_k"]) for row in representatives).items())),
        "weak_label_distributions": {
            field: dict(sorted(Counter(str(row[field]) for row in rows).items()))
            for field in LABEL_FIELDS
        },
        "labels_by_candidate_type": {
            candidate_type: {
                field: dict(
                    sorted(
                        Counter(
                            str(row[field])
                            for row in rows
                            if row["candidate_type"] == candidate_type
                        ).items()
                    )
                )
                for field in LABEL_FIELDS
            }
            for candidate_type in CANDIDATE_TYPES
        },
        "character_edit_ratio_by_candidate_type": edit_stats,
        "structural_candidate_checks": {
            name: {
                "passed_states": int(comparative_checks[name]),
                "total_states": len(by_state),
                "fraction": comparative_checks[name] / max(1, len(by_state)),
            }
            for name in (
                "partial_less_changed_than_correct",
                "over_more_changed_than_partial",
                "over_more_changed_than_correct",
                "wrong_target_is_non_noop",
            )
        },
        "raw_chain_sources": dict(
            sorted(
                Counter(
                    Path(str(row["provenance"]["raw_chain_file"])).name
                    for row in representatives
                ).items()
            )
        ),
        "integrity": {
            "schema_failures": failures[:50],
            "incomplete_states": incomplete_states,
            "all_labels_marked_weak": all(row["weak_supervision"] is True for row in rows),
            "one_state_per_essay": len(by_state) == essay_count,
        },
        "artifacts": {
            "dataset": output_path,
            "audit": audit_path,
            "schema": schema_path,
        },
        "scope": "Data feasibility pilot only; no Revision Verifier model was trained.",
    }


def anchor_digest(anchor: ResolvedTransition) -> str:
    payload = {
        "transition_id": anchor.transition_id,
        "previous": anchor.previous_text,
        "current": anchor.current_text,
        "next": anchor.next_text,
        "target_rubric": anchor.row["target_rubric"],
        "intended_action": anchor.row["reverse_action"],
        "corruption_type": anchor.row["corruption_op"],
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _step_matches(row: Mapping[str, Any], step: Mapping[str, Any]) -> bool:
    return all(
        str(row.get(row_key) or "") == str(step.get(step_key) or "")
        for row_key, step_key in (
            ("corruption_op", "corruption_op"),
            ("target_rubric", "target_rubric"),
            ("reverse_action", "reverse_action"),
            ("intent", "intent"),
        )
    ) and list(row.get("edits") or []) == list(step.get("edits") or [])


def _state_text(state: Any) -> str:
    if isinstance(state, Mapping):
        return str(state.get("text") or "")
    return str(state)


def _field_result(
    classification: str,
    training_count: int,
    joined_count: int,
    notes: str,
) -> dict[str, Any]:
    return {
        "classification": classification,
        "training_pool_rows": training_count,
        "after_trajectory_join_rows": joined_count,
        "notes": notes,
    }


def _normalize_changed_spans(edits: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"edit_index": index, **dict(edit)}
        for index, edit in enumerate(edits)
    ]


def _replay_first_corruption_edit(anchor: ResolvedTransition) -> str:
    """Apply one of two known corruptions, yielding an exact partial repair."""

    edits = list(anchor.row.get("edits") or [])
    if len(edits) < 2:
        raise ValueError(
            f"{anchor.transition_id} needs at least two edits for partial replay"
        )
    edit = edits[0]
    operation = str(edit.get("operation") or "")
    target = str(edit.get("target_span") or "")
    value = str(edit.get("text") or "")
    text = anchor.previous_text
    if operation == "delete":
        candidate = _replace_once(text, target, "")
    elif operation == "insert_after":
        candidate = _replace_once(text, target, f"{target} {value}")
    elif operation == "move_after":
        without = _replace_once(text, target, "")
        candidate = _replace_once(without, value, f"{value} {target}")
    elif operation == "replace":
        candidate = _replace_once(text, target, value)
    else:
        raise ValueError(f"unsupported corruption edit operation: {operation}")
    candidate = " ".join(candidate.split())
    if candidate in {anchor.previous_text, anchor.current_text}:
        raise ValueError(
            f"{anchor.transition_id} edit replay did not produce a partial state"
        )
    return candidate


def _replace_once(text: str, target: str, replacement: str) -> str:
    if not target or target not in text:
        raise ValueError("corruption edit target is missing during partial replay")
    return text.replace(target, replacement, 1)


def _portable_path(path: Path) -> str:
    project_root = Path(__file__).resolve().parents[2]
    try:
        return str(path.resolve().relative_to(project_root))
    except ValueError:
        return str(path)


def _state_id(chain_id: str, state_index: int) -> str:
    return f"{chain_id}:state{state_index}"


def _stable_key(identifier: str, seed: int) -> tuple[str, str]:
    digest = hashlib.sha256(f"{seed}:{identifier}".encode("utf-8")).hexdigest()
    return digest, identifier


def _edit_distance_ratio(before: str, after: str) -> float:
    return 1.0 - SequenceMatcher(None, before, after, autojunk=False).ratio()
