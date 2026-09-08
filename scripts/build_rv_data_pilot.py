#!/usr/bin/env python
"""Build the 50-essay Revision Verifier transition-data feasibility pilot."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Mapping, Sequence

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.rv.generation import (
    LLM_CANDIDATE_TYPES,
    generate_llm_candidate_texts,
    validate_llm_candidate_texts,
)
from feak_tc.rv.pilot import (
    ResolvedTransition,
    anchor_digest,
    audit_corruption_data,
    build_candidate_rows,
    build_pilot_report,
    read_jsonl,
    resolve_training_rows,
    select_pilot_anchors,
)
from feak_tc.rv.schema import RV_SAMPLE_JSON_SCHEMA


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/rv_pilot.yaml")
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--workers", type=int)
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="write the source audit and schema without calling the LLM",
    )
    args = parser.parse_args()

    config_path = _project_path(args.config)
    with config_path.open(encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    sample_size = int(args.sample_size or cfg["sample_size"])
    workers = int(args.workers or cfg["llm"].get("workers", 1))
    if workers < 1:
        raise SystemExit("--workers must be positive")

    input_cfg = cfg["input"]
    output_cfg = cfg["output"]
    training_path = _project_path(input_cfg["training_pool"])
    raw_paths = [_project_path(path) for path in input_cfg["raw_chains"]]
    outputs = {name: _project_path(path) for name, path in output_cfg.items()}

    rows = read_jsonl(training_path)
    resolved, resolution_report = resolve_training_rows(rows, raw_paths)
    audit = audit_corruption_data(rows, resolved, resolution_report)
    _write_json(outputs["audit"], audit)
    _write_json(outputs["schema"], RV_SAMPLE_JSON_SCHEMA)
    print(
        f"audit: {len(rows)} transitions, {len(resolved)} exact raw joins, "
        f"{resolution_report['rows_with_next_state']} next-linked",
        flush=True,
    )
    if args.audit_only:
        print(f"wrote {outputs['audit']}", flush=True)
        print(f"wrote {outputs['schema']}", flush=True)
        return 0

    anchors = select_pilot_anchors(
        resolved,
        sample_size=sample_size,
        seed=int(cfg["seed"]),
    )
    llm_cfg = cfg["llm"]
    validation_cfg = llm_cfg.get("validation", {})
    cached = _load_valid_cache(
        outputs["llm_cache"],
        anchors,
        model=str(llm_cfg["model"]),
        validation_cfg=validation_cfg,
    )
    missing = [anchor for anchor in anchors if anchor.transition_id not in cached]
    print(
        f"selected: {len(anchors)} essays; cache hits={len(anchors) - len(missing)}, "
        f"uncached LLM anchors={len(missing)}, model={llm_cfg['model']}",
        flush=True,
    )
    generated = _generate_missing(
        missing,
        llm_cfg,
        cache_path=outputs["llm_cache"],
        workers=workers,
    )
    llm_by_transition = {**cached, **generated}

    pilot_rows: list[dict[str, Any]] = []
    for anchor in anchors:
        pilot_rows.extend(
            build_candidate_rows(
                anchor,
                llm_by_transition[anchor.transition_id],
                cfg["labels"]["mapping"],
                dataset_version=str(cfg["version"]),
                label_source=str(cfg["labels"]["source"]),
                llm_model=str(llm_cfg["model"]),
                llm_validation_cfg=validation_cfg,
            )
        )
    _write_jsonl(outputs["dataset"], pilot_rows)
    report = build_pilot_report(
        pilot_rows,
        requested_essays=sample_size,
        audit_path=_relative(outputs["audit"]),
        schema_path=_relative(outputs["schema"]),
        output_path=_relative(outputs["dataset"]),
    )
    report["generation"] = {
        "model": str(llm_cfg["model"]),
        "configured_reasoning_effort": llm_cfg.get("reasoning", {}).get("effort"),
        "cache_hits": len(cached),
        "newly_generated_essays_this_run": len(generated),
        "llm_generated_essays": len(anchors),
        "llm_candidate_types": list(LLM_CANDIDATE_TYPES),
        "trajectory_candidate_types": [
            "correct_repair",
            "further_corruption",
            "no_edit",
        ],
        "edit_replay_candidate_types": ["partial_repair"],
    }
    report["artifacts"]["dataset_sha256"] = _sha256(outputs["dataset"])
    report["artifacts"]["audit_sha256"] = _sha256(outputs["audit"])
    report["artifacts"]["schema_sha256"] = _sha256(outputs["schema"])
    _write_json(outputs["report"], report)
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    return 0 if report["passed"] else 2


def _generate_missing(
    anchors: Sequence[ResolvedTransition],
    llm_cfg: Mapping[str, Any],
    *,
    cache_path: Path,
    workers: int,
) -> dict[str, dict[str, str]]:
    if not anchors:
        return {}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    generated: dict[str, dict[str, str]] = {}
    failures: list[str] = []
    with ThreadPoolExecutor(max_workers=workers) as executor:
        futures = {
            executor.submit(generate_llm_candidate_texts, anchor, llm_cfg): anchor
            for anchor in anchors
        }
        completed = 0
        for future in as_completed(futures):
            anchor = futures[future]
            completed += 1
            try:
                texts = future.result()
            except Exception as exc:  # Preserve successful calls in the resume cache.
                failures.append(f"{anchor.transition_id}: {exc}")
                print(
                    f"[{completed}/{len(anchors)}] failed {anchor.transition_id}: {exc}",
                    flush=True,
                )
                continue
            generated[anchor.transition_id] = texts
            entry = {
                "transition_id": anchor.transition_id,
                "anchor_sha256": anchor_digest(anchor),
                "model": str(llm_cfg["model"]),
                "candidates": texts,
            }
            with cache_path.open("a", encoding="utf-8") as file:
                file.write(json.dumps(entry, ensure_ascii=False) + "\n")
            print(
                f"[{completed}/{len(anchors)}] generated {anchor.transition_id}",
                flush=True,
            )
    if failures:
        raise RuntimeError(
            f"{len(failures)} RV candidate calls failed; rerun to resume. "
            + " | ".join(failures[:5])
        )
    return generated


def _load_valid_cache(
    path: Path,
    anchors: Sequence[ResolvedTransition],
    *,
    model: str,
    validation_cfg: Mapping[str, Any],
) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    by_id = {anchor.transition_id: anchor for anchor in anchors}
    cached: dict[str, dict[str, str]] = {}
    for entry in read_jsonl(path):
        transition_id = str(entry.get("transition_id") or "")
        anchor = by_id.get(transition_id)
        if anchor is None:
            continue
        if entry.get("anchor_sha256") != anchor_digest(anchor) or entry.get("model") != model:
            continue
        texts = entry.get("candidates")
        if not isinstance(texts, Mapping):
            continue
        try:
            validate_llm_candidate_texts(anchor, texts, validation_cfg)
        except ValueError:
            continue
        cached[transition_id] = {
            name: str(texts[name]).strip() for name in LLM_CANDIDATE_TYPES
        }
    return cached


def _project_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _relative(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    temporary.replace(path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
