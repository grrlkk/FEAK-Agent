"""Audit final Phase 3 files without APIs, training data, or reward implementation."""

import argparse
from collections import Counter
from pathlib import Path

from ..common import DEFAULT_CONFIG, file_sha, load_config, read_json, sha_text, write_json
from ..data_policy import assert_data_tree_clean
from ..phase2 import read_jsonl
from ..view_data import view_fingerprint
from ..corrupt.document import BareunBank, Document, source_document
from ..corrupt.operators import exact_restoration_satisfies, restore_record
from ..corrupt.qc import qc_payload, validate_judgments
from ..corrupt.sources import select_sources


def verify(config, *, candidates=False):
    output = config["paths"]["phase3_output"]
    gate = read_json(output / "qc_results.json")
    if not gate["complete"] or gate["api_calls_reserved"] > 100:
        raise ValueError("Incomplete QC or API budget exceeded")
    preserved = read_json(output / "preserved_hashes.json")
    # Chain the accepted earlier manifest, which covers Phase 2 EC and Phase 2b judgments.
    preserved.update(read_json(config["paths"]["phase2c_output"] / "preserved_hashes.json"))
    changed = [p for p, digest in preserved.items() if file_sha(Path(p)) != digest]
    if changed:
        raise ValueError(f"Frozen analyzer/judgment files changed: {changed}")
    bank, results, question_sets = BareunBank(config), {}, {}
    for split in ("agent_train", "agent_dev"):
        path = (output / f"pre_qc_{split}.jsonl" if candidates else
                config["paths"]["metadata"] / "corrupt" / (split + ".jsonl"))
        rows = read_jsonl(path)
        build_stats = read_json(path.with_suffix(".stats.json"))
        sources, _ = select_sources(config, split)
        source_by_id = {e.id: e for e in sources}
        # Seed every source (also possible same-split donors); verification must
        # not silently re-contact Bareun for already recorded observations.
        original_docs = {e.id: source_document(config, e, bank) for e in sources}
        scores = read_json(output / f"sources_{split}.json")["rows"]
        expected = {sid for sid, r in scores.items() if r["eligible"]}
        counts = Counter(r["source_id"] for r in rows)
        if set(counts) != expected or any(v != 2 for v in counts.values()):
            raise ValueError("Final dataset must have exactly two essays per selected parseable source")
        question_sets[split] = {r["question_hash"] for r in rows}
        if len({r["episode_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate episode ID")
        for row in rows:
            source = source_by_id[row["source_id"]]
            if row["source_text"] != source.text or row["question"] != source.question:
                raise ValueError("Episode provenance mismatch")
            if row["source_hash"] != sha_text(source.text) or row["corrupted_hash"] != sha_text(row["corrupted_text"]):
                raise ValueError("Episode text hash mismatch")
            if not candidates and (row["q_corrupted"] is None or not 1 <= row["q_corrupted"] <= 9):
                raise ValueError("Missing real Kanana Q")
            if not row["compact_tokens"] <= 3000 or row["source_text"] == row["corrupted_text"]:
                raise ValueError("Empty corruption or compact budget violation")
            used, global_n, local_n = set(), 0, 0
            current = Document.restore(row["corrupted_layout"], bank)
            for record in reversed(row["records"]):
                if (not candidates and record["op"] in gate["disabled_operators"]) or used.intersection(record["sids"]):
                    raise ValueError("Disabled operator or multiple operators on one sentence")
                used.update(record["sids"])
                global_n += record["level"] == "GLOBAL"
                local_n += record["level"] != "GLOBAL"
                current = restore_record(current, record, bank)
            allowed = {"L1": {(0,1)}, "L2": {(0,2),(0,3)}, "L3": {(1,0),(1,1)}, "L4": {(2,1),(2,2)}}
            if (global_n, local_n) not in allowed[row["level"]] or current.text != source.text:
                raise ValueError("Curriculum or exact inverse failed")
            if not all(exact_restoration_satisfies(original_docs[source.id], r) for r in row["records"]):
                raise ValueError("Source does not simultaneously satisfy all recorded targets")
            qc_payload(row)  # Every transition's complete text matches its record hash.
        results[split] = {"sources": len(counts), "episodes": len(rows), "sha256": file_sha(path),
            "per_source": 2, "operators": dict(Counter(r["op"] for row in rows for r in row["records"])),
            "operator_levels": dict(Counter(r["level"] for row in rows for r in row["records"])),
            "curriculum_levels": dict(Counter(row["level"] for row in rows)),
            "max_compact_tokens": max(row["compact_tokens"] for row in rows),
            "q_corrupted_missing": sum(row["q_corrupted"] is None for row in rows),
            "limited_unbalanced": build_stats.get("limited_unbalanced", False)}
    if question_sets["agent_train"] & question_sets["agent_dev"]:
        raise ValueError("Question leakage")
    deny = set(read_json(config["paths"]["metadata"] / "audit_index.json")["train"]["essay_hashes"])
    if candidates:
        # Restrict to candidate records, not arbitrary old files in outputs/.
        def check(value):
            if isinstance(value, str) and sha_text(value) in deny:
                raise ValueError("Exact training essay in candidate data")
            if isinstance(value, dict):
                for item in value.values(): check(item)
            if isinstance(value, list):
                for item in value: check(item)
        for split in ("agent_train", "agent_dev"):
            for row in read_jsonl(output / f"pre_qc_{split}.jsonl"): check(row)
    else:
        assert_data_tree_clean(config["paths"]["metadata"] / "corrupt", set(deny))
    for row in read_jsonl(output / "qc_sample.jsonl"):
        validate_judgments(row, row["llm_judgment"])
        if row["human_ok"] is not None:
            raise ValueError("LLM verification must not be mislabeled human review")
    result = {"datasets": results, "preserved_files": len(preserved), "changed_preserved_files": changed,
        "question_overlap": 0, "train_exact_essay_matches": 0, "view_fingerprint": view_fingerprint(config),
        "api_calls": gate["api_calls_reserved"], "reward_implemented": False, "phase4_started": False}
    result.update(candidate_only=candidates,
                  balanced_final_dataset_ready=not candidates and not any(
                      value["limited_unbalanced"] for value in results.values()))
    write_json(output / ("candidate_validation.json" if candidates else "final_validation.json"), result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--candidates", action="store_true", help="Audit private unscored pre-QC candidates only")
    args = parser.parse_args()
    result = verify(load_config(args.config), candidates=args.candidates)
    print({k: v["episodes"] for k, v in result["datasets"].items()})


if __name__ == "__main__":
    main()
