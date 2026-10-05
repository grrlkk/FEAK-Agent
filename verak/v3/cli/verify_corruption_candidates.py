"""Offline audit of Phase 3b candidates; no scorer or train/test source access."""

import argparse
from collections import Counter
from pathlib import Path

from ..common import DEFAULT_CONFIG, file_sha, load_config, read_json, sha_text, write_json
from ..corrupt.builder import load_sources
from ..corrupt.document import Document, MARKER
from ..corrupt.instance_policy import ACTIVE_LEVELS, POLARITY_PAIRS, back_reference, candidate_bank
from ..corrupt.operators import exact_restoration_satisfies, restore_record
from ..corrupt.qc import qc_payload
from ..phase2 import read_jsonl


def verify(config):
    output, root = config["paths"]["phase3b_output"], config["paths"]["repo"]
    preserved = read_json(output / "preserved_hashes.json")
    # Also retain the accepted earlier report and judgment preservation chain.
    preserved.update(read_json(config["paths"]["phase2c_output"] / "preserved_hashes.json"))
    changed = [p for p, digest in preserved.items() if file_sha(root / p) != digest]
    if changed:
        raise ValueError(f"Preserved files changed: {changed}")
    deny = set(read_json(config["paths"]["metadata"] / "audit_index.json")["train"]["essay_hashes"])
    def no_training_text(value):
        if isinstance(value, str) and sha_text(value) in deny:
            raise ValueError("Exact training essay in candidate data")
        if isinstance(value, dict):
            for item in value.values(): no_training_text(item)
        if isinstance(value, list):
            for item in value: no_training_text(item)
    class OfflineOnly:
        def profile(self, text):
            raise ValueError("Missing recorded Bareun observation; audit must stay offline")
    results, questions = {}, {}
    for split in ("agent_train", "agent_dev"):
        bank = candidate_bank(config)
        bank.analyzer = OfflineOnly()
        sources = {e.id: (e, doc, score) for e, doc, score in load_sources(config, split, bank)}
        path = output / f"candidates_{split}.jsonl"
        rows = read_jsonl(path)
        counts = Counter(r["source_id"] for r in rows)
        if set(counts) != set(sources) or any(n != 3 for n in counts.values()):
            raise ValueError("Three candidates per eligible source are required")
        if len({r["episode_id"] for r in rows}) != len(rows):
            raise ValueError("Duplicate episode IDs")
        variants = {(sid, r["corrupted_hash"]) for sid in counts for r in rows if r["source_id"] == sid}
        if len(variants) != len(rows):
            raise ValueError("Duplicate variants of a source")
        questions[split] = {r["question_hash"] for r in rows}
        for row in rows:
            e, source, score = sources[row["source_id"]]
            if (row["source_text"], row["question"], row["source_hash"], row["q_source"]) != (
                    e.text, e.question, e.essay_hash, score["mean"]):
                raise ValueError("Source provenance mismatch")
            if row["q_corrupted"] is not None or row["compact_tokens"] > 3000:
                raise ValueError("Unauthorized scoring or view overflow")
            if sha_text(row["corrupted_text"]) != row["corrupted_hash"] or row["corrupted_text"] == e.text:
                raise ValueError("Invalid corruption hash")
            no_training_text(row)
            qc_payload(row)
            current, used = Document.restore(row["corrupted_layout"], bank), set()
            for rec in reversed(row["records"]):
                if rec["op"] not in ACTIVE_LEVELS or ACTIVE_LEVELS[rec["op"]] != rec["level"]:
                    raise ValueError("Removed or mislabeled operator")
                if used.intersection(rec["sids"]):
                    raise ValueError("Overlapping direct corruption targets")
                used.update(rec["sids"])
                current = restore_record(current, rec, bank)
                if rec["op"] == "G_DELETE_SUPPORT":
                    units = current.units
                    index = next(i for i, u in enumerate(units) if u.sid == rec["sids"][0])
                    following = units[index+1]
                    if current.locate(rec["sids"][0])[1] == 0 or not back_reference(following.text):
                        raise ValueError("Deletion lacks the required next-sentence back reference")
                    if following.sid != rec["params"]["next_sentence_id"]:
                        raise ValueError("Deletion successor record mismatch")
                if rec["op"] == "L_POLARITY" and POLARITY_PAIRS.get(rec["params"]["old"]) != rec["params"]["new"]:
                    raise ValueError("Removed obligation-to-possibility pattern")
                if rec["op"].startswith("L_"):
                    for sid, original in rec["original_text"].items():
                        if MARKER.findall(original) != MARKER.findall(rec["corrupted_text"][sid]):
                            raise ValueError("Local operator changed an anonymization marker")
                    if any(a.multi_unit for a in current.structure().annotations if a.sid in rec["sids"]):
                        raise ValueError("Local operator targeted a multi-unit")
                if not exact_restoration_satisfies(source, rec):
                    raise ValueError("Source does not satisfy a recovery target")
            global_n = sum(r["level"] == "GLOBAL" for r in row["records"])
            allowed = {"L1": {(0,1)}, "L2": {(0,2),(0,3)}, "L3": {(1,0),(1,1)}, "L4": {(2,1),(2,2)}}
            if (global_n, len(row["records"])-global_n) not in allowed[row["level"]] or current.text != e.text:
                raise ValueError("Curriculum or inverse restoration failed")
        results[split] = {"sources": len(sources), "candidates": len(rows), "per_source": 3,
            "sha256": file_sha(path), "operators": dict(Counter(r["op"] for x in rows for r in x["records"])),
            "record_levels": dict(Counter(r["level"] for x in rows for r in x["records"])),
            "curriculum_levels": dict(Counter(r["level"] for r in rows)),
            "genres": dict(Counter(r["genre"] for r in rows)),
            "max_compact_tokens": max(r["compact_tokens"] for r in rows),
            "q_corrupted_null": sum(r["q_corrupted"] is None for r in rows)}
        print(f"Validated {split}: {len(rows)} candidates", flush=True)
    if questions["agent_train"] & questions["agent_dev"]:
        raise ValueError("Question leakage")
    result = {"datasets": results, "preserved_files": len(preserved), "changed_preserved_files": changed,
              "question_overlap": 0, "train_exact_matches": 0, "phase4_started": False}
    write_json(output / "candidate_validation.json", result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    verify(load_config(args.config))


if __name__ == "__main__":
    main()
