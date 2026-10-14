"""Phase 1b fixed-cohort reconstruction and aggregation; no API dependencies."""

from collections import Counter
import json
import statistics

from .common import pair_key, read_json, sha_text
from .data_policy import assert_not_training_essay, load_examples
from .score.averaging import average_prefix

KS = (1, 3, 5)


def read_rows(path):
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_fixed_calibration(config):
    """Use the old 100-essay cohort even if its new split membership has changed."""
    data, phase1 = config["paths"]["metadata"], config["paths"]["output"]
    frozen = read_json(phase1 / "calibration_manifest.json")
    chosen = frozen["selected"]
    if len(chosen) != 100 or len({r["id"] for r in chosen}) != 100:
        raise ValueError("Expected exactly the 100 frozen Phase 1 calibration essays")
    examples, membership = {}, {}
    for split in ("agent_train", "agent_dev"):
        for example in load_examples(config, split):
            examples[example.id] = example
            membership[example.id] = split
    old_dev = set(read_json(data / "splits_v1.json")["agent_dev"])
    deny = set(read_json(data / "audit_index.json")["train"]["essay_hashes"])
    baselines = {r["id"]: r["result"] for r in read_rows(phase1 / "scorer_benchmark.jsonl") if r["status"] == "ok"}
    noise = {(r["id"], r["variant"]): r for r in read_rows(phase1 / "noise_samples.jsonl")}
    result = []
    for metadata in chosen:
        example = examples[metadata["id"]]
        if example.metadata() != metadata or example.question_hash not in old_dev:
            raise ValueError("Frozen calibration identity differs from the source")
        variant = read_json(phase1 / "calibration_variants" / f"{pair_key(example.question, example.text)}.json")
        target = variant["target"]
        if variant["status"] != "ok" or example.text[target["start"]:target["end"]] != target["text"]:
            raise ValueError("Frozen paraphrase target differs")
        paraphrase = example.text[:target["start"]] + variant["replacement"] + example.text[target["end"]:]
        space = noise[(example.id, "whitespace")]
        p = space["inserted_space_at"]
        whitespace = example.text[:p] + " " + example.text[p:]
        if sha_text(paraphrase) != variant["candidate_hash"] or sha_text(whitespace) != space["candidate_hash"]:
            raise ValueError("Existing variant hash mismatch")
        inputs = {"original": example.text, "whitespace": whitespace, "paraphrase": paraphrase}
        for text in inputs.values():
            assert_not_training_essay(text, deny)
        result.append({"example": example, "new_split": membership[example.id], "inputs": inputs,
            "previous": {"original": baselines[example.id], "whitespace": space["after"],
                         "paraphrase": noise[(example.id, "paraphrase")]["after"]}})
    return result, frozen


def delta_statistics(deltas):
    absolute = [abs(d) for d in deltas]
    return {"n": len(deltas), "mean_signed_delta": statistics.mean(deltas),
        "std_signed_delta": statistics.pstdev(deltas),
        "mean_abs_delta": statistics.mean(absolute), "max_abs_delta": max(absolute),
        "ddof": 0}


def choose_average_k(floors, deterministic):
    if not deterministic:
        return {"proceed_phase2": False, "chosen_k": 1, "reason": "determinism_failed"}
    if floors[1] <= 0:
        return {"proceed_phase2": False, "chosen_k": 1, "reason": "zero_baseline_noise_floor"}
    reduction = 1 - floors[5] / floors[1]
    if reduction < .25:
        return {"proceed_phase2": False, "chosen_k": 1,
                "reason": "k5_noise_reduction_below_25_percent", "k5_reduction_fraction": reduction}
    chosen = next(k for k in KS if floors[k] <= 1.1 * floors[5])
    return {"proceed_phase2": True, "chosen_k": chosen,
            "reason": "smallest_k_within_10_percent_of_k5", "k5_reduction_fraction": reduction}


def summarize_averaged(records, *, repeat_records, expected_ids):
    index = {(r["id"], r["variant"]): r for r in records}
    expected = {(key, variant) for key in expected_ids for variant in ("original", "whitespace", "paraphrase")}
    if len(index) != len(records) or set(index) != expected:
        raise ValueError("Incomplete or duplicate averaged calibration records")
    for row in records:
        if row["result"]["k"] != 5 or any(r["cache_hit"] for r in row["result"]["members"]):
            raise ValueError("Calibration requires five uncached component scores")
    deterministic = bool(repeat_records) and all(r["identical"] for r in repeat_records)
    results = {}
    for k in KS:
        variants, times = {}, []
        for variant in ("whitespace", "paraphrase"):
            deltas = []
            for key in expected_ids:
                before, after = index[(key, "original")], index[(key, variant)]
                delta = average_prefix(after["result"]["members"], k) - average_prefix(before["result"]["members"], k)
                deltas.append({"id": key, "genre": before["genre"], "delta": delta})
            genres = sorted({r["genre"] for r in deltas})
            variants[variant] = {"overall": delta_statistics([r["delta"] for r in deltas]),
                "by_genre": {genre: delta_statistics([r["delta"] for r in deltas if r["genre"] == genre]) for genre in genres}}
        for key in expected_ids:
            row = index[(key, "original")]
            times.append({"genre": row["genre"], "seconds": sum(row["result"]["member_seconds"][:k])})
        def timing(rows):
            values = [r["seconds"] for r in rows]
            return {"n": len(values), "mean_seconds": statistics.mean(values),
                    "median_seconds": statistics.median(values), "max_seconds": max(values)}
        results[str(k)] = {"variants": variants,
            "noise_floor": 2 * variants["paraphrase"]["overall"]["std_signed_delta"],
            "noise_floor_by_genre": {g: 2 * s["std_signed_delta"] for g, s in variants["paraphrase"]["by_genre"].items()},
            "scoring_time_per_essay": {"overall": timing(times),
                "by_genre": {g: timing([r for r in times if r["genre"] == g]) for g in sorted({r["genre"] for r in times})}}}
    floors = {k: results[str(k)]["noise_floor"] for k in KS}
    return {"n_essays": len(expected_ids), "api_calls": 0, "ks": results,
        "noise_floor_formula": "2 * population_std(signed paraphrase averaged_Q_after - averaged_Q_before)",
        "timing_method": "Sum of the first k uncached component-call times on the same 100 originals; excludes model loading and probe/aggregation overhead",
        "determinism": {"repeated_essays": len(repeat_records), "passed": deterministic},
        "decision": choose_average_k(floors, deterministic)}
