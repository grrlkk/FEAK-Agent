"""Phase 1b: k=1/3/5 noise comparison on the old 100 pairs; zero API calls."""

import argparse
from collections import Counter
from copy import deepcopy
import json

from ..averaged_calibration import KS, load_fixed_calibration, read_rows, summarize_averaged
from ..common import DEFAULT_CONFIG, file_sha, load_config, read_json, sha_text, write_json
from ..data_policy import assert_not_training_essay, stratified_sample
from ..score.averaging import VERSION, average_prefix, whitespace_probes
from ..score.kanana import KananaScorer


def append_row(path, row):
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        handle.flush()


def same_scores(first, second):
    return (first["positions"] == second["positions"] and first["mean"] == second["mean"]
        and all(a["expected"] == b["expected"] and a["integers"] == b["integers"]
                for a, b in zip(first["members"], second["members"]))
        and len(first["members"]) == len(second["members"]))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--repeat", type=int, default=20)
    args = parser.parse_args(argv)
    if args.repeat < 1 or args.repeat > 100:
        parser.error("repeat must be 1–100")
    config = load_config(args.config)
    data, phase1 = config["paths"]["metadata"], config["paths"]["output"]
    output = phase1.parent / "phase1b"
    output.mkdir(parents=True, exist_ok=True)
    cohort, old_manifest = load_fixed_calibration(config)
    seed = config["scorer"].get("average_seed", 13)
    deny = set(read_json(data / "audit_index.json")["train"]["essay_hashes"])
    for item in cohort:
        for text in item["inputs"].values():
            for probe in whitespace_probes(text, 5, seed=seed):
                assert_not_training_essay(probe.text, deny)
    repeats = {e.id for e in stratified_sample([x["example"] for x in cohort], args.repeat, 113)}
    manifest = {"version": VERSION, "seed": seed, "ks": list(KS),
        "source_sha256": file_sha(config["paths"]["valid"]),
        "old_split_sha256": file_sha(data / "splits_v1.json"),
        "new_split_sha256": file_sha(data / "splits.json"),
        "phase1_noise_sha256": file_sha(data / "scorer_noise.json"),
        "phase1_calibration_manifest_sha256": file_sha(phase1 / "calibration_manifest.json"),
        "scorer_fingerprint": old_manifest["scorer_fingerprint"],
        "selected": [{**x["example"].metadata(), "new_split": x["new_split"]} for x in cohort],
        "new_split_counts": dict(Counter(x["new_split"] for x in cohort)),
        "repeat_ids": sorted(repeats), "cache_used_for_measurement": False}
    frozen = output / "averaging_manifest.json"
    if frozen.exists() and read_json(frozen) != manifest:
        raise ValueError("Refusing to change the frozen averaging experiment")
    write_json(frozen, manifest)
    records = read_rows(output / "averaged_scores.jsonl")
    index = {(r["id"], r["variant"]): r for r in records}
    repeated = read_rows(output / "determinism.jsonl")
    repeat_index = {r["id"]: r for r in repeated}
    if len(index) != len(records) or len(repeat_index) != len(repeated):
        raise ValueError("Duplicate experiment rows")
    if not set(repeat_index) <= repeats or any(not r["identical"] for r in repeated):
        raise ValueError("Previous determinism failure or mismatched repeat cohort; stop")
    score_config = deepcopy(config)
    score_config["paths"]["output"] = output
    scorer = KananaScorer(score_config)
    try:
        if scorer.fingerprint != old_manifest["scorer_fingerprint"]:
            raise ValueError("Individual scorer changed since Phase 1")
        # Check all inputs up front; no length-based replacement of this fixed cohort.
        for item in cohort:
            for text in item["inputs"].values():
                for probe in whitespace_probes(text, 5, seed=seed):
                    scorer.input_tokens(item["example"].question, probe.text)
        for item in cohort:
            example = item["example"]
            for name, text in item["inputs"].items():
                key = (example.id, name)
                if key not in index:
                    result = scorer.score_averaged(example.question, text, 5, use_cache=False).to_dict()
                    baseline = item["previous"][name]
                    matches = result["members"][0]["expected"] == baseline["expected"]
                    row = {**example.metadata(), "variant": name, "input_hash": sha_text(text),
                        "new_split": item["new_split"], "phase1_k1_identical": matches, "result": result}
                    append_row(output / "averaged_scores.jsonl", row)
                    records.append(row)
                    index[key] = row
                    if not matches:
                        raise RuntimeError(f"Determinism failed against Phase 1 at {example.id}/{name}")
                    print(f"scored inputs {len(records)}/300; repeats {len(repeated)}/{args.repeat}", flush=True)
                row = index[key]
                if row["input_hash"] != sha_text(text) or not row["phase1_k1_identical"]:
                    raise RuntimeError("Stored input or determinism check failed; stop")
                if name == "original" and example.id in repeats and example.id not in repeat_index:
                    second = scorer.score_averaged(example.question, text, 5, use_cache=False).to_dict()
                    result = row["result"]
                    checks = {str(k): average_prefix(result["members"], k) == average_prefix(second["members"], k) for k in KS}
                    record = {**example.metadata(), "identical": same_scores(result, second) and all(checks.values()),
                        "averaged_Q_identical_by_k": checks, "cache_used": False, "first": result, "second": second}
                    append_row(output / "determinism.jsonl", record)
                    repeated.append(record)
                    repeat_index[example.id] = record
                    if not record["identical"]:
                        raise RuntimeError(f"Averaged scorer determinism failed at {example.id}")
        if set(repeat_index) != repeats:
            raise ValueError("Incomplete determinism cohort")
        summary = summarize_averaged(records, repeat_records=repeated,
            expected_ids=[x["example"].id for x in cohort])
        if file_sha(data / "scorer_noise.json") != manifest["phase1_noise_sha256"]:
            raise ValueError("Original scorer_noise.json changed")
        summary.update(version=VERSION, seed=seed, scorer_fingerprint=scorer.fingerprint,
            cohort="Frozen Phase 1 calibration; not resampled from the new split",
            new_split_counts=manifest["new_split_counts"], phase1_k1_comparisons=len(records),
            limits=["Same-model GPT paraphrases from Phase 1 are not human meaning-preservation gold.",
                    "Only 100 fixed calibration essays; per-genre samples are small."])
        write_json(data / "scorer_noise_averaged.json", summary)
        write_json(output / "experiment_status.json", {"status": "completed", "decision": summary["decision"], "api_calls": 0})
        print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)
    except Exception as exc:
        write_json(output / "experiment_status.json", {"status": "stopped", "error": str(exc),
            "completed_inputs": len(records), "completed_repeats": len(repeated), "api_calls": 0})
        raise
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
