"""100 dev whitespace/paraphrase comparisons, bounded by the shared Phase 1 budget."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

from ..api import PhaseTeacher
from ..calibration import (choose_sentence, generate_paraphrase, make_analyzer,
                           summarize_noise, whitespace_variant)
from ..common import DEFAULT_CONFIG, load_config, read_json, sha_text, write_json
from ..data_policy import assert_not_training_essay, load_examples, stratified_sample
from ..score.kanana import KananaScorer


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["agent_dev"], default="agent_dev")
    parser.add_argument("--n", type=int, default=100)
    parser.add_argument("--max-api-calls", required=True, type=int)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if args.n < 1 or not 1 <= args.workers <= 8:
        parser.error("n must be positive and workers must be 1–8")
    config = load_config(args.config)
    output, data = config["paths"]["output"], config["paths"]["metadata"]
    teacher = PhaseTeacher(config, args.max_api_calls)
    baseline_rows = [json.loads(line) for line in (output / "scorer_benchmark.jsonl").read_text().splitlines() if line.strip()]
    baseline = {row["id"]: row["result"] for row in baseline_rows if row["status"] == "ok"}
    examples = [example for example in load_examples(config, args.split)
                if example.id in baseline and baseline[example.id]["input_tokens"] <= 3000]
    analyzer, targets, eligible, exclusions = make_analyzer(config), {}, [], []
    for example in examples:
        try:
            target = choose_sentence(example, analyzer.profile(example.text))
            whitespace_variant(example.text)
            targets[example.id] = target
            eligible.append(example)
        except ValueError as exc:
            exclusions.append({**example.metadata(), "reason": str(exc)})
    selection = stratified_sample(eligible, args.n, 1013)
    manifest = {"seed": 1013, "selected": [example.metadata() for example in selection],
        "selection_rule": "Balanced genre sample from frozen parsed benchmark; <=3000 input tokens and a 40–250-character complete sentence",
        "eligibility_exclusions": exclusions,
        "scorer_fingerprint": read_json(output / "scorer_metrics.json")["scorer_fingerprint"]}
    manifest_path = output / "calibration_manifest.json"
    if manifest_path.exists() and read_json(manifest_path) != manifest:
        raise ValueError("Refusing to overwrite a different frozen calibration sample")
    write_json(manifest_path, manifest)
    deny = set(read_json(data / "audit_index.json")["train"]["essay_hashes"])
    variants = {}
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(generate_paraphrase, example, targets[example.id],
            teacher=teacher, config=config, deny_hashes=deny): example.id for example in selection}
        for future in as_completed(futures):
            key = futures[future]
            variants[key] = future.result()
            if len(variants) % 5 == 0:
                print(f"paraphrases {len(variants)}/{args.n}; API reservations={teacher.budget.used}", flush=True)
    failed = [row for row in variants.values() if row["status"] != "ok"]
    write_json(output / "calibration_generation_status.json", {
        "requested": args.n, "valid": len(variants)-len(failed), "failed": failed,
        "phase_api_calls": teacher.budget.used})
    if failed:
        raise SystemExit(f"{len(failed)} paraphrases still need valid replacements; cached successes retained")
    scorer = KananaScorer(config)
    if scorer.fingerprint != manifest["scorer_fingerprint"]:
        raise ValueError("Calibration scorer differs from frozen benchmark")
    journal = output / "noise_samples.jsonl"
    records = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()] if journal.exists() else []
    seen = {(row["id"], row["variant"]) for row in records}
    if len(seen) != len(records) or not {row["id"] for row in records} <= {e.id for e in selection}:
        raise ValueError("Noise journal does not match frozen sample")
    with journal.open("a", encoding="utf-8") as handle:
        for example in selection:
            before = baseline[example.id]
            whitespace, location = whitespace_variant(example.text)
            para = variants[example.id]
            target = para["target"]
            paraphrase = example.text[:target["start"]] + para["replacement"] + example.text[target["end"]:]
            assert sha_text(paraphrase) == para["candidate_hash"]
            for name, text in (("whitespace", whitespace), ("paraphrase", paraphrase)):
                if (example.id, name) in seen:
                    continue
                assert_not_training_essay(text, deny)
                after = scorer.score(example.question, text).to_dict()
                row = {**example.metadata(), "variant": name, "candidate_hash": sha_text(text),
                       "q_before": before["mean"], "q_after": after["mean"],
                       "delta": after["mean"]-before["mean"], "after": after}
                if name == "whitespace":
                    row["inserted_space_at"] = location
                else:
                    row["surface_checks"] = para["surface_checks"]
                    row["semantic_check"] = para["semantic_check"]
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
                handle.flush()
                records.append(row)
                if len(records) % 10 == 0:
                    print(f"noise scores {len(records)}/{2*args.n}", flush=True)
    summary = summarize_noise(records, args.n)
    summary.update(scorer_fingerprint=scorer.fingerprint, phase_api_calls=teacher.budget.used)
    write_json(data / "scorer_noise.json", summary)
    lines = ["# Phase 1 scorer noise calibration", "", f"Essays: {args.n}; population standard deviation (ddof=0).", "",
        "| Variant | Genre | n | Mean abs(ΔQ) | Std abs(ΔQ) | Mean ΔQ | Std ΔQ |",
        "|---|---|---:|---:|---:|---:|---:|"]
    for name, stats in summary["variants"].items():
        for genre, values in {"ALL": stats["overall"], **stats["by_genre"]}.items():
            lines.append(f"| {name} | {genre} | {values['n']} | {values['mean_abs_delta']:.8f} | {values['std_abs_delta']:.8f} | {values['mean_signed_delta']:.8f} | {values['std_signed_delta']:.8f} |")
    lines += ["", f"noise_floor = 2 × std(signed paraphrase ΔQ) = {summary['noise_floor']:.10f}", "",
              "Paraphrases passed length, sentence count, marker, number, register checks and a separate GPT semantic check.",
              "These checks are not independent human ground truth. No train/test essay was scored.", ""]
    (output / "noise_report.md").write_text("\n".join(lines), encoding="utf-8")
    scorer.close()
    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
