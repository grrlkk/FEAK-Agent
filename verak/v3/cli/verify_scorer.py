"""Frozen 200-essay parse benchmark and 20 uncached repeated scores on agent_dev."""

import argparse
from collections import Counter, defaultdict
import json
import time

from ..common import DEFAULT_CONFIG, load_config, read_json, write_json
from ..data_policy import load_examples, stratified_sample
from ..score.kanana import InputTooLong, KananaScorer, ScoreParseError


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--n", type=int, default=200)
    parser.add_argument("--repeat", type=int, default=20)
    parser.add_argument("--gpu", type=int)
    args = parser.parse_args(argv)
    config = load_config(args.config)
    if args.gpu is not None:
        config["scorer"]["gpu"] = args.gpu
    output = config["paths"]["output"]
    output.mkdir(parents=True, exist_ok=True)
    scorer = KananaScorer(config)
    examples = load_examples(config)
    eligible, too_long = [], []
    for example in examples:
        try:
            scorer.input_tokens(example.question, example.text)
            eligible.append(example)
        except InputTooLong as exc:
            too_long.append({**example.metadata(), "input_tokens": exc.tokens})
    selection = stratified_sample(eligible, args.n, config["split"]["seed"])
    manifest = {"scorer_fingerprint": scorer.fingerprint, "seed": config["split"]["seed"],
        "selected": [e.metadata() for e in selection], "too_long": too_long,
        "eligible_dev": len(eligible), "all_dev": len(examples)}
    path = output / "scorer_benchmark_manifest.json"
    if path.exists() and read_json(path) != manifest:
        raise ValueError("Refusing to overwrite a different frozen benchmark")
    write_json(path, manifest)
    rows = []
    journal = output / "scorer_benchmark.jsonl"
    if journal.exists():
        rows = [json.loads(line) for line in journal.read_text().splitlines() if line.strip()]
    seen = {row["id"] for row in rows}
    if len(seen) != len(rows) or not seen <= {e.id for e in selection}:
        raise ValueError("Benchmark journal does not match its manifest")
    with journal.open("a", encoding="utf-8") as handle:
        for example in selection:
            if example.id in seen:
                continue
            started = time.monotonic()
            row = example.metadata()
            try:
                result = scorer.score(example.question, example.text, use_cache=False)
                row.update(status="ok", result=result.to_dict())
            except ScoreParseError as exc:
                row.update(status="parse_error", error=str(exc))
            row["elapsed_s"] = time.monotonic() - started
            rows.append(row)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()
            if len(rows) % 10 == 0:
                print(f"scored {len(rows)}/{len(selection)}", flush=True)
    per_genre = {}
    for genre in sorted({e.genre for e in selection}):
        subset = [row for row in rows if row["genre"] == genre]
        successes = sum(row["status"] == "ok" for row in subset)
        per_genre[genre] = {"n": len(subset), "success": successes, "rate": successes / len(subset)}
    successes = {row["id"]: row for row in rows if row["status"] == "ok"}
    repeats = stratified_sample([e for e in selection if e.id in successes], args.repeat, 113)
    repeated = []
    for example in repeats:
        # Both original and repeat are true model computations, never cache hits.
        first = successes[example.id]["result"]
        second = scorer.score(example.question, example.text, use_cache=False).to_dict()
        repeated.append({**example.metadata(), "first": first, "second": second,
            "identical_expected": first["expected"] == second["expected"],
            "identical_integers": first["integers"] == second["integers"],
            "max_abs_delta": max(abs(a-b) for a,b in zip(first["expected"], second["expected"]))})
        write_json(output / "determinism.json", repeated)
    metrics = {"n": len(rows), "parse_success": len(successes),
        "parse_rate": len(successes) / len(rows), "by_genre": per_genre,
        "input_length_exclusions": len(too_long), "repeats": len(repeated),
        "identical_expected_repeats": sum(row["identical_expected"] for row in repeated),
        "determinism_used_cache": False, "scorer_fingerprint": scorer.fingerprint}
    write_json(output / "scorer_metrics.json", metrics)
    scorer.close()
    print(json.dumps(metrics, ensure_ascii=False, indent=2), flush=True)
    if metrics["parse_rate"] < .99 or metrics["identical_expected_repeats"] != args.repeat:
        raise SystemExit("Scorer acceptance failed")


if __name__ == "__main__":
    main()
