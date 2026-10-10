"""Two independent Sol EC judgments; an explicit pilot cost gate precedes the rest."""

import argparse
from concurrent.futures import as_completed, ThreadPoolExecutor
import fcntl
import json

from ..common import DEFAULT_CONFIG, file_sha, load_config, read_json, sha_text, write_json
from ..ec_metrics import agreement, pilot_projection, token_usage
from ..ec_verification import both_runs_ok, Phase2API, prompt_hash
from ..phase2 import read_jsonl, write_jsonl


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--stage", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-api-calls", type=int, required=True)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    config = load_config(args.config)
    api = Phase2API(config, args.max_api_calls)
    # Avoid two CLI processes replacing the shared EC sample concurrently.
    with (api.output / "judgments.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        prep = read_json(api.output / "preparation.json")
        # The EC tagging audit is independent of the episode-view token budget.
        if prep["split_sha256"] != file_sha(config["paths"]["metadata"] / "splits.json"):
            raise ValueError("Agent split changed since sample preparation")
        path = config["paths"]["metadata"] / "ec_check.jsonl"
        rows = read_jsonl(path)
        base = [{key: value for key, value in row.items() if not key.startswith("llm_") and key != "human_ok"}
                for row in rows]
        if len(rows) != 100 or sha_text(json.dumps(base, ensure_ascii=False, sort_keys=True)) != prep["sample_inputs_sha256"]:
            raise ValueError("EC sample inputs changed")
        for row in rows:
            for run in (1, 2):
                existing = row[f"llm_judgment_{run}"]
                if existing and (existing["prompt_sha256"] != prompt_hash(row) or
                                 existing["model"] != config["ec_judge"]["model"]):
                    raise ValueError("Cannot mix model or prompt versions in independent judgments")
        if args.stage == "remaining":
            gate = pilot_projection(rows[:10], read_jsonl(api.output / "calls.jsonl"))
            write_json(api.output / "pilot_usage.json", gate)
            if not gate["continue_allowed"]:
                print(json.dumps({"status": "STOP_PROJECTED_COST", **gate}, indent=2))
                return 2
        selected = range(10) if args.stage == "pilot" else range(10, 100)
        jobs = [(i, run) for i in selected for run in (1, 2) if rows[i][f"llm_judgment_{run}"] is None]
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(api.judge, rows[i], run): (i, run) for i, run in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                i, run = futures[future]
                try:
                    rows[i][f"llm_judgment_{run}"] = future.result()
                    rows[i]["llm_ok"] = both_runs_ok(rows[i])
                    write_jsonl(path, rows)  # Preserve successful calls across restarts.
                except Exception as exc:
                    errors.append({"sentence_id": rows[i]["sentence_id"], "run": run,
                                   "error_type": type(exc).__name__})
                print(f"EC {args.stage} completed={done}/{len(jobs)}; failures={len(errors)}; "
                      f"ledger={api.budget.used}/220", flush=True)
        records = read_jsonl(api.output / "calls.jsonl")
        report = {"judge_model": config["ec_judge"]["model"], "reasoning_effort": "high",
                  "api_calls_used": api.budget.used, "usage": token_usage(records),
                  "verification": agreement(rows), "errors_this_run": errors,
                  "human_ok_filled": sum(row["human_ok"] is not None for row in rows)}
        write_json(api.output / "ec_results.json", report)
        if args.stage == "pilot" and not errors:
            gate = pilot_projection(rows[:10], records)
            write_json(api.output / "pilot_usage.json", gate)
            print(json.dumps({"pilot": gate}, ensure_ascii=False, indent=2))
            if not gate["continue_allowed"]:
                return 2
        print(json.dumps({"stage": args.stage, "api_calls_used": api.budget.used,
                          "usage": report["usage"], "errors": errors}, ensure_ascii=False, indent=2))
        return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
