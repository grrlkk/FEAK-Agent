"""Phase 2c: 40 pilot requests, 460 nominal total, 470 shared hard ceiling."""

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import fcntl
import json

from verak.v3.common import load_config, read_json, write_json
from verak.v3.phase2 import read_jsonl, write_jsonl
from verak.v3.phase2b import input_hash
from verak.v3.phase2c import assert_preserved
from verak.v3.phase2c_metrics import pilot_projection, summarize
from verak.v3.phase2c_verification import Phase2cAPI, both_ok, prompt_hash
from verak.v3.view_data import view_fingerprint


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--stage", choices=("pilot", "remaining"), required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config()
    api = Phase2cAPI(config, args.max_api_calls)
    path = config["paths"]["metadata"] / "structure_check_phase2c.jsonl"
    with (api.output / "judge.lock").open("a+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        assert_preserved(config)
        rows, prepared = read_jsonl(path), read_json(api.output / "preparation.json")
        if input_hash(rows) != prepared["input_sha256"] or view_fingerprint(config) != prepared["view_fingerprint"]:
            raise ValueError("Frozen sample or structural implementation changed")
        hashes = {r["item_id"]: prompt_hash(r) for r in rows}
        frozen = api.output / "prompt_hashes.json"
        if frozen.exists() and read_json(frozen) != hashes:
            raise ValueError("Frozen prompts changed")
        write_json(frozen, hashes)
        if args.stage == "remaining":
            gate = pilot_projection(rows, read_jsonl(api.output / "calls.jsonl"))
            if not gate["continue_allowed"]:
                raise ValueError("Pilot projected cost is not below $10")
        chosen = [i for i, row in enumerate(rows) if row["pilot"] == (args.stage == "pilot")]
        jobs = [(i, run) for i in chosen for run in (1, 2) if rows[i][f"llm_judgment_{run}"] is None]
        if api.budget.used + len(jobs) > args.max_api_calls:
            raise ValueError("Insufficient remaining API budget for this stage")
        errors = []
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(api.judge, rows[i], run): (i, run) for i, run in jobs}
            for done, future in enumerate(as_completed(futures), 1):
                i, run = futures[future]
                try:
                    rows[i][f"llm_judgment_{run}"] = future.result()
                    rows[i]["llm_ok"] = both_ok(rows[i])
                    write_jsonl(path, rows)
                except Exception as exc:
                    errors.append({"item_id": rows[i]["item_id"], "run": run, "error_type": type(exc).__name__})
                print(f"{args.stage} {done}/{len(jobs)} failures={len(errors)} ledger={api.budget.used}/470", flush=True)
        records = read_jsonl(api.output / "calls.jsonl")
        result = {**summarize(rows, records), "api_calls_used": api.budget.used, "errors_this_run": errors}
        write_json(api.output / "results.json", result)
        if args.stage == "pilot" and not errors:
            gate = pilot_projection(rows, records)
            write_json(api.output / "pilot_usage.json", gate)
            print(json.dumps(gate, ensure_ascii=False, indent=2), flush=True)
            if not gate["continue_allowed"]:
                return 2
        assert_preserved(config)
        print(json.dumps({"api_calls_used": api.budget.used, "usage": result["usage"], "errors": errors}, ensure_ascii=False, indent=2))
        return 1 if errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
