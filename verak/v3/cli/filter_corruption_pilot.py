"""Prepare or judge only the frozen Phase 3b 100-essay pilot."""

import argparse
from ..common import DEFAULT_CONFIG, load_config
from ..corrupt.instance_pilot import prepare_pilot, run_pilot


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "judge"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 0 <= args.max_api_calls <= 100 or not 1 <= args.workers <= 8:
        parser.error("Budget must be 0..100; workers 1..8")
    config = load_config(args.config)
    if args.stage == "prepare":
        plan = prepare_pilot(config)
        print({"sample": plan["n"], "strata": len(plan["strata"]), "population": plan["population"]})
    else:
        result = run_pilot(config, args.max_api_calls, args.workers)
        print(result["essay_yield"]["overall"])


if __name__ == "__main__":
    main()
