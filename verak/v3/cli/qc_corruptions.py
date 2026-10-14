"""Generate a bounded vague cache or judge the fixed sixty-essay corruption sample."""

import argparse
from pathlib import Path
from ..common import DEFAULT_CONFIG, load_config
from ..corrupt.qc import prepare_vague, run_qc


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["vague", "judge"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-api-calls", type=int, required=True)
    parser.add_argument("--pool", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    config = load_config(args.config)
    if args.stage == "vague":
        prepare_vague(config, args.max_api_calls)
    else:
        if args.pool is None:
            parser.error("judge requires --pool")
        run_qc(config, args.pool, args.max_api_calls, args.workers)


if __name__ == "__main__":
    main()
