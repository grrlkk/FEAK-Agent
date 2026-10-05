"""Complete approved Phase 3b judging, balance whole essays, and score the corpus."""

import argparse
from ..common import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["judge", "audit", "balance", "score", "verify"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-api-calls", type=int, required=True,
                        help="Cumulative Phase 3b count, including the 100 pilot calls; maximum 3033")
    parser.add_argument("--max-cost-usd", type=float, default=45)
    parser.add_argument("--workers", type=int, default=12)
    parser.add_argument("--scorer-workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.max_api_calls <= 3033 or not 0 < args.max_cost_usd <= 45 or not 1 <= args.workers <= 16:
        parser.error("Authorization limit exceeded")
    config = load_config(args.config)
    if args.stage == "judge":
        from ..corrupt.full_judging import run_full
        print(run_full(config, max_api_calls=args.max_api_calls, max_cost_usd=args.max_cost_usd, workers=args.workers))
    else:
        from ..corrupt.final_corpus import audit_judgments, balance_corpus, score_corpus, verify_corpus
        if args.stage == "audit":
            print(audit_judgments(config))
        elif args.stage == "balance":
            print(balance_corpus(config))
        elif args.stage == "score":
            print(score_corpus(config, workers=args.scorer_workers))
        else:
            print(verify_corpus(config))


if __name__ == "__main__":
    main()
