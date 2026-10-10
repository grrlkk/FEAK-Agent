"""Complete approved Phase 3b judging, balance whole essays, and score the corpus."""

import argparse
from ..common import DEFAULT_CONFIG, load_config


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["judge", "audit", "balance", "score", "exclude-unscorable", "verify"])
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-api-calls", type=int, required=True,
                        help="Cumulative Phase 3b attempts including pilot/retries; maximum 7797")
    parser.add_argument("--max-cost-usd", type=float, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--scorer-workers", type=int, default=1)
    args = parser.parse_args()
    if not 0 <= args.max_api_calls <= 7797 or not 0 < args.max_cost_usd <= 50 or not 1 <= args.workers <= 4:
        parser.error("Authorization limit exceeded")
    config = load_config(args.config)
    if args.stage == "judge":
        from ..corrupt.resilient_judging import run_resilient
        print(run_resilient(config, max_api_calls=args.max_api_calls, max_cost_usd=args.max_cost_usd, workers=args.workers))
    else:
        from ..corrupt.final_corpus import (audit_judgments, balance_corpus, score_corpus,
                                          exclude_unscorable, verify_corpus)
        if args.stage == "audit":
            print(audit_judgments(config))
        elif args.stage == "balance":
            print(balance_corpus(config))
        elif args.stage == "score":
            print(score_corpus(config, workers=args.scorer_workers))
        elif args.stage == "exclude-unscorable":
            print(exclude_unscorable(config))
        else:
            print(verify_corpus(config))


if __name__ == "__main__":
    main()
