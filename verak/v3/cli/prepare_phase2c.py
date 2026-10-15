"""Offline Phase 2c preparation, cached Bareun only; no GPT or held-out test access."""

import argparse
import json

from verak.v3.common import load_config
from verak.v3.phase2c import prepare


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    result = prepare(load_config(), args.workers)
    print(json.dumps({k: v for k, v in result.items() if k not in {"views", "excluded_previous_sample_ids"}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
