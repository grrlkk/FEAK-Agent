"""Resolve the Phase 2 judge, counting GET /v1/models in the shared EC ledger."""

import argparse
import json

from ..common import DEFAULT_CONFIG, load_config
from ..ec_verification import Phase2API


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--max-api-calls", type=int, required=True)
    args = parser.parse_args(argv)
    api = Phase2API(load_config(args.config), args.max_api_calls)
    result = api.discover()
    print(json.dumps({"selected_model": result["selected_model"],
                      "available_flagship_models": result["available_flagship_models"],
                      "api_calls_used": api.budget.used,
                      "status": "ready" if result["selected_model"] else "STOP_NO_AUTHORIZED_SOL"},
                     ensure_ascii=False, indent=2))
    return 0 if result["selected_model"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
