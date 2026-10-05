"""Create the Phase 2 EC sample and 20 renderings from the current agent_dev."""

import argparse
import json

from ..common import DEFAULT_CONFIG, load_config
from ..phase2 import prepare


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    result = prepare(load_config(args.config))
    print(json.dumps({key: value for key, value in result.items()
                      if key not in {"render_review", "sample_ids"}}, ensure_ascii=False, indent=2))
    return 2 if result["token_budget_probe"]["ask_required"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
