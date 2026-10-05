"""Measure compact views and exclude over-budget essays without changing the source."""

import argparse
import json

from ..common import DEFAULT_CONFIG, load_config
from ..view_data import prepare_views


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args(argv)
    if not 1 <= args.workers <= 8:
        parser.error("workers must be between 1 and 8")
    result = prepare_views(load_config(args.config), args.workers)
    print(json.dumps({key: value for key, value in result.items() if key != "review"}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
