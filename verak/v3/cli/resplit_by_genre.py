"""Phase 1b question/genre stratification; no train/test reads or API requests."""

import argparse
import json

from ..common import DEFAULT_CONFIG, load_config
from ..data_policy import resplit_by_genre


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args(argv)
    manifest = resplit_by_genre(load_config(args.config))
    print(json.dumps(manifest["counts"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
