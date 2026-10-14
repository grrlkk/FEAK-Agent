"""Build three Phase 3b candidates per source, without GPT or corrupted scoring."""

import argparse

from ..common import DEFAULT_CONFIG, load_config
from ..corrupt.instance_policy import build_candidates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["agent_train", "agent_dev", "both"], required=True)
    args = parser.parse_args()
    config = load_config(args.config)
    for split in (("agent_train", "agent_dev") if args.split == "both" else (args.split,)):
        build_candidates(config, split)


if __name__ == "__main__":
    main()
