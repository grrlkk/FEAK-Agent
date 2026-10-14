"""Score eligible sources on one GPU; zero GPT calls, no train/test access."""

import argparse
import copy
import sqlite3

from ..common import DEFAULT_CONFIG, load_config
from ..corrupt.sources import score_sources


def phase3_scorer(config):
    # Small CPU kernels between single-GPU decoding steps otherwise oversubscribe
    # the host. This changes scheduling only, never model/scoring arithmetic.
    import torch
    torch.set_num_threads(1)
    from ..score.kanana import KananaScorer
    current = copy.deepcopy(config)
    current["paths"]["output"] = config["paths"]["phase3_output"]
    current["paths"]["output"].mkdir(parents=True, exist_ok=True)
    old = config["paths"]["output"] / "score_cache.sqlite"
    new = current["paths"]["output"] / "score_cache.sqlite"
    if old.exists() and not new.exists():
        with sqlite3.connect(f"file:{old}?mode=ro", uri=True) as source, sqlite3.connect(new) as target:
            source.backup(target)
    return KananaScorer(current)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["both", "agent_train", "agent_dev"], default="both")
    args = parser.parse_args()
    config = load_config(args.config)
    scorer = phase3_scorer(config)
    try:
        for split in (["agent_dev", "agent_train"] if args.split == "both" else [args.split]):
            result = score_sources(config, split, scorer)
            print(split, result["accepted"], "/", result["candidates"], flush=True)
    finally:
        scorer.close()


if __name__ == "__main__":
    main()
