#!/usr/bin/env python
"""Re-measure every corruption chain state with the diagnoser (GPU phase).

Shards by record so several GPUs can run in parallel:
  python scripts/measure_corruption_chains.py --input chains.jsonl \
      --output measured_gpu1.jsonl --shard 0 --num-shards 3 --device-id 1
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.diagnose import get_diagnoser


def build_diagnoser(args: argparse.Namespace):
    if args.diagnoser == "kanana":
        return get_diagnoser(
            "kanana",
            question="다음 글을 평가하세요.",
            config_kwargs={
                "m": args.kanana_m,
                "generate_feedback": False,
                "device_id": args.device_id,
                "load_in_4bit": True,
            },
        )
    return get_diagnoser(args.diagnoser)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="chains jsonl from run_corruption_chains.py")
    parser.add_argument("--output", required=True)
    parser.add_argument("--shard", type=int, default=0)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--kanana-m", type=int, default=10)
    parser.add_argument("--diagnoser", default="kanana", choices=["kanana", "stub"])
    args = parser.parse_args()

    chains = []
    with open(args.input, encoding="utf-8") as f:
        for idx, line in enumerate(f):
            if idx % args.num_shards == args.shard:
                chain = json.loads(line)
                if chain["status"] in ("ok", "partial") and len(chain["states"]) >= 2:
                    chains.append(chain)

    output = Path(args.output)
    done: set[tuple[str, int]] = set()
    if output.exists():
        with output.open() as f:
            for line in f:
                row = json.loads(line)
                done.add((row["record_id"], row["state_index"]))

    diagnoser = build_diagnoser(args)
    total = sum(len(c["states"]) for c in chains)
    measured = len(done)
    started = time.time()
    with output.open("a", encoding="utf-8") as out:
        for chain in chains:
            diagnoser.question = chain.get("question") or "다음 글을 평가하세요."
            for state_index, text in enumerate(chain["states"]):
                if (chain["record_id"], state_index) in done:
                    continue
                t0 = time.time()
                diag = diagnoser.diagnose(text)
                row = {
                    "record_id": chain["record_id"],
                    "state_index": state_index,
                    "rubrics": diag.rubrics,
                    "rf_corrected": diag.metadata.get("rf_corrected_score"),
                    "soft_mean": diag.metadata.get("soft_mean"),
                    "soft_std": diag.metadata.get("soft_std"),
                    "features": diag.features,
                    "seconds": round(time.time() - t0, 1),
                }
                out.write(json.dumps(row, ensure_ascii=False) + "\n")
                out.flush()
                measured += 1
                print(f"[shard {args.shard}] {measured}/{total} "
                      f"{chain['record_id']} x{state_index} ({row['seconds']}s)",
                      flush=True)

    print(f"shard {args.shard} done: {measured}/{total} states, "
          f"{(time.time() - started) / 60:.1f} min",
          flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
