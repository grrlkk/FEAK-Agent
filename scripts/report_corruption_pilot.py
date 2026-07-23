#!/usr/bin/env python
"""Pilot report: per-step drop checks, discard rate, memorization check, pairs.

Usage:
  python scripts/report_corruption_pilot.py \
      --chains experiments/results/corruption_pilot_50_chains.jsonl \
      --measured experiments/results/corruption_pilot_50_measured_gpu*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
import statistics
import sys
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.diagnose.constants import RUBRIC_KEYS

# Just above the observed m=10 rerun noise ceiling (|diff| max 0.225 on the
# 5-row sweep; see configs/heuristic_stage_a_soft.yaml target_gain_min note).
DROP_THRESHOLD = 0.3


def rubric_scores(row: dict) -> dict[str, float]:
    rf = row.get("rf_corrected")
    if isinstance(rf, list) and len(rf) == len(RUBRIC_KEYS):
        return dict(zip(RUBRIC_KEYS, map(float, rf)))
    return {k: float(v) for k, v in row["rubrics"].items()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chains", required=True)
    parser.add_argument("--measured", required=True, nargs="+", help="measured jsonl paths or globs")
    parser.add_argument("--summary-out", default=None)
    args = parser.parse_args()

    chains = {}
    with open(args.chains, encoding="utf-8") as f:
        for line in f:
            chain = json.loads(line)
            chains[chain["record_id"]] = chain

    measured: dict[tuple[str, int], dict] = {}
    for pattern in args.measured:
        for path in sorted(glob.glob(pattern)) or [pattern]:
            with open(path, encoding="utf-8") as f:
                for line in f:
                    row = json.loads(line)
                    measured[(row["record_id"], row["state_index"])] = row

    # --- memorization check: kanana x0 vs human grader average -------------
    kanana_x0, grader = [], []
    for rid, chain in chains.items():
        row = measured.get((rid, 0))
        if row and chain.get("grader_avg") is not None:
            kanana_x0.append(statistics.mean(rubric_scores(row).values()))
            grader.append(float(chain["grader_avg"]))
    corr = _pearson(kanana_x0, grader) if len(kanana_x0) >= 3 else float("nan")

    # --- per-step drop / discard -------------------------------------------
    step_stats: dict[str, Counter] = defaultdict(Counter)
    step_drops: dict[str, list[float]] = defaultdict(list)
    verbose_rubric_deltas: dict[str, list[float]] = defaultdict(list)
    valid_steps: dict[str, set[int]] = defaultdict(set)
    seconds = []

    for rid, chain in chains.items():
        for k, step in enumerate(chain["steps"]):
            before, after = measured.get((rid, k)), measured.get((rid, k + 1))
            if not before or not after:
                step_stats[step["operator"]]["unmeasured"] += 1
                continue
            seconds.extend(r.get("seconds", 0) for r in (before, after))
            b, a = rubric_scores(before), rubric_scores(after)
            if step["operator"] == "VERBOSE_REPEAT":
                # Empirical axis discovery + feature sanity (length must grow).
                for key in RUBRIC_KEYS:
                    verbose_rubric_deltas[key].append(b[key] - a[key])
                grew = after["features"].get("word_Cnt", 0) > before["features"].get("word_Cnt", 0)
                ok = grew
                drop = max(b[key] - a[key] for key in RUBRIC_KEYS)
            else:
                drop = max(b[key] - a[key] for key in step["intended_rubrics"])
                ok = drop >= DROP_THRESHOLD
            step_drops[step["operator"]].append(drop)
            step_stats[step["operator"]]["kept" if ok else "discarded"] += 1
            if ok:
                valid_steps[rid].add(k)

    # --- pair extraction (all steps between i and j must be kept) ----------
    pair_count, pairs_by_gap = 0, Counter()
    for rid, chain in chains.items():
        n = len(chain["states"])
        for i in range(n):
            for j in range(i + 1, n):
                if all(k in valid_steps[rid] for k in range(i, j)):
                    pair_count += 1
                    pairs_by_gap[j - i] += 1

    # --- report -------------------------------------------------------------
    statuses = Counter(c["status"] for c in chains.values())
    print(f"chains: {len(chains)} {dict(statuses)}")
    print(f"measured states: {len(measured)}")
    if seconds:
        print(f"diagnose time: mean {statistics.mean(seconds):.1f}s / state")
    print(f"\nmemorization check (n={len(kanana_x0)}): "
          f"kanana x0 mean {statistics.mean(kanana_x0):.2f} (1-9) vs grader {statistics.mean(grader):.2f} (1-5), "
          f"pearson r={corr:.3f}")
    print(f"\nper-operator (drop threshold {DROP_THRESHOLD} on intended rubrics):")
    for op in sorted(step_stats):
        st, drops = step_stats[op], step_drops[op]
        total = st["kept"] + st["discarded"]
        rate = st["discarded"] / total if total else float("nan")
        mean_drop = statistics.mean(drops) if drops else float("nan")
        print(f"  {op:16s} kept {st['kept']:3d} / discarded {st['discarded']:3d} "
              f"(discard {rate:.0%}), mean max-drop {mean_drop:+.2f}")
    if verbose_rubric_deltas:
        top = sorted(verbose_rubric_deltas, key=lambda k: -statistics.mean(verbose_rubric_deltas[k]))[:3]
        print("  VERBOSE_REPEAT empirical top-drop rubrics: "
              + ", ".join(f"{k} {statistics.mean(verbose_rubric_deltas[k]):+.2f}" for k in top))
    print(f"\npreference pairs: {pair_count} (by stage gap: {dict(sorted(pairs_by_gap.items()))})")

    if args.summary_out:
        summary = {
            "chains": dict(statuses),
            "measured_states": len(measured),
            "memorization_pearson": corr,
            "per_operator": {op: dict(st) for op, st in step_stats.items()},
            "pairs": pair_count,
            "pairs_by_gap": dict(pairs_by_gap),
        }
        Path(args.summary_out).write_text(json.dumps(summary, ensure_ascii=False, indent=2))
        print(f"summary -> {args.summary_out}")
    return 0


def _pearson(xs: list[float], ys: list[float]) -> float:
    mx, my = statistics.mean(xs), statistics.mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs) ** 0.5
    vy = sum((y - my) ** 2 for y in ys) ** 0.5
    return cov / (vx * vy) if vx and vy else float("nan")


if __name__ == "__main__":
    raise SystemExit(main())
