#!/usr/bin/env python
"""Compare two MVP batch JSONL logs (before/after a controller change).

  python scripts/compare_mvp_logs.py \
      experiments/results/mvp_stage_a_20.jsonl \
      experiments/results/mvp_stage_a_20_v2.jsonl
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from feak_tc.diagnose.constants import ACTION_TYPES


def summarize(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    ok_rows = [row for row in rows if row.get("status") == "ok"]

    decisions = Counter()
    chosen_actions = Counter()
    reject_reasons = Counter()
    action_reject = Counter()
    action_count = Counter()
    chosen_risk = {"gain<=0": 0, "drop>=1": 0, "preserve<0.9": 0, "total": 0}
    rediagnoses = 0

    for row in ok_rows:
        output = row["output"]
        decision = output["decision"]
        decisions[decision["decision"]] += 1
        results = output["results"]
        for result in results:
            action = result["candidate"]["action_type"]
            action_count[action] += 1
            if result["rejected"]:
                action_reject[action] += 1
                for reason in result["reject_reasons"]:
                    reject_reasons[reason] += 1
            if not any(str(r).startswith("validity:") for r in result["reject_reasons"]):
                transition = result["transition"]
                if transition["edit_ratio"] > 0:
                    rediagnoses += 1
        idx = decision.get("chosen_index")
        if decision["decision"] == "accept" and idx is not None:
            chosen = results[idx]
            transition = chosen["transition"]
            chosen_actions[chosen["candidate"]["action_type"]] += 1
            chosen_risk["total"] += 1
            if transition["target_gain"] <= 0:
                chosen_risk["gain<=0"] += 1
            if transition["non_target_drop"] >= 1:
                chosen_risk["drop>=1"] += 1
            if transition["goal_preservation"] < 0.9:
                chosen_risk["preserve<0.9"] += 1

    return {
        "path": str(path),
        "rows": len(rows),
        "ok": len(ok_rows),
        "decisions": dict(decisions),
        "chosen_actions": dict(chosen_actions),
        "chosen_risk": chosen_risk,
        "reject_reasons": dict(reject_reasons),
        "action_reject_rate": {
            action: f"{action_reject[action]}/{action_count[action]}"
            for action in ACTION_TYPES
            if action_count[action]
        },
        "rediagnoses_spent": rediagnoses,
    }


def print_side_by_side(before: dict, after: dict) -> None:
    def line(label: str, b, a) -> None:
        print(f"{label:<28} {str(b):<32} {str(a)}")

    print(f"{'':<28} {'BEFORE':<32} {'AFTER'}")
    line("file", Path(before["path"]).name, Path(after["path"]).name)
    line("rows ok", before["ok"], after["ok"])
    line("decisions", before["decisions"], after["decisions"])
    print()
    print("chosen action distribution")
    for action in ACTION_TYPES:
        b = before["chosen_actions"].get(action, 0)
        a = after["chosen_actions"].get(action, 0)
        if b or a:
            line(f"  {action}", b, a)
    print()
    print("chosen-candidate risk signals")
    for key in ("gain<=0", "drop>=1", "preserve<0.9"):
        b_total = max(1, before["chosen_risk"]["total"])
        a_total = max(1, after["chosen_risk"]["total"])
        b = f"{before['chosen_risk'][key]}/{before['chosen_risk']['total']} ({before['chosen_risk'][key]/b_total:.0%})"
        a = f"{after['chosen_risk'][key]}/{after['chosen_risk']['total']} ({after['chosen_risk'][key]/a_total:.0%})"
        line(f"  {key}", b, a)
    print()
    print("action reject rate")
    for action in ACTION_TYPES:
        b = before["action_reject_rate"].get(action, "-")
        a = after["action_reject_rate"].get(action, "-")
        if b != "-" or a != "-":
            line(f"  {action}", b, a)
    print()
    print("reject reasons")
    for reason in sorted(set(before["reject_reasons"]) | set(after["reject_reasons"])):
        line(f"  {reason}", before["reject_reasons"].get(reason, 0), after["reject_reasons"].get(reason, 0))
    print()
    line("re-diagnoses spent", before["rediagnoses_spent"], after["rediagnoses_spent"])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("before", type=Path)
    parser.add_argument("after", type=Path)
    parser.add_argument("--json", action="store_true", help="Print raw JSON summaries instead of a table.")
    args = parser.parse_args()

    before = summarize(args.before)
    after = summarize(args.after)
    if args.json:
        print(json.dumps({"before": before, "after": after}, ensure_ascii=False, indent=2))
    else:
        print_side_by_side(before, after)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
