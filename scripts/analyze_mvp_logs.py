#!/usr/bin/env python
"""Summarize FEAK-TC MVP JSONL transition logs."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Optional


ACTION_ORDER = [
    "ADD_DETAIL",
    "DELETE_OR_FOCUS",
    "COMPRESS",
    "RESTRUCTURE",
    "STYLE_REFINE",
    "STOP",
]
NUMERIC_TRANSITION_KEYS = [
    "heuristic_score",
    "target_gain",
    "non_target_drop",
    "target_gap_reduction",
    "evidence_match",
    "edit_ratio",
    "goal_preservation",
    "emb_sim",
]


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", help="MVP JSONL log file")
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON summary")
    parser.add_argument("--top-examples", type=_non_negative_int, default=5)
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    summary = analyze_log(Path(args.input), top_examples=args.top_examples)
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print_human_summary(summary)
    return 0


def analyze_log(path: Path, top_examples: int = 5) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)

    rows = {"total": 0, "ok": 0, "error": 0}
    decisions: Counter[str] = Counter()
    chosen_actions: Counter[str] = Counter()
    candidate_sources: Counter[str] = Counter()
    patchers: Counter[str] = Counter()
    input_chars: list[float] = []
    action_acc = defaultdict(_new_action_acc)
    error_examples: list[dict[str, Any]] = []
    suspicious_chosen: list[dict[str, Any]] = []

    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            rows["total"] += 1
            row = json.loads(line)
            status = str(row.get("status", ""))
            if status != "ok":
                rows["error"] += 1
                if len(error_examples) < top_examples:
                    error_examples.append(
                        {
                            "line": line_no,
                            "record_id": row.get("record_id"),
                            "error": row.get("error"),
                        }
                    )
                continue

            rows["ok"] += 1
            input_text = str(row.get("input", {}).get("text", ""))
            input_chars.append(float(len(input_text)))
            output = row.get("output", {})
            decision = output.get("decision", {})
            decision_name = str(decision.get("decision", "unknown"))
            decisions[decision_name] += 1
            chosen_index = decision.get("chosen_index")
            results = output.get("results", [])

            for idx, result in enumerate(results):
                candidate = result.get("candidate", {})
                transition = result.get("transition", {})
                action_type = str(candidate.get("action_type", "unknown"))
                acc = action_acc[action_type]
                acc["count"] += 1
                acc["rejected"] += int(bool(result.get("rejected")))
                reject_reasons = result.get("reject_reasons", [])
                if "no_effect" in reject_reasons:
                    acc["no_effect"] += 1
                for reason in reject_reasons:
                    acc["reject_reasons"][str(reason)] += 1

                metadata = candidate.get("metadata", {})
                source = str(metadata.get("source", "unknown"))
                patcher = str(metadata.get("patcher", "unknown"))
                candidate_sources[source] += 1
                patchers[patcher] += 1
                acc["sources"][source] += 1
                acc["patchers"][patcher] += 1

                values = {
                    "heuristic_score": _float_or_none(result.get("heuristic_score")),
                    **{key: _float_or_none(transition.get(key)) for key in NUMERIC_TRANSITION_KEYS if key != "heuristic_score"},
                }
                for key, value in values.items():
                    if value is not None:
                        acc["values"][key].append(value)

                if idx == chosen_index:
                    acc["chosen"] += 1
                    chosen_actions[action_type] += 1
                    maybe_suspicious = _chosen_warning(row, idx, result)
                    if maybe_suspicious and len(suspicious_chosen) < top_examples:
                        suspicious_chosen.append(maybe_suspicious)

    actions = {}
    for action_type in sorted(action_acc, key=_action_sort_key):
        actions[action_type] = _finalize_action(action_acc[action_type])

    return {
        "path": str(path),
        "rows": rows,
        "input_chars": _stats(input_chars),
        "decisions": dict(decisions),
        "chosen_actions": dict(chosen_actions),
        "candidate_sources": dict(candidate_sources),
        "patchers": dict(patchers),
        "actions": actions,
        "suspicious_chosen": suspicious_chosen,
        "error_examples": error_examples,
    }


def print_human_summary(summary: dict[str, Any]) -> None:
    rows = summary["rows"]
    print(f"Log: {summary['path']}")
    print(f"Rows: total={rows['total']} ok={rows['ok']} error={rows['error']}")
    chars = summary["input_chars"]
    if chars["count"]:
        print(
            "Input chars: "
            f"mean={chars['mean']:.1f} min={chars['min']:.0f} max={chars['max']:.0f}"
        )
    print()

    print("Decisions")
    _print_counter(summary["decisions"])
    print()

    print("Chosen actions")
    _print_counter(summary["chosen_actions"], order=ACTION_ORDER)
    print()

    print("Candidate sources")
    _print_counter(summary["candidate_sources"])
    print()

    print("Patchers")
    _print_counter(summary["patchers"])
    print()

    print("Action metrics")
    header = (
        "action             count chosen reject% no_effect% "
        "score_mean gain_mean drop_mean edit_mean preserve_mean"
    )
    print(header)
    print("-" * len(header))
    for action_type in sorted(summary["actions"], key=_action_sort_key):
        item = summary["actions"][action_type]
        print(
            f"{action_type:<17} "
            f"{item['count']:>5} "
            f"{item['chosen']:>6} "
            f"{item['reject_rate'] * 100:>6.1f} "
            f"{item['no_effect_rate'] * 100:>10.1f} "
            f"{_fmt_stat_mean(item, 'heuristic_score'):>10} "
            f"{_fmt_stat_mean(item, 'target_gain'):>9} "
            f"{_fmt_stat_mean(item, 'non_target_drop'):>9} "
            f"{_fmt_stat_mean(item, 'edit_ratio'):>9} "
            f"{_fmt_stat_mean(item, 'goal_preservation'):>13}"
        )
    print()

    if summary["suspicious_chosen"]:
        print("Chosen warnings")
        for item in summary["suspicious_chosen"]:
            print(
                f"- {item['record_id']} [{item['index']}] {item['action_type']} "
                f"gain={item['target_gain']:.3f} drop={item['non_target_drop']:.3f} "
                f"edit={item['edit_ratio']:.3f}: {item['warning']}"
            )
            if item.get("target_span"):
                print(f"  span: {item['target_span']}")
            if item.get("after"):
                print(f"  after: {item['after']}")
        print()

    if summary["error_examples"]:
        print("Error examples")
        for item in summary["error_examples"]:
            print(f"- line={item['line']} record_id={item['record_id']} error={item['error']}")


def _new_action_acc() -> dict[str, Any]:
    return {
        "count": 0,
        "chosen": 0,
        "rejected": 0,
        "no_effect": 0,
        "reject_reasons": Counter(),
        "sources": Counter(),
        "patchers": Counter(),
        "values": defaultdict(list),
    }


def _finalize_action(acc: dict[str, Any]) -> dict[str, Any]:
    count = int(acc["count"])
    return {
        "count": count,
        "chosen": int(acc["chosen"]),
        "rejected": int(acc["rejected"]),
        "no_effect": int(acc["no_effect"]),
        "reject_rate": _rate(acc["rejected"], count),
        "no_effect_rate": _rate(acc["no_effect"], count),
        "chosen_rate": _rate(acc["chosen"], count),
        "reject_reasons": dict(acc["reject_reasons"]),
        "sources": dict(acc["sources"]),
        "patchers": dict(acc["patchers"]),
        "stats": {key: _stats(values) for key, values in acc["values"].items()},
    }


def _chosen_warning(row: dict[str, Any], index: int, result: dict[str, Any]) -> Optional[dict[str, Any]]:
    transition = result.get("transition", {})
    candidate = result.get("candidate", {})
    patch = candidate.get("patch") or {}
    target_gain = float(transition.get("target_gain", 0.0))
    non_target_drop = float(transition.get("non_target_drop", 0.0))
    edit_ratio = float(transition.get("edit_ratio", 0.0))

    warnings = []
    if target_gain <= 0:
        warnings.append("chosen candidate has non-positive target_gain")
    if non_target_drop >= 1:
        warnings.append("chosen candidate has non-target drop")
    if candidate.get("action_type") == "DELETE_OR_FOCUS" and float(transition.get("goal_preservation", 1.0)) < 0.9:
        warnings.append("DELETE_OR_FOCUS chosen with low goal_preservation")
    if not warnings:
        return None

    return {
        "record_id": row.get("record_id"),
        "index": index,
        "action_type": candidate.get("action_type"),
        "target_rubric": candidate.get("target_rubric"),
        "target_gain": target_gain,
        "non_target_drop": non_target_drop,
        "edit_ratio": edit_ratio,
        "warning": "; ".join(warnings),
        "target_span": _shorten(str(candidate.get("target_span", ""))),
        "after": _shorten(str(patch.get("after", ""))),
    }


def _stats(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "mean": None, "min": None, "max": None}
    return {
        "count": len(values),
        "mean": sum(values) / len(values),
        "min": min(values),
        "max": max(values),
    }


def _rate(part: int, total: int) -> float:
    return float(part) / total if total else 0.0


def _float_or_none(value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _print_counter(counter: dict[str, int], order: Optional[list[str]] = None) -> None:
    if not counter:
        print("  (none)")
        return
    keys = list(order or [])
    keys.extend(key for key in sorted(counter) if key not in keys)
    for key in keys:
        if key in counter:
            print(f"  {key}: {counter[key]}")


def _fmt_stat_mean(item: dict[str, Any], key: str) -> str:
    stat = item["stats"].get(key, {})
    mean = stat.get("mean")
    return "n/a" if mean is None else f"{mean:.3f}"


def _action_sort_key(action_type: str) -> tuple[int, str]:
    try:
        return (ACTION_ORDER.index(action_type), action_type)
    except ValueError:
        return (len(ACTION_ORDER), action_type)


def _shorten(text: str, limit: int = 120) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be >= 0")
    return parsed


if __name__ == "__main__":
    raise SystemExit(main())
