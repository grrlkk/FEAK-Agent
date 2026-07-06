#!/usr/bin/env python
"""Run one FEAK-TC MVP revision step."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Optional

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.diagnose import get_diagnoser
from feak_tc.mvp import serializable_one_step


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text", help="Essay text")
    source.add_argument("--text-file", help="Path to essay text file")
    parser.add_argument("--config", default="configs/heuristic.yaml")
    parser.add_argument("--diagnoser", default="stub", choices=["stub", "kanana", "feak_kobert"])
    parser.add_argument("--question", default="다음 글을 평가하세요.")
    parser.add_argument("--keywords", default=None)
    parser.add_argument("--kanana-m", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--proposer-mode", choices=["auto", "llm", "deterministic"], default=None)
    parser.add_argument("--patcher-mode", choices=["auto", "llm", "deterministic"], default=None)
    parser.add_argument("--json", action="store_true", help="Print raw JSON instead of readable summary")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    cfg = _load_yaml(args.config)
    _apply_cli_overrides(cfg, args)
    text = args.text if args.text is not None else Path(args.text_file).read_text(encoding="utf-8")
    diagnoser = _build_diagnoser(args, cfg)
    output = serializable_one_step(text=text, diagnoser=diagnoser, cfg=cfg)
    if args.json:
        print(json.dumps(output, ensure_ascii=False, indent=2))
    else:
        _print_summary(output)
    return 0


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.proposer_mode is not None:
        cfg.setdefault("proposer", {})["mode"] = args.proposer_mode
    if args.patcher_mode is not None:
        cfg.setdefault("patcher", {})["mode"] = args.patcher_mode


def _build_diagnoser(args: argparse.Namespace, cfg: dict[str, Any]):
    weak_top_n = int(cfg.get("weak_rubric_top_n", 3))
    if args.diagnoser == "kanana":
        return get_diagnoser(
            "kanana",
            question=args.question,
            keywords=args.keywords,
            weak_top_n=weak_top_n,
            config_kwargs={
                "m": args.kanana_m,
                "generate_feedback": False,
                "device_id": args.device_id,
                "load_in_4bit": not args.no_4bit,
            },
        )
    return get_diagnoser(args.diagnoser, weak_top_n=weak_top_n)


def _print_summary(output: dict[str, Any]) -> None:
    before = output["before"]
    print("== Diagnosis ==")
    print("weak_rubrics:", ", ".join(before["weak_rubrics"]))
    print("rubrics:", json.dumps(before["rubrics"], ensure_ascii=False))
    print("\n== Candidates ==")
    for idx, result in enumerate(output["results"]):
        cand = result["candidate"]
        trans = result["transition"]
        status = "REJECT" if result["rejected"] else "OK"
        print(
            f"[{idx}] {status} {cand['action_type']}->{cand['target_rubric']} "
            f"score={result['heuristic_score']:.3f} "
            f"gain={trans['target_gain']:.3f} drop={trans['non_target_drop']:.3f} "
            f"edit={trans['edit_ratio']:.3f} preserve={trans['goal_preservation']:.3f}"
        )
        if result["reject_reasons"]:
            print("    reject_reasons:", ", ".join(result["reject_reasons"]))
        print("    instruction:", cand["instruction"])
    print("\n== Decision ==")
    decision = output["decision"]
    print(f"{decision['decision']} chosen={decision['chosen_index']} reason={decision['reason']}")


if __name__ == "__main__":
    raise SystemExit(main())
