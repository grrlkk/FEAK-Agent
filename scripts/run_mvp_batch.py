#!/usr/bin/env python
"""Run FEAK-TC MVP over many essays and write JSONL transition logs."""

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
from feak_tc.mvp.batch import run_batch


def parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="txt/jsonl/json file or directory of txt files")
    parser.add_argument("--output", default="experiments/results/mvp_batch.jsonl")
    parser.add_argument("--config", default="configs/heuristic.yaml")
    parser.add_argument("--diagnoser", default="stub", choices=["stub", "kanana", "feak_kobert"])
    parser.add_argument("--question", default="다음 글을 평가하세요.")
    parser.add_argument("--keywords", default=None)
    parser.add_argument("--kanana-m", type=int, default=3)
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    parser.add_argument("--proposer-mode", choices=["auto", "llm", "deterministic"], default=None)
    parser.add_argument("--patcher-mode", choices=["auto", "llm", "deterministic"], default=None)
    parser.add_argument("--surface-normalizer", choices=["off", "bareun", "hanspell"], default=None)
    parser.add_argument("--n-per-action", type=_positive_int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-chars", type=int, default=0, help="Skip records with essay text shorter than this")
    parser.add_argument("--append", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--fail-fast", action="store_true")
    return parser.parse_args(argv)


def main(argv: Optional[list[str]] = None) -> int:
    args = parse_args(argv)
    output_path = Path(args.output)
    if output_path.exists() and not args.append and not args.overwrite:
        raise SystemExit(f"{output_path} already exists; pass --overwrite or --append.")

    cfg = _load_yaml(args.config)
    _apply_cli_overrides(cfg, args)
    diagnoser = _build_diagnoser(args, cfg)
    summary = run_batch(
        input_path=args.input,
        output_path=output_path,
        diagnoser=diagnoser,
        cfg=cfg,
        limit=args.limit,
        min_chars=args.min_chars,
        append=args.append,
        fail_fast=args.fail_fast,
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


def _load_yaml(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _apply_cli_overrides(cfg: dict[str, Any], args: argparse.Namespace) -> None:
    if args.n_per_action is not None:
        cfg["n_per_action"] = args.n_per_action
    if args.proposer_mode is not None:
        cfg.setdefault("proposer", {})["mode"] = args.proposer_mode
    if args.patcher_mode is not None:
        cfg.setdefault("patcher", {})["mode"] = args.patcher_mode
    if args.surface_normalizer is not None:
        cfg.setdefault("surface_normalizer", {})["mode"] = args.surface_normalizer


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed < 1:
        raise argparse.ArgumentTypeError("must be >= 1")
    return parsed


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


if __name__ == "__main__":
    raise SystemExit(main())
