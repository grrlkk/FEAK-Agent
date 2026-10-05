"""Build source-filtered corruption datasets; GPT calls require an explicit budget."""

import argparse
from pathlib import Path

from ..common import DEFAULT_CONFIG, load_config, read_json, write_json
from ..corrupt.builder import build_dataset
from ..corrupt.operators import LEVELS
from ..phase2 import read_jsonl, write_jsonl


def score_episodes(config, path):
    from .prepare_corruption_sources import phase3_scorer
    from ..score.kanana import InputTooLong, ScoreParseError
    scorer = phase3_scorer(config)
    rows = read_jsonl(path)
    failures = []
    try:
        for i, row in enumerate(rows, 1):
            if row["q_corrupted"] is not None:
                continue
            try:
                value = scorer.score(row["question"], row["corrupted_text"]).to_dict()
                row.update(q_corrupted=value["mean"], corrupted_score=value)
            except (InputTooLong, ScoreParseError) as error:
                row["score_error"] = type(error).__name__
                failures.append({"episode_id": row["episode_id"], "error": type(error).__name__})
            if i % 20 == 0 or i == len(rows):
                print(f"Corrupted scores {path.name} {i}/{len(rows)}", flush=True)
                write_jsonl(path, rows)
    finally:
        scorer.close()
    write_json(path.with_suffix(".score_failures.json"), failures)
    if failures:
        raise ValueError("Corrupted scorer failures require explicit review; no silent partial dataset")
    stats = read_json(path.with_suffix(".stats.json"))
    stats["scored"] = True
    write_json(path.with_suffix(".stats.json"), stats)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--split", choices=["agent_train", "agent_dev"], required=True)
    parser.add_argument("--levels", default="L1,L2,L3,L4")
    parser.add_argument("--per-essay", type=int, default=2)
    parser.add_argument("--seed", type=int, default=13)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-api-calls", type=int, required=True,
                        help="Shared Phase 3 ceiling (0..100); builder itself uses only cached vague variants")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument("--defer-scoring", action="store_true", help="Write an explicitly unscored QC pool")
    parser.add_argument("--pre-qc", action="store_true", help="Build frozen QC candidates before the operator gate")
    parser.add_argument("--allow-unbalanced", action="store_true",
                        help="Explicitly authorized limited data only; preserve missing-level status")
    args = parser.parse_args()
    config = load_config(args.config)
    from ..api import PhaseBudget
    PhaseBudget(config["paths"]["phase3_output"] / "api_budget.json", args.max_api_calls,
                authorized_ceiling=100, phase="v3_phase3_corruption")
    if args.score_only:
        score_episodes(config, args.out)
        return
    disabled = []
    if not args.pre_qc:
        gate = read_json(config["paths"]["phase3_output"] / "qc_results.json")
        if gate["essays"] != 60 or not gate["complete"]:
            raise ValueError("Final build requires complete 60-essay QC")
        disabled = gate["disabled_operators"]
        missing = sorted({"WORD", "SENTENCE", "TEXT"} - {v for k, v in LEVELS.items() if k not in disabled})
        if missing and not args.allow_unbalanced:
            raise ValueError("QC disabled every local operator for " + ", ".join(missing) +
                "; cannot satisfy equal-level sampling. Final build requires a user decision.")
    build_dataset(config, args.split, args.out, seed=args.seed, per_essay=args.per_essay,
                  levels=tuple(args.levels.split(",")), disabled=disabled)
    stats = read_json(args.out.with_suffix(".stats.json"))
    stats.update(qc_status="candidate_only" if args.pre_qc else "operator_filtered",
                 limited_unbalanced=bool(args.allow_unbalanced),
                 balanced_final_dataset_ready=not args.pre_qc and not args.allow_unbalanced)
    write_json(args.out.with_suffix(".stats.json"), stats)
    if not args.defer_scoring:
        score_episodes(config, args.out)


if __name__ == "__main__":
    main()
