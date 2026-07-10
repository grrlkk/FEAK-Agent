#!/usr/bin/env python
"""Build elite feature bands from high-scoring AI-Hub essays.

Selects the top essays by mean human grader score from a scoring JSONL file,
extracts FEAK features for each, and writes per-feature [low, high] quantile
bands to configs/elite_features.yaml. transition.target_gap_reduction uses
these bands to measure whether an edit moved the target features toward the
elite range.

Examples:
  # Real FEAK features (needs the sibling essay_scoring_llm package):
  python scripts/build_elite_band.py --input data/data_jsonl/train.jsonl --top-n 200

  # Mechanics check without the real extractor (do NOT ship the output):
  python scripts/build_elite_band.py --extractor stub --top-n 20 \
      --output /tmp/elite_features_stub.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import date
from pathlib import Path
from typing import Callable

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.diagnose.constants import FEAK_FEATURE_NAMES, require_feak_features
from feak_tc.mvp.batch import _parse_scoring_user_prompt


def iter_scored_essays(path: Path):
    with path.open(encoding="utf-8") as f:
        for idx, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            parsed = _parse_scoring_user_prompt(obj.get("user"))
            if not parsed:
                continue
            scores = []
            for key in ("grader_1_scores", "grader_2_scores"):
                values = obj.get(key)
                if isinstance(values, list) and values:
                    scores.extend(float(v) for v in values)
            if not scores:
                continue
            yield {
                "record_id": f"{path.stem}_{idx}",
                "text": parsed["essay"],
                "mean_score": sum(scores) / len(scores),
            }


def make_extractor(kind: str, package_path: str) -> Callable[[str], dict[str, float]]:
    if kind == "stub":
        from feak_tc.diagnose import get_diagnoser

        diagnoser = get_diagnoser("stub")
        return lambda text: dict(diagnoser.diagnose(text).features)
    if kind == "feak":
        resolved = Path(package_path).expanduser()
        if str(resolved) not in sys.path:
            sys.path.insert(0, str(resolved))
        from essay_scoring_llm.feak import extract_feak_features

        return lambda text: require_feak_features(extract_feak_features(text))
    raise ValueError(f"Unknown extractor: {kind}")


def quantile(sorted_values: list[float], q: float) -> float:
    if not sorted_values:
        raise ValueError("empty values")
    pos = q * (len(sorted_values) - 1)
    lower = int(pos)
    upper = min(lower + 1, len(sorted_values) - 1)
    frac = pos - lower
    return sorted_values[lower] * (1 - frac) + sorted_values[upper] * frac


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", default="data/data_jsonl/train.jsonl")
    parser.add_argument("--output", default="configs/elite_features.yaml")
    parser.add_argument("--top-n", type=int, default=200, help="Essays with the highest mean human score.")
    parser.add_argument("--min-score", type=float, default=None, help="Optional mean-score floor applied before top-n.")
    parser.add_argument("--min-chars", type=int, default=250)
    parser.add_argument("--extractor", choices=("feak", "stub"), default="feak")
    parser.add_argument("--package-path", default="/home/chanwoo/essay_scoring_llm")
    parser.add_argument("--low-q", type=float, default=0.25)
    parser.add_argument("--high-q", type=float, default=0.75)
    args = parser.parse_args()

    input_path = (PROJECT_ROOT / args.input) if not Path(args.input).is_absolute() else Path(args.input)
    records = [
        rec
        for rec in iter_scored_essays(input_path)
        if len(rec["text"]) >= args.min_chars
        and (args.min_score is None or rec["mean_score"] >= args.min_score)
    ]
    records.sort(key=lambda rec: rec["mean_score"], reverse=True)
    selected = records[: args.top_n]
    if not selected:
        print("No qualifying essays found.", file=sys.stderr)
        return 1
    print(
        f"selected {len(selected)} essays "
        f"(mean_score range {selected[-1]['mean_score']:.2f}..{selected[0]['mean_score']:.2f})"
    )

    extract = make_extractor(args.extractor, args.package_path)
    per_feature: dict[str, list[float]] = {name: [] for name in FEAK_FEATURE_NAMES}
    failures = 0
    for i, rec in enumerate(selected, 1):
        try:
            features = extract(rec["text"])
        except Exception as exc:
            failures += 1
            print(f"[{i}/{len(selected)}] {rec['record_id']} failed: {exc}", file=sys.stderr)
            continue
        for name in FEAK_FEATURE_NAMES:
            if name in features:
                per_feature[name].append(float(features[name]))
        if i % 20 == 0:
            print(f"[{i}/{len(selected)}] features extracted")

    bands = {}
    for name, values in per_feature.items():
        if len(values) < 10:
            continue
        ordered = sorted(values)
        bands[name] = {
            "low": round(quantile(ordered, args.low_q), 6),
            "high": round(quantile(ordered, args.high_q), 6),
            "mean": round(sum(ordered) / len(ordered), 6),
            "n": len(ordered),
        }
    if not bands:
        print("No feature bands computed; aborting without writing.", file=sys.stderr)
        return 1

    payload = {
        "meta": {
            "source": str(input_path),
            "extractor": args.extractor,
            "top_n": len(selected),
            "failures": failures,
            "low_q": args.low_q,
            "high_q": args.high_q,
            "built": date.today().isoformat(),
        },
        "features": bands,
    }
    output_path = (PROJECT_ROOT / args.output) if not Path(args.output).is_absolute() else Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=True)
    print(f"wrote {len(bands)} feature bands -> {output_path}")
    if args.extractor == "stub":
        print("WARNING: stub features are synthetic; do not use this output for real runs.", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
