#!/usr/bin/env python
"""Run or resume the validation-only TVM Stage-1 grid sequentially."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tvm_stage1.yaml")
    parser.add_argument("--runs-root", required=True)
    parser.add_argument("--models", nargs="+", choices=["qwen", "kanana"])
    parser.add_argument(
        "--feature-variants", nargs="+", choices=["full", "scorer_free"]
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    with Path(args.config).open(encoding="utf-8") as file:
        config = yaml.safe_load(file) or {}
    models = args.models or list(config["models"])
    variants = args.feature_variants or list(config["feature_variants"])
    rates = [float(value) for value in config["training"]["learning_rates"]]
    root = Path(args.runs_root)
    plan = []
    for model in models:
        for variant in variants:
            for rate in rates:
                output = root / model / variant / _rate_name(rate)
                plan.append((model, variant, rate, output))

    for position, (model, variant, rate, output) in enumerate(plan, 1):
        report_path = output / "validation_report.json"
        if report_path.exists():
            report = json.loads(report_path.read_text(encoding="utf-8"))
            if (
                report.get("gate") != "tvm_stage1_validation"
                or str(report.get("model_key")) != model
                or str(report.get("feature_variant")) != variant
                or float(report.get("learning_rate")) != rate
                or bool(report.get("test_split_evaluated"))
            ):
                raise SystemExit(f"invalid completed run: {output}")
            print(f"[{position}/{len(plan)}] skip completed {output}", flush=True)
            continue
        if output.exists() and any(output.iterdir()):
            raise SystemExit(f"refusing incomplete non-empty run directory: {output}")
        command = [
            sys.executable,
            str(PROJECT_ROOT / "scripts" / "train_tvm_stage1.py"),
            "--config",
            args.config,
            "--model-key",
            model,
            "--feature-variant",
            variant,
            "--learning-rate",
            str(rate),
            "--output-dir",
            str(output),
        ]
        print(
            f"[{position}/{len(plan)}] run {model}/{variant} lr={rate:.1e}",
            flush=True,
        )
        if args.dry_run:
            print(" ".join(command), flush=True)
        else:
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
    return 0


def _rate_name(rate: float) -> str:
    return f"lr_{rate:.0e}".replace("e-0", "e-").replace("e+0", "e+")


if __name__ == "__main__":
    raise SystemExit(main())
