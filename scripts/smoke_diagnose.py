#!/usr/bin/env python
"""Smoke-test a configured diagnoser on one essay."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.diagnose import get_diagnoser


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--diagnoser", default="stub", choices=["stub", "kanana", "feak_kobert"])
    parser.add_argument("--text", default="인권은 인간이 태어날 때부터 가지는 기본적인 권리이다.")
    parser.add_argument("--question", default="인권의 뜻과 특징에 대해 서술하세요")
    parser.add_argument("--keywords", default="인간(사람), 당연, 권리, 존중(침해)")
    parser.add_argument("--kanana-m", type=int, default=3)
    parser.add_argument("--chunk-m", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--max-seq-length", type=int, default=3072)
    parser.add_argument("--device-id", type=int, default=3)
    parser.add_argument("--no-4bit", action="store_true")
    args = parser.parse_args()

    kwargs = {}
    if args.diagnoser == "kanana":
        kwargs = {
            "question": args.question,
            "keywords": args.keywords,
            "config_kwargs": {
                "m": args.kanana_m,
                "chunk_m": args.chunk_m,
                "max_new_tokens": args.max_new_tokens,
                "max_seq_length": args.max_seq_length,
                "device_id": args.device_id,
                "load_in_4bit": not args.no_4bit,
                "generate_feedback": False,
            },
        }
    diagnosis = get_diagnoser(args.diagnoser, **kwargs).diagnose(args.text)
    print(json.dumps(
        {
            "rubrics": diagnosis.rubrics,
            "features_count": len(diagnosis.features),
            "weak_rubrics": diagnosis.weak_rubrics,
            "metadata": diagnosis.metadata,
        },
        ensure_ascii=False,
        indent=2,
    ))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
