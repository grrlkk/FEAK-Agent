#!/usr/bin/env python
"""Replace OFFTOPIC at its original stage and truncate contaminated suffixes."""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.corruption.chain import _generate_step, _merged_spec, _select_generator


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--config", default="configs/corruption.yaml")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    output = Path(args.output)
    if output.exists() and not args.overwrite:
        raise SystemExit(f"{output} exists; pass --overwrite")
    with open(args.config, encoding="utf-8") as file:
        cfg = yaml.safe_load(file)
    with open(args.input, encoding="utf-8") as file:
        chains = [json.loads(line) for line in file if line.strip()]

    results = [
        regenerate_offtopic_prefix(chain, cfg)
        for chain in chains
        if any(step.get("operator") == "INSERT_OFFTOPIC" for step in chain.get("steps", []))
    ]
    failures = [row for row in results if row.get("status") != "ok"]
    with output.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(
        json.dumps(
            {
                "input_chains": len(chains),
                "replacement_chains": len(results),
                "statuses": dict(Counter(row["status"] for row in results)),
                "replacement_stages": dict(
                    Counter(row.get("regenerated_from_stage") for row in results)
                ),
                "output": str(output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 2 if failures else 0


def regenerate_offtopic_prefix(chain: dict, cfg: dict) -> dict:
    steps = list(chain.get("steps") or [])
    states = list(chain.get("states") or [])
    indices = [
        index for index, step in enumerate(steps) if step.get("operator") == "INSERT_OFFTOPIC"
    ]
    if len(indices) != 1:
        raise ValueError(
            f"{chain.get('record_id')} requires exactly one OFFTOPIC step, got {indices}"
        )
    step_index = indices[0]
    operator = "INSERT_OFFTOPIC"
    configured = cfg["operators"][operator]
    spec = _merged_spec(operator, configured)
    modes = list(spec.get("generation_modes", cfg.get("generation", {}).get("modes", [])))
    generator = _select_generator(
        modes,
        record_id=str(chain["record_id"]),
        operator=operator,
        step_idx=step_index,
    )
    rng = random.Random(f"{cfg.get('seed', 0)}:{chain['record_id']}")
    new_step, errors = _generate_step(
        record_id=str(chain["record_id"]),
        operator=operator,
        generator=generator,
        text=states[step_index],
        source_text=states[0],
        question=str(chain.get("question") or "다음 글을 평가하세요."),
        llm_cfg=dict(cfg.get("llm", {})),
        normalization_cfg=dict(cfg.get("normalization", {})),
        validity_cfg=dict(cfg.get("validity", {})),
        spec=spec,
        rng=rng,
    )
    if new_step is None:
        return {
            **chain,
            "status": "failed",
            "failure_errors": errors,
            "regenerated_from_stage": step_index + 1,
        }

    new_text = new_step.pop("new_text")
    new_step["replacement_of_rulev4_offtopic"] = True
    prefix_steps = steps[:step_index]
    prefix_states = states[: step_index + 1]
    prefix_normalizations = list(chain.get("normalizations") or [])[: step_index + 1]
    return {
        **chain,
        "planned_operators": [step["operator"] for step in prefix_steps] + [operator],
        "status": "ok",
        "failure_errors": None,
        "states": prefix_states + [new_text],
        "normalizations": prefix_normalizations + [new_step["normalization"]],
        "steps": prefix_steps + [new_step],
        "regenerated_from_stage": step_index + 1,
        "discarded_downstream_steps": len(steps) - step_index - 1,
        "source_chain_status": chain.get("status"),
    }


if __name__ == "__main__":
    raise SystemExit(main())
