#!/usr/bin/env python
"""Run 1-10 essays through the four-criterion RV pilot and save every attempt."""

import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.agent import run_pilot
from feak_tc.agent.offline import OfflineRoles, OfflineScorer
from feak_tc.agent.roles import PROMPT_VERSION, Roles
from feak_tc.agent.schemas import PilotConfig, Sample
from feak_tc.agent.scorer import StateScorer
from feak_tc.runtime.kanana import KananaWorker
from feak_tc.runtime.openai import OpenAIJSONClient


def write_jsonl(handle, row):
    handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    handle.flush()


def load_samples(args):
    if args.input:
        samples = [Sample.model_validate(json.loads(line)) for line in
                   args.input.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
        samples = [Sample(sample_id=args.sample_id, writing_prompt=args.question or "", text=text)]
    if not samples:
        raise ValueError("Input must contain at least one sample")
    if len({row.sample_id for row in samples}) != len(samples):
        raise ValueError("sample_id must be unique")
    return samples[:args.limit]


def code_identity():
    files = [Path(__file__).resolve(), *sorted((PROJECT_ROOT / "feak_tc/agent").glob("*.py")),
             *sorted((PROJECT_ROOT / "feak_tc/runtime").glob("*.py"))]
    return {str(p.relative_to(PROJECT_ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}


def build_report(results, records, offline):
    attempts = [row for result in results for row in result["attempts"]]
    compared = [row for row in attempts if row.get("score_baseline")]
    usage = {key: sum((row.get("usage") or {}).get(key, 0) for row in records)
             for key in ("input_tokens", "output_tokens", "total_tokens")}
    return {"mode": "offline_smoke" if offline else "real", "samples": len(results),
            "completed": sum(row["status"] == "completed" for row in results),
            "accepted": sum(row["accepted"] for row in results), "candidate_attempts": len(attempts),
            "rejected": sum(row.get("acceptance_decision") == "REJECT" for row in attempts),
            "revision_retries": sum(row["attempt"] > 0 for row in attempts),
            "reverifications": sum(max(0, len(row["rv_checks"]) - 1) for row in attempts),
            "aggregate_score_up_but_rv_rejected": sum(
                row["score_baseline"]["aggregate_accept"] and row["acceptance_decision"] == "REJECT"
                for row in compared),
            "api_calls": len(records), "usage": usage,
            "note": "Execution and automatic judgments only; not a human quality benchmark."}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--input", type=Path, help="JSONL: sample_id, writing_prompt, text")
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--question", help="Required with --text/--text-file")
    parser.add_argument("--sample-id", default="sample_1")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pilot_gpt.yaml")
    parser.add_argument("--output-dir", type=Path, required=True, help="Must be a new directory")
    parser.add_argument("--limit", type=int, default=5, choices=range(1, 11))
    parser.add_argument("--max-iterations", type=int, choices=range(1, 4))
    parser.add_argument("--kanana-device", type=int)
    parser.add_argument("--kanana-m", type=int)
    parser.add_argument("--offline-smoke", action="store_true", help="No models, API calls or quality judgments")
    args = parser.parse_args(argv)
    try:
        values = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if args.max_iterations is not None:
            values.setdefault("controller", {})["max_iterations"] = args.max_iterations
        if args.kanana_device is not None:
            values.setdefault("kanana", {})["device_id"] = args.kanana_device
        if args.kanana_m is not None:
            values.setdefault("kanana", {})["m"] = args.kanana_m
        cfg = PilotConfig.model_validate(values)
        samples = load_samples(args)
        args.output_dir.mkdir(parents=True, exist_ok=False)
    except (ValueError, OSError, yaml.YAMLError) as exc:
        parser.error(str(exc))

    os.environ.setdefault("FEAK_FEATURE_CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("FEAK_EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    manifest = {"started_at": datetime.now(timezone.utc).isoformat(), "config": cfg.model_dump(),
                "mode": "offline_smoke" if args.offline_smoke else "real",
                "prompt_version": PROMPT_VERSION, "code_sha256": code_identity(),
                "git_commit": subprocess.check_output(["git", "rev-parse", "HEAD"],
                                                       cwd=PROJECT_ROOT, text=True).strip()}
    (args.output_dir / "config.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    results = []
    client = None
    with ExitStack() as stack:
        handles = {name: stack.enter_context((args.output_dir / f"{name}.jsonl").open("x", encoding="utf-8"))
                   for name in ("inputs", "trajectories", "events", "llm_calls", "summaries")}
        for sample in samples:
            write_jsonl(handles["inputs"], sample.model_dump())
        if not args.offline_smoke:
            client = OpenAIJSONClient(cfg.openai, lambda row: write_jsonl(handles["llm_calls"], row))
            stack.callback(client.close)
        for sample in samples:
            def on_event(row):
                write_jsonl(handles["events"], row)
                if row["event"] in {"phase", "decision", "stop", "error"}:
                    print(f"[{sample.sample_id}] {row['event']} {row.get('stage', row.get('reason', ''))}",
                          file=sys.stderr, flush=True)
            with ExitStack() as sample_stack:
                if args.offline_smoke:
                    scorer, roles = OfflineScorer(), OfflineRoles()
                else:
                    client.start_sample(sample.sample_id)
                    worker = KananaWorker(timeout_s=cfg.diagnosis_timeout_s,
                                          question=sample.writing_prompt, config_kwargs=cfg.kanana.model_dump())
                    sample_stack.callback(worker.close)
                    scorer = StateScorer(worker, cfg.score_source)
                    roles = Roles(client, cfg.controller.schema_retries)
                result = run_pilot(sample, scorer=scorer, roles=roles, config=cfg.controller,
                                   on_attempt=lambda row: write_jsonl(handles["trajectories"], row),
                                   on_event=on_event)
                write_jsonl(handles["summaries"], result)
                results.append(result)
            if result["status"] == "error":
                break
        report = build_report(results, client.records if client else [], args.offline_smoke)
        (args.output_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0 if all(row["status"] == "completed" for row in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
