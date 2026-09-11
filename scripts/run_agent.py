#!/usr/bin/env python
"""Run the full revision loop with trained Kanana and a general local LLM."""

import argparse
import json
import os
from pathlib import Path
import sys

import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from feak_tc.agent import run_agent
from feak_tc.agent.local_llm import LocalJSONClient
from feak_tc.agent.observer import KananaWorker
from feak_tc.agent.retrieval import ExemplarStore
from feak_tc.agent.roles import LocalRoles, OfflineRoles
from feak_tc.agent.schemas import ControllerConfig, LocalModelConfig
from feak_tc.diagnose import get_diagnoser


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--text")
    source.add_argument("--text-file", type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/agent_local.yaml")
    parser.add_argument("--output", type=Path, required=True, help="New summary JSON; events go to .events.jsonl")
    parser.add_argument("--offline-smoke", action="store_true", help="Explicit stub-only wiring check")
    parser.add_argument("--local-model")
    parser.add_argument("--llm-device")
    parser.add_argument("--kanana-device", type=int)
    parser.add_argument("--kanana-m", type=int)
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--candidates", type=int)
    parser.add_argument("--exemplars", type=Path)
    parser.add_argument("--essay-id")
    parser.add_argument("--source-group")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    for key, value in (("model", args.local_model), ("device", args.llm_device)):
        if value is not None:
            cfg["local_llm"][key] = value
    for key, value in (("device_id", args.kanana_device), ("m", args.kanana_m)):
        if value is not None:
            cfg["kanana"][key] = value
    for key, value in (("max_steps", args.max_steps), ("candidates_per_step", args.candidates)):
        if value is not None:
            cfg["controller"][key] = value
    ControllerConfig.model_validate(cfg["controller"])
    model_cfg = LocalModelConfig.model_validate(cfg["local_llm"])
    if cfg["kanana"]["m"] < 1:
        parser.error("--kanana-m must be positive")
    if cfg["kanana"]["device_id"] < 0 or cfg["diagnosis_timeout_s"] <= 0:
        parser.error("Kanana device must be nonnegative and diagnosis timeout must be positive")
    text = args.text if args.text is not None else args.text_file.read_text(encoding="utf-8")
    if not text.strip() or not args.question.strip():
        parser.error("Essay and question must not be empty")
    retrieval_cfg = dict(cfg["retrieval"])
    if args.exemplars:
        retrieval_cfg["path"] = str(args.exemplars)
    if retrieval_cfg["path"] and not args.source_group:
        parser.error("--source-group is required with exemplars to exclude related source essays")
    cfg["retrieval"] = retrieval_cfg
    exemplars = ExemplarStore(**retrieval_cfg)
    # Feature extraction uses the existing isolated subprocess and local morphology server.
    os.environ.setdefault("FEAK_FEATURE_CUDA_VISIBLE_DEVICES", "")
    os.environ.setdefault("FEAK_EMBEDDING_DEVICE", "cpu")
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    client = None
    similarity_fn = None
    if args.offline_smoke:
        diagnoser = get_diagnoser("stub", weak_top_n=cfg["weak_rubric_top_n"])
        roles = OfflineRoles()
        # Keep the wiring check entirely free of embedding downloads/loads.
        from feak_tc.mvp import transition
        similarity_fn = lambda a, b: (
            transition.source_token_retention(a, b), {"method": "offline_token_retention"}
        )
    else:
        diagnoser = KananaWorker(timeout_s=cfg["diagnosis_timeout_s"], question=args.question,
                                weak_top_n=cfg["weak_rubric_top_n"], config_kwargs=cfg["kanana"])
        client = LocalJSONClient(model_cfg)
        roles = LocalRoles(client, cfg)

    event_path = args.output.with_suffix(".events.jsonl")
    if args.output == event_path or args.output.exists() or event_path.exists():
        parser.error("Output or event log already exists; choose a new --output path")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as summary, event_path.open("x", encoding="utf-8") as log:
        def on_event(row):
            log.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
            log.flush()
            print(f"[{row['event']}] step={row.get('step', '-')} "
                  f"{row.get('reason', row.get('reasons', ''))}", file=sys.stderr, flush=True)

        try:
            result = run_agent(text, question=args.question, diagnoser=diagnoser, roles=roles,
                               cfg=cfg, exemplars=exemplars, essay_id=args.essay_id,
                               source_group=args.source_group, on_event=on_event, similarity_fn=similarity_fn)
        finally:
            if isinstance(diagnoser, KananaWorker):
                diagnoser.close()
        result["runtime"] = {"offline_smoke": args.offline_smoke, "config": cfg,
                             "llm_calls": client.calls if client else 0,
                             "llm_records": client.records if client else []}
        summary.write(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False) + "\n")
    print(result["final_text"])
    print(f"\nstatus={result['status']} stop={result['stop_reason']} "
          f"accepted={result['accepted']} rollbacks={result['rollbacks']} output={args.output}")
    return 0 if result["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
