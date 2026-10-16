"""Prepare the frozen three-essay smoke; run only the requested main condition."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

from .config import load_settings
from .events import save_json

PROJECT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["prepare", "live", "report"])
    parser.add_argument("--config", type=Path, default=PROJECT / "configs/staged_feedback/default.yaml")
    parser.add_argument("--run-dir", type=Path)
    parser.add_argument("--from-run", type=Path, help="같은 표본으로 새 시도를 준비하고 기존 $5 원장을 공유합니다.")
    args = parser.parse_args(argv)
    cfg = load_settings(args.config)
    directory = args.run_dir or PROJECT / "runs" / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    if args.command == "report":
        from .report import render
        print(json.dumps(render(directory), ensure_ascii=False, indent=2))
        return 0
    if args.command == "prepare":
        if (directory / "selection.json").exists():
            raise ValueError("Selection already exists; use a fresh directory")
        if args.from_run:
            parent = args.from_run.resolve()
            selection = json.loads((parent / "selection.json").read_text())
            save_json(directory / "selection.json", selection)
            essays = selection["essays"]
            shutil.copy2(parent / "evaluation_exclusions.json", directory / "evaluation_exclusions.json")
            reference = parent / "budget_reference.json"
            ledger = json.loads(reference.read_text())["ledger"] if reference.exists() else str(parent / "budget.json")
            save_json(directory / "budget_reference.json", {"ledger": ledger, "parent_run": str(parent)})
        else:
            from .selection import select_three
            essays = select_three(cfg, directory)
        from .metadata import prepare_metadata
        prepare_metadata(cfg, directory)
        save_json(directory / "settings.json", cfg)
        shutil.copytree(PROJECT / cfg.prompts, directory / "prompts")
        print(json.dumps({"run_dir": str(directory), "selected": [
            {k: e[k] for k in ("essay_id", "stratum", "human_original_total")} for e in essays],
            "budget_usd": cfg.budget_usd, "api_calls": 0}, ensure_ascii=False, indent=2))
        return 0
    if cfg.condition not in {"planned", "routed"}:
        raise ValueError("The requested live smoke executes the main planned condition only; comparisons are implementation-only")
    if (directory / "calls.jsonl").exists():
        raise ValueError("Existing API calls found; do not overwrite or silently rerun an experiment")
    selection = json.loads((directory / "selection.json").read_text(encoding="utf-8"))
    if json.loads((directory / "settings.json").read_text()) != cfg.model_dump():
        raise ValueError("Prepared settings changed; prepare a new run")
    sys.dont_write_bytecode = True
    os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
    import torch
    preflight = {"python": sys.executable, "cuda_available": torch.cuda.is_available(),
                 "cuda_devices": torch.cuda.device_count(), "nvidia_device_paths": [str(p) for p in Path("/dev").glob("nvidia*")]}
    save_json(directory / "preflight.json", preflight)
    if not preflight["cuda_available"]:
        save_json(directory / "preflight.error.json", {"status": "blocked", "component": "native_kanana_gpu",
            "reason": "CUDA GPU를 사용할 수 없어 기존 N32 채점기를 실행하지 못했습니다. API 호출은 0건입니다.", **preflight})
        from .report import render
        render(directory)
        print("GPU를 사용할 수 없습니다. preflight.error.json을 확인하세요. API 호출: 0건", file=sys.stderr)
        return 2
    previous_error = directory / "preflight.error.json"
    if previous_error.exists():
        save_json(previous_error, {**json.loads(previous_error.read_text()), "resolved": True,
            "resolution": "이후 실행에서 CUDA 사용 가능 여부를 확인했습니다."})
    from .adapters import NativeDiagnostic, native_executor, fingerprint_legacy
    from .engine import Engine
    from .llm import LiveLLM
    from .report import render
    old = fingerprint_legacy(cfg.legacy_root)
    save_json(directory / "legacy_before.json", old)
    diagnostic, llm, failures = None, None, 0
    try:
        diagnostic = NativeDiagnostic(cfg, directory)
        executor = native_executor(cfg)
        # Prompts are frozen at prepare time, together with settings and essays.
        reference = directory / "budget_reference.json"
        ledger = Path(json.loads(reference.read_text())["ledger"]) if reference.exists() else None
        llm = LiveLLM(cfg.model_copy(update={"prompts": "prompts"}), directory, directory, budget_ledger=ledger)
        for essay in selection["essays"]:
            item = {key: essay[key] for key in ("essay_id", "prompt", "text")}
            destination = directory / "essays" / essay["essay_id"]
            try:
                result = Engine(cfg, llm=llm, diagnostic=diagnostic, executor=executor, directory=destination).run(item)
                print(f"완료 {essay['essay_id']}: {result['route']}", flush=True)
            except Exception as exc:
                failures += 1
                save_json(destination / "execution.error.json", {"essay_id": essay["essay_id"],
                    "error_type": type(exc).__name__, "error": str(exc)})
                print(f"실행 실패 {essay['essay_id']}: {exc}", file=sys.stderr, flush=True)
    finally:
        if llm:
            llm.close()
        if diagnostic:
            diagnostic.close()
        new = fingerprint_legacy(cfg.legacy_root)
        save_json(directory / "legacy_unchanged.json", {"unchanged": old == new,
            "changed_paths": sorted(k for k in old.keys() | new.keys() if old.get(k) != new.get(k))})
        print(json.dumps(render(directory), ensure_ascii=False, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
