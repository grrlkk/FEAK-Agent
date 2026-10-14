#!/usr/bin/env python
"""Report whether this machine can run the real Kanana-based agent.

This does not download or fix anything by itself. It only reports what is
missing so real model verification is never silently skipped in favor of a
stub. Exit code is 0 only if every check needed for a real Kanana run passes.
"""

import argparse
import json
import os
from pathlib import Path
import socket
import sys
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))


def check_openai():
    from feak_tc.runtime.openai import load_api_environment
    load_api_environment()
    try:
        import openai
        version = openai.__version__
    except ImportError:
        version = None
    key_present = bool(os.getenv("OPENAI_API_KEY"))
    return bool(version) and key_present, {"sdk_version": version, "key_present": key_present,
                                         "api_access": "not checked (run a real smoke)"}


def check_python_cuda():
    info = {"python": sys.version.split()[0]}
    try:
        import torch
        info["torch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        info["cuda_device_count"] = torch.cuda.device_count() if info["cuda_available"] else 0
    except ModuleNotFoundError:
        info["torch"] = None
        info["cuda_available"] = False
        info["cuda_device_count"] = 0
    ok = bool(info["torch"]) and info["cuda_available"] and info["cuda_device_count"] > 0
    return ok, info


def check_essay_scoring_llm():
    package_path = os.getenv("ESSAY_SCORING_LLM_PATH")
    if package_path:
        sys.path.insert(0, str(Path(package_path).expanduser()))
    try:
        import essay_scoring_llm  # noqa: F401
        from essay_scoring_llm.config import ScoringConfig  # noqa: F401
        from essay_scoring_llm.feak import extract_feak_features  # noqa: F401
        location = Path(essay_scoring_llm.__file__).resolve().parent
        return True, {"importable": True, "location": str(location)}
    except ModuleNotFoundError as exc:
        return False, {"importable": False, "error": str(exc), "ESSAY_SCORING_LLM_PATH": package_path}


def check_adapter_and_correction():
    package_path = os.getenv("ESSAY_SCORING_LLM_PATH")
    if package_path:
        sys.path.insert(0, str(Path(package_path).expanduser()))
    try:
        from essay_scoring_llm.config import ScoringConfig
    except ModuleNotFoundError as exc:
        return False, {"error": str(exc)}
    cfg = ScoringConfig()
    adapter = Path(cfg.lora_adapter_path)
    correction = Path(cfg.correction_models)
    adapter_files = list(adapter.glob("*")) if adapter.is_dir() else []
    adapter_ok = any(f.suffix == ".safetensors" for f in adapter_files)
    correction_ok = correction.exists()
    return adapter_ok and correction_ok, {
        "adapter_path": str(adapter), "adapter_ok": adapter_ok,
        "correction_path": str(correction), "correction_ok": correction_ok,
    }


def check_model_cache(model_id):
    cache = Path(os.getenv("HF_HOME", Path.home() / ".cache/huggingface")) / "hub"
    name = "models--" + model_id.replace("/", "--")
    path = cache / name
    return path.is_dir(), {"model": model_id, "cache_dir": str(path), "present": path.is_dir()}


def check_bareun():
    try:
        with socket.create_connection(("127.0.0.1", 5656), timeout=2):
            return True, {"host": "127.0.0.1:5656", "reachable": True}
    except OSError as exc:
        return False, {"host": "127.0.0.1:5656", "reachable": False, "error": str(exc)}


def check_utagger():
    try:
        import pyutagger  # noqa: F401
    except ModuleNotFoundError as exc:
        return False, {"pyutagger_importable": False, "error": str(exc)}
    config_path = Path.home() / "pyutagger_path.json"
    if not config_path.exists():
        return False, {"pyutagger_importable": True, "config_found": False, "config_path": str(config_path)}
    try:
        config = json.loads(config_path.read_text(encoding="utf-8"))
    except ValueError as exc:
        return False, {"pyutagger_importable": True, "config_found": True, "error": str(exc)}
    install_ok = all(Path(v).exists() for v in config.values())
    return install_ok, {"pyutagger_importable": True, "config_found": True,
                        "config": config, "install_paths_exist": install_ok}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--json", action="store_true", help="machine-readable output")
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "configs/pilot_gpt.yaml")
    args = parser.parse_args(argv)
    cfg = yaml.safe_load(args.config.read_text(encoding="utf-8"))

    checks = {
        "python_cuda": check_python_cuda(),
        "essay_scoring_llm_import": check_essay_scoring_llm(),
        "adapter_and_correction_files": check_adapter_and_correction(),
        "kanana_model_cache": check_model_cache("kakaocorp/kanana-1.5-8b-instruct-2505"),
        "bge_m3_cache": check_model_cache("BAAI/bge-m3"),
        "bareun_localhost_5656": check_bareun(),
        "utagger": check_utagger(),
    }
    if "openai" in cfg:
        checks["openai_api"] = check_openai()

    if args.json:
        print(json.dumps({name: {"ok": ok, **detail} for name, (ok, detail) in checks.items()},
                          ensure_ascii=False, indent=2))
    else:
        for name, (ok, detail) in checks.items():
            mark = "OK  " if ok else "MISS"
            print(f"[{mark}] {name}: {detail}")

    all_ok = all(ok for ok, _ in checks.values())
    print(f"\n{'All checks passed.' if all_ok else 'Some checks failed; see above for exact missing items.'}",
          file=sys.stderr)
    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
