"""Read-only reuse of the installed para_agent adapters and native analyses."""
import hashlib
import importlib
from pathlib import Path
import sys

from .observations import Observer


def bootstrap(root):
    sys.dont_write_bytecode = True
    root = Path(root).resolve()
    source = root / "src"
    if not (source / "para_agent").is_dir():
        raise ValueError(f"para_agent source is missing: {source}")
    if str(source) not in sys.path:
        sys.path.insert(0, str(source))
    module = importlib.import_module("para_agent")
    if not Path(module.__file__).resolve().is_relative_to(source):
        raise ValueError("A different para_agent package was already imported")
    return root


def legacy_config(cfg, run_dir):
    root = bootstrap(cfg.legacy_root)
    from para_agent.config import load_config
    value = load_config(root / cfg.legacy_config)
    value.internal_kanana_n = 32
    value.scorer.cache_dir = str(Path(run_dir).resolve() / "cache/scorer")
    if (Path(run_dir) / "scorer_metadata/genres.json").exists():
        value.scorer.metadata_dir = str(Path(run_dir).resolve() / "scorer_metadata")
    elif value.scorer.metadata_dir:
        value.scorer.metadata_dir = str(root / value.scorer.metadata_dir)
    return value


class NativeDiagnostic:
    def __init__(self, cfg, directory):
        self.root = bootstrap(cfg.legacy_root)
        self.cfg = legacy_config(cfg, directory)
        from para_agent.diagnose.service import build_diagnoser
        from para_agent.kb.lookup import KnowledgeBase
        from para_agent.config import project_path
        self.service = build_diagnoser(self.cfg, log_path=Path(directory) / "native_diagnostics.jsonl")
        self.kb = KnowledgeBase(project_path(self.cfg.kb.features), project_path(self.cfg.kb.feature_key_map))
        self.calls, self.feature_calls = 0, 0

    def _related(self, values):
        # The KB remains unchanged. Supply entries for observed native feature keys.
        return [{**entry, "feature_key": key, "observed_value": values[key]} for entry in self.kb.entries
                if (key := entry.get("feature_key", self.kb.key_map.get(entry.get("code", "")))) in values
                and values[key] is not None]

    def __call__(self, text, prompt, essay_id):
        self.calls += 1
        result = self.service.diagnose(text, [], "cum", prompt, use_cache=False, essay_id=essay_id)
        if result.score_metadata.get("n") != 32:
            raise ValueError("The native diagnosis did not use N=32")
        return {"scores": result.rubric_scores, "features": result.features,
                "kb_items": self._related(result.features), "score_metadata": result.score_metadata,
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "mock": False}

    def features(self, text):
        from para_agent.diagnose.service import paragraph_count
        self.feature_calls += 1
        values = self.service.features.extract(text, unit="cum", paragraph_count=paragraph_count(text))
        return {"features": values, "kb_items": self._related(values),
                "text_sha256": hashlib.sha256(text.encode()).hexdigest(), "mock": False}

    def close(self):
        self.service.close()


def native_observer(cfg):
    bootstrap(cfg.legacy_root)
    from para_agent.text import sentence_spans
    from para_agent.anonymization import MARKER, PLACEHOLDER
    return Observer(sentence_spans=sentence_spans, marker_pattern=MARKER, placeholder_pattern=PLACEHOLDER)


def native_anonymizer(cfg):
    bootstrap(cfg.legacy_root)
    from para_agent.anonymization import TurnAnonymizer
    return TurnAnonymizer


def fingerprint_legacy(root):
    root = Path(root)
    files = [p for name in ("src", "configs", "prompts", "docs", "runs") for p in (root / name).rglob("*")
             if p.is_file() and "__pycache__" not in p.parts]
    return {str(p.relative_to(root)): {"size": p.stat().st_size, "mtime_ns": p.stat().st_mtime_ns,
            **({"sha256": hashlib.sha256(p.read_bytes()).hexdigest()} if "runs" not in p.relative_to(root).parts else {})}
            for p in files}
