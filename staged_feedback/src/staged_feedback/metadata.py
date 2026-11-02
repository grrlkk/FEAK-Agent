"""Derive missing genres with the unchanged heading rules, in the new run only."""
from collections import Counter, defaultdict
import importlib
import json
from pathlib import Path
import sys

from .adapters import bootstrap
from .events import save_json


def prepare_metadata(cfg, directory):
    root = bootstrap(cfg.legacy_root)
    from para_agent.config import load_config
    from para_agent.artifacts import file_hash
    from para_agent.data.loader import iter_source, source_question_essay, text_hash
    from para_agent.diagnose.scorer_adapter import load_scorer_settings
    from para_agent.diagnose.scorer_metadata import merge_heading_evidence, validate_metadata_override
    legacy = load_config(root / cfg.legacy_config)
    settings = load_scorer_settings(legacy.scorer.model_copy(update={"metadata_dir": None}))
    base_path = settings["paths"]["metadata"] / "genres.json"
    directory = Path(directory)
    selection = json.loads((directory / "selection.json").read_text())
    audit = selection["source_audit"]
    source = Path(audit["source_paths"]["test_jsonl"])
    if file_hash(source) != audit["source_sha256"]["test_jsonl"]:
        raise ValueError("선정 후 원본 데이터 파일이 바뀌었습니다.")
    targets = {text_hash(e["prompt"]) for e in selection["essays"]}
    scorer_root = str(Path(legacy.scorer.repository_path).resolve())
    if scorer_root not in sys.path:
        sys.path.insert(0, scorer_root)
    policy = importlib.import_module("verak.v3.data_policy")
    if not Path(policy.__file__).resolve().is_relative_to(scorer_root):
        raise ValueError("기존 장르 규칙을 지정한 저장소에서 읽지 못했습니다.")
    observations = defaultdict(Counter)
    for _, row in iter_source(source):
        prompt, _ = source_question_essay(row)
        key = text_hash(prompt)
        if key in targets:
            observations[key][policy.feedback_headings(row["assistant"])] += 1
    if set(observations) != targets:
        raise ValueError("선정 문항의 기존 루브릭 이름을 확인할 수 없습니다.")
    metadata, added = merge_heading_evidence(json.loads(base_path.read_text()), policy.HEADING_SETS, observations)
    metadata["provenance"] = {"base_metadata_sha256": file_hash(base_path),
        "source_sha256": file_hash(source), "heading_rules_sha256": file_hash(Path(policy.__file__)),
        "method": "unchanged_native_heading_rules", "llm_api_calls": 0}
    output = directory / "scorer_metadata/genres.json"
    save_json(output, metadata)
    validate_metadata_override(base_path, output)
    report = {"added_questions": added, "coverage": [{"essay_id": e["essay_id"],
        "genre": metadata["questions"][text_hash(e["prompt"])]["genre"]} for e in selection["essays"]],
        "genres_sha256": file_hash(output), "provenance": metadata["provenance"]}
    save_json(directory / "scorer_metadata/provenance.json", report)
    return report
