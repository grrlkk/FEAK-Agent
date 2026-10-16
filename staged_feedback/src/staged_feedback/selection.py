"""Freeze three held-out originals without changing any legacy exclusions."""
import json
from pathlib import Path
import re

import yaml

from .adapters import bootstrap, legacy_config
from .events import save_json


def select_three(cfg, directory):
    root = bootstrap(cfg.legacy_root)
    from para_agent.data.loader import load_heldout_candidates, stratified_sample, text_hash
    from para_agent.artifacts import file_hash
    legacy = legacy_config(cfg, directory)
    keys = yaml.safe_load((root / legacy.paths.contracts).read_text())["rubric_keys"]
    candidates, audit, source_excluded = load_heldout_candidates(
        legacy.data.source, keys, min_paras=2, max_paras=10000)
    methodology = (root / "docs/METHODOLOGY.md").read_text(encoding="utf-8")
    section = methodology.split("## 13. 데이터 분리", 1)[1].split("## 14.", 1)[0]
    paths = [root / value for value in re.findall(r"\]\(\.\./(data/splits/[^)]+\.json)\)", section)]
    if not paths:
        raise ValueError("No evaluation exclusion manifests in the legacy methodology")
    excluded_ids = set(re.findall(r"^14-[A-Za-z0-9-]+$", section, re.M))
    excluded_hashes = set()
    manifest_provenance = {}
    for path in paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        excluded_ids.update(data["exclude_essay_ids"])
        excluded_hashes.update(data["exclude_whitespace_ignored_sha256"])
        manifest_provenance[str(path)] = file_hash(path)
    candidates = [c for c in candidates if c.essay.essay_id not in excluded_ids
                  and c.provenance["whitespace_ignored_sha256"] not in excluded_hashes]
    selected, stratification = stratified_sample(candidates, count=3, seed=cfg.seed)
    strata = {item["essay_id"]: item["stratum"] for item in stratification["selected"]}
    essays = [{"essay_id": row.essay.essay_id, "prompt": row.essay.prompt,
               "text": "\n\n".join(row.essay.paragraphs), "stratum": strata[row.essay.essay_id],
               "human_original_total": row.total, "provenance": row.provenance} for row in selected]
    directory = Path(directory)
    save_json(directory / "selection.json", {"essays": essays, "sampling": stratification,
        "source_audit": audit, "source_excluded_counts": audit["excluded_counts"],
        "exclusion_manifests_sha256": manifest_provenance,
        "excluded_id_count": len(excluded_ids), "excluded_compact_hash_count": len(excluded_hashes),
        "eligible_after_exclusions": len(candidates), "development_5_ids": sorted(
            re.findall(r"^14-[A-Za-z0-9-]+$", section, re.M)),
        "human_scores_exposed_to_llm": False})
    save_json(directory / "evaluation_exclusions.json", {
        "exclude_essay_ids": [e["essay_id"] for e in essays],
        "exclude_whitespace_ignored_sha256": [text_hash(e["text"], ignore_whitespace=True) for e in essays],
        "reason": "New staged-feedback implementation smoke; preserve legacy exclusion files unchanged."})
    return essays
