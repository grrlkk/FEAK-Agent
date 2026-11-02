"""Freeze three held-out originals without changing any legacy exclusions."""
import json
import importlib
from itertools import permutations
from pathlib import Path
import random
import re
from statistics import quantiles
import sys

import yaml

from .adapters import bootstrap, legacy_config
from .events import save_json


def diverse_sample(candidates, genres, *, seed):
    """One human-score tertile each and three distinct native genres."""
    lower, upper = quantiles([row.total for row in candidates], n=3, method="inclusive")
    names = ("low", "middle", "high")
    groups = {name: [] for name in names}
    for row in sorted(candidates, key=lambda r: r.essay.essay_id):
        name = "low" if row.total <= lower else "middle" if row.total <= upper else "high"
        groups[name].append(row)
    rng = random.Random(seed)
    assignments = list(permutations(sorted(set(genres.values())), 3))
    rng.shuffle(assignments)
    for assignment in assignments:
        pools = [[row for row in groups[name] if genres.get(row.essay.essay_id) == genre]
                 for name, genre in zip(names, assignment)]
        if all(pools):
            selected = [rng.choice(pool) for pool in pools]
            return selected, {"seed": seed, "cutpoints": [lower, upper],
                "score": "sum of eight original human-derived rubric scores",
                "quantile_method": "inclusive; equal scores stay in the same stratum",
                "available": {name: len(group) for name, group in groups.items()},
                "quotas": {name: 1 for name in names},
                "genre_policy": "Seeded feasible genre assignment, then one random essay per stratum/genre",
                "selected": [{**row.provenance, "stratum": name, "genre": genre}
                             for row, name, genre in zip(selected, names, assignment)]}
    raise ValueError("Cannot select three distinct genres across low/middle/high human-score strata")


def select_three(cfg, directory, *, exclude_runs=(), distinct_genres=False):
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
    for run in exclude_runs:
        path = Path(run) / "selection.json"
        for essay in json.loads(path.read_text())["essays"]:
            excluded_ids.add(essay["essay_id"])
            excluded_hashes.add(text_hash(essay["text"], ignore_whitespace=True))
        manifest_provenance[str(path.resolve())] = file_hash(path)
    candidates = [c for c in candidates if c.essay.essay_id not in excluded_ids
                  and c.provenance["whitespace_ignored_sha256"] not in excluded_hashes]
    genres = {}
    if distinct_genres:
        from para_agent.data.loader import iter_source
        scorer_root = str(Path(legacy.scorer.repository_path).resolve())
        sys.path.insert(0, scorer_root)
        policy = importlib.import_module("verak.v3.data_policy")
        if not Path(policy.__file__).resolve().is_relative_to(scorer_root):
            raise ValueError("Genre rules must come from the configured native repository")
        by_line = {row.provenance["source_line"]: row.essay.essay_id for row in candidates}
        for line, row in iter_source(Path(audit["source_paths"]["test_jsonl"])):
            if line in by_line:
                headings = policy.feedback_headings(row["assistant"])
                matches = [g for g, expected in policy.HEADING_SETS.items()
                           if len(headings) == 8 and set(headings) == expected]
                if len(matches) == 1:
                    genres[by_line[line]] = matches[0]
        selected, stratification = diverse_sample(candidates, genres, seed=cfg.seed)
    else:
        selected, stratification = stratified_sample(candidates, count=3, seed=cfg.seed)
    strata = {item["essay_id"]: item["stratum"] for item in stratification["selected"]}
    essays = [{"essay_id": row.essay.essay_id, "prompt": row.essay.prompt,
               "text": "\n\n".join(row.essay.paragraphs), "stratum": strata[row.essay.essay_id],
               "human_original_total": row.total, "provenance": row.provenance,
               **({"genre": genres[row.essay.essay_id]} if distinct_genres else {})} for row in selected]
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


def select_comparison_six(cfg, directory, baseline):
    baseline, directory = Path(baseline).resolve(), Path(directory).resolve()
    previous = json.loads((baseline / "selection.json").read_text())
    if len(previous["essays"]) != 3:
        raise ValueError("Comparison requires exactly three frozen baseline essays")
    # Exclude every earlier local selection, including the original three and retries.
    exclude_runs = [p.parent for p in baseline.parent.glob("*/selection.json") if p.parent != directory]
    fresh = select_three(cfg, directory, exclude_runs=exclude_runs, distinct_genres=True)
    manifest = json.loads((directory / "selection.json").read_text())
    from para_agent.data.loader import text_hash
    if manifest["source_audit"]["source_sha256"] != previous["source_audit"]["source_sha256"]:
        raise ValueError("Source versions differ between the baseline and new cohort")
    coverage = {e["essay_id"]: e["genre"] for e in json.loads(
        (baseline / "scorer_metadata/provenance.json").read_text())["coverage"]}
    repeated = [{**e, "cohort": "repeated", "genre": coverage[e["essay_id"]]} for e in previous["essays"]]
    essays = repeated + [{**e, "cohort": "new"} for e in fresh]
    manifest.update(essays=essays, baseline_run=str(baseline), sampling_new=manifest.pop("sampling"))
    save_json(directory / "selection.json", manifest)
    save_json(directory / "evaluation_exclusions.json", {
        "exclude_essay_ids": [e["essay_id"] for e in essays],
        "exclude_whitespace_ignored_sha256": [text_hash(e["text"], ignore_whitespace=True) for e in essays],
        "reason": "Six-essay staged-feedback student-gate validation; original data remain local."})
    save_json(directory / "comparison_reference.json", {"baseline_run": str(baseline),
        "repeated_ids": [e["essay_id"] for e in repeated], "new_ids": [e["essay_id"] for e in fresh],
        "budget_scope": "One new shared cap for all six essays and any retries of this experiment"})
    return essays
