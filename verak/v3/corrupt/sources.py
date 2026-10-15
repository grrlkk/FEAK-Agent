"""Offline gold selection over valid only; never export gold to episodes."""

from collections import defaultdict
import json
import statistics

from ..common import read_json, write_json
from ..view_data import load_episode_examples


def select_sources(config, split):
    examples = load_episode_examples(config, split)
    wanted = {e.source_line: e for e in examples}
    groups = defaultdict(list)
    with config["paths"]["valid"].open(encoding="utf-8") as handle:
        for line, raw in enumerate(handle, 1):
            if line not in wanted:
                continue
            row, example = json.loads(raw), wanted[line]
            a, b = row["grader_1_scores"], row["grader_2_scores"]
            if len(a) != 8 or len(b) != 8:
                raise ValueError("Expected eight human rubric scores from both graders")
            total = sum((x + y) / 2 for x, y in zip(a, b))
            group = example.genre if example.genre in {"설명", "논증", "정서"} else example.question_hash
            groups[group].append((example, total))
    chosen, statistics_by_group = [], {}
    for group, items in sorted(groups.items()):
        values = [v for _, v in items]
        threshold = statistics.quantiles(values, n=4, method="inclusive")[2] if len(values) > 1 else values[0]
        selected = [e for e, score in items if score >= threshold and 500 <= len(e.text) <= 2500]
        chosen.extend(selected)
        statistics_by_group[group] = {"population": len(items), "gold_q75": threshold,
                                      "length_and_quartile_eligible": len(selected)}
    return sorted(chosen, key=lambda e: e.source_line), {
        "split": split, "gold": "sum of the two human graders' per-rubric means",
        "threshold_population": "each genre within this agent split, before length filtering",
        "quantile": "inclusive linear q75; boundary ties retained", "groups": statistics_by_group,
        "candidates": len(chosen)}


def score_sources(config, split, scorer, *, progress=print):
    """Resume atomically per source. Parse failures remain explicit exclusions."""
    from ..score.kanana import InputTooLong, ScoreParseError
    candidates, stats = select_sources(config, split)
    path = config["paths"]["phase3_output"] / f"sources_{split}.json"
    old = read_json(path) if path.exists() else {"rows": {}}
    rows = old["rows"]
    if rows and old.get("scorer_fingerprint") != scorer.fingerprint:
        raise ValueError("Source scores belong to a different scorer")
    for i, example in enumerate(candidates, 1):
        if example.id in rows:
            if rows[example.id]["essay_hash"] != example.essay_hash:
                raise ValueError("Source score hash mismatch")
            continue
        try:
            score = scorer.score(example.question, example.text).to_dict()
            row = {**example.metadata(), "eligible": True, "score": score}
        except (InputTooLong, ScoreParseError) as error:
            row = {**example.metadata(), "eligible": False, "error": type(error).__name__}
        rows[example.id] = row
        write_json(path, {**stats, "scorer_fingerprint": scorer.fingerprint, "rows": rows})
        if i % 20 == 0 or i == len(candidates):
            progress(f"Source scores {split} {i}/{len(candidates)}", flush=True)
    stats.update(scorer_fingerprint=scorer.fingerprint, rows=rows,
                 accepted=sum(r["eligible"] for r in rows.values()))
    write_json(path, stats)
    return stats
