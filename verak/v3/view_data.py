"""Single compact episode format and a fail-closed view-budget eligibility index."""

from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
from pathlib import Path
from threading import local

from .common import file_sha, read_json, sha_text, write_json
from .data_policy import load_examples, stratified_sample
from .ko import KoreanStructure, annotate, render
from .ko.annotation import LEXICONS, read_lexicons
from .phase2 import restore_profile


def view_fingerprint(config):
    ko = Path(__file__).with_name("ko")
    files = sorted(ko.glob("*.py")) + sorted(LEXICONS.glob("*.yaml"))
    return sha_text(json.dumps({"files": {str(path.relative_to(ko)): file_sha(path) for path in files},
                               "tokenizer": file_sha(config["paths"]["policy_base"] / "tokenizer.json"),
                               "view": config["view"], "budget": config["view_token_budget"]}, sort_keys=True))


def percentiles(values):
    values = sorted(values)
    def percentile(q):
        index = (len(values) - 1) * q
        low = int(index)
        high = min(low + 1, len(values) - 1)
        return values[low] + (values[high] - values[low]) * (index - low)
    return {"n": len(values), "median": percentile(.5), "p90": percentile(.9),
            "max": max(values), "min": min(values), "percentile_method": "linear (n-1)*q"}


def filter_episode_examples(examples, records, budget=3000):
    result = []
    for example in examples:
        record = records.get(example.id)
        if record is None or record["essay_hash"] != example.essay_hash:
            raise ValueError("Missing or stale compact-view eligibility; build views before creating v3 datasets")
        if record["compact_tokens"] <= budget:
            result.append(example)
    return result


def load_episode_examples(config, split="agent_dev"):
    """Dataset entry point from Phase 2 onward. Raw load_examples is for audits only."""
    path = config["paths"]["metadata"] / "view_eligibility.json"
    if not path.exists():
        raise ValueError("Compact-view eligibility must be prepared before creating v3 datasets")
    index = read_json(path)
    if (index["view_fingerprint"] != view_fingerprint(config) or
            index["split_sha256"] != file_sha(config["paths"]["metadata"] / "splits.json") or
            index["valid_sha256"] != file_sha(config["paths"]["valid"])):
        raise ValueError("View eligibility is stale; rebuild before creating v3 datasets")
    return filter_episode_examples(load_examples(config, split), index["essays"], config["view_token_budget"])


def prepare_views(config, workers=4, *, output=None, cache_dir=None, frozen_review_path=None):
    if config["view"]["format"] != "compact" or config["view_token_budget"] != 3000:
        raise ValueError("User decision: compact-only, hard budget 3000, no truncation")
    source = {split: load_examples(config, split) for split in ("agent_train", "agent_dev")}
    jobs = [(split, example) for split, examples in source.items() for example in examples]
    output = output or config["paths"]["phase2_output"]
    output.mkdir(parents=True, exist_ok=True)
    cache = cache_dir or output / "bareun_profiles"
    cache.mkdir(parents=True, exist_ok=True)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config["paths"]["policy_base"]), local_files_only=True)
    lexicons = read_lexicons()
    state = local()

    def get_structure(example):
        path = cache / f"{example.source_line}.json"
        if path.exists():
            cached = read_json(path)
            if cached["essay_hash"] != example.essay_hash:
                raise ValueError("Cached Bareun profile does not match the essay")
            profile = restore_profile(cached["profile"])
        else:
            if not hasattr(state, "ko"):
                state.ko = KoreanStructure.from_config(config)
            profile = state.ko.analyzer.profile(example.text)
            write_json(path, {"essay_hash": example.essay_hash, "profile": profile.to_dict()})
            state.ko.analyzer.cache.clear()
        return annotate(example.text, profile, lexicons)

    def measure(job):
        split, example = job
        structure = get_structure(example)
        compact = render(structure, compact=True)
        tokens = len(tokenizer.encode(compact, add_special_tokens=False))
        return {**example.metadata(), "split": split, "characters": len(example.text),
                "sentences": len(structure.annotations), "compact_tokens": tokens,
                "eligible": tokens <= config["view_token_budget"],
                "exclusion_reason": None if tokens <= config["view_token_budget"] else "compact_view_exceeds_3000_tokens"}

    records = {}
    # Measure dev first so progress reports can immediately describe the requested distribution.
    jobs.sort(key=lambda job: (job[0] != "agent_dev", job[1].source_line))
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(measure, job) for job in jobs]
        for done, future in enumerate(as_completed(futures), 1):
            record = future.result()
            records[record["id"]] = record
            if done % 200 == 0 or done == len(jobs):
                print(f"Compact views {done}/{len(jobs)}; excluded={sum(not r['eligible'] for r in records.values())}", flush=True)
                write_json(output / "view_progress.json", {"completed": done, "total": len(jobs), "essays": records})
    stats = {}
    for split in source:
        chosen = [row for row in records.values() if row["split"] == split]
        stats[split] = {**percentiles([row["compact_tokens"] for row in chosen]),
                        "excluded": sum(not row["eligible"] for row in chosen),
                        "by_genre": {genre: {**percentiles([row["compact_tokens"] for row in chosen if row["genre"] == genre]),
                            "excluded": sum(not row["eligible"] for row in chosen if row["genre"] == genre)}
                            for genre in sorted({row["genre"] for row in chosen})}}
    index = {"format": "compact", "view_token_budget": 3000, "truncation": False,
             "view_fingerprint": view_fingerprint(config),
             "split_sha256": file_sha(config["paths"]["metadata"] / "splits.json"),
             "valid_sha256": file_sha(config["paths"]["valid"]),
             "statistics_before_exclusion": stats,
             "essays": dict(sorted(records.items(), key=lambda item: item[1]["source_line"]))}
    write_json(config["paths"]["metadata"] / "view_eligibility.json", index)
    write_json(config["paths"]["metadata"] / "view_exclusions.json", {
        "format": "compact", "view_token_budget": 3000, "source_unchanged": True,
        "rows": [row for row in index["essays"].values() if not row["eligible"]]})

    eligible_dev = filter_episode_examples(source["agent_dev"], records)
    frozen_path = frozen_review_path or output / "preparation.json"
    frozen_ids = [item["id"] for item in read_json(frozen_path)["render_review"]] if frozen_path.exists() else []
    by_id = {example.id: example for example in (source["agent_dev"] if frozen_review_path else eligible_dev)}
    if frozen_review_path and (len(frozen_ids) != 20 or not all(sid in by_id for sid in frozen_ids)):
        raise ValueError("All twenty original review essays must be re-rendered")
    review_examples = ([by_id[sid] for sid in frozen_ids] if len(frozen_ids) == 20 and all(sid in by_id for sid in frozen_ids)
                       else stratified_sample(eligible_dev, 20, seed=23))
    if sum(example.genre == "논증" for example in review_examples) < 7:
        raise ValueError("Need seven argumentative essays in the compact rendering review")
    review = []
    render_dir = output / "compact_review"
    render_dir.mkdir(exist_ok=True)
    for example in review_examples:
        structure = get_structure(example)
        prefix = render_dir / str(example.source_line)
        prefix.with_suffix(".txt").write_text(render(structure, compact=True) + "\n", encoding="utf-8")
        prefix.with_suffix(".debug.txt").write_text(render(structure, compact=False) + "\n", encoding="utf-8")
        write_json(prefix.with_suffix(".json"), structure.to_dict())
        review.append({**records[example.id], "path": str(prefix.with_suffix(".txt"))})
    result = {"statistics_before_exclusion": stats, "excluded_total": sum(not r["eligible"] for r in records.values()),
              "review_genres": dict(Counter(row["genre"] for row in review)), "review": review}
    write_json(output / "compact_views.json", result)
    return result
