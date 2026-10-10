"""Fresh seed-31 Phase 2c verification samples, frozen before any judgment."""

from collections import Counter
import random
import shutil

from .common import file_sha, read_json, write_json
from .ko import annotate_structural
from .ko.annotation import read_lexicons
from .phase2 import read_jsonl, restore_profile, write_jsonl
from .phase2b import input_hash
from .view_data import load_episode_examples, prepare_views, view_fingerprint

COUNTS = {"omission": 80, "conjunction": 40, "word": 50, "style": 60}
LEVELS = {"omission": "SENTENCE", "conjunction": "SENTENCE", "word": "WORD", "style": "TEXT"}


def preserve_phase2b(config):
    """Create immutable audit snapshots on first preparation, not on each resume."""
    output, repo = config["paths"]["phase2c_output"], config["paths"]["repo"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preserved_hashes.json"
    if not path.exists():
        names = ["imple/reports/V3_PHASE_2.md", "imple/reports/V3_PHASE_2b.md",
            "verak/v3/data/ec_check.jsonl", "verak/v3/data/ec_disagreements.jsonl",
            "verak/v3/data/structure_check_phase2b.jsonl", "verak/v3/data/structure_disagreements_phase2b.jsonl",
            "verak/v3/data/splits.json", "data/data_jsonl/valid.jsonl", "verak/v3/ko/patterns.py",
            "verak/v3/outputs/phase2b/sentence_high_60.jsonl"]
        for phase in ("phase1", "phase2", "phase2b"):
            for name in ("api_budget.json", "calls.jsonl"):
                relative = f"verak/v3/outputs/{phase}/{name}"
                if (repo / relative).exists():
                    names.append(relative)
        write_json(path, {name: file_sha(repo / name) for name in names})
    for name in ("view_eligibility.json", "view_exclusions.json"):
        backup = output / ("phase2b_" + name)
        if not backup.exists():
            shutil.copyfile(config["paths"]["metadata"] / name, backup)
    assert_preserved(config)


def assert_preserved(config):
    for relative, expected in read_json(config["paths"]["phase2c_output"] / "preserved_hashes.json").items():
        if file_sha(config["paths"]["repo"] / relative) != expected:
            raise ValueError(f"Preserved artifact changed: {relative}")


def prepare(config, workers=4):
    preserve_phase2b(config)
    output, metadata = config["paths"]["phase2c_output"], config["paths"]["metadata"]
    path = metadata / "structure_check_phase2c.jsonl"
    if path.exists():
        raise ValueError("Phase 2c sample already frozen; never overwrite judgments")
    views = prepare_views(config, workers, output=output,
        cache_dir=config["paths"]["phase2_output"] / "bareun_profiles",
        frozen_review_path=config["paths"]["phase2_output"] / "preparation.json")
    old_ids = {r["sentence_id"] for name in ("ec_check.jsonl", "structure_check_phase2b.jsonl")
               for r in read_jsonl(metadata / name)}
    examples = load_episode_examples(config, "agent_dev")
    lexicons, all_rows = read_lexicons(), []
    for example in examples:
        cached = read_json(config["paths"]["phase2_output"] / "bareun_profiles" / f"{example.source_line}.json")
        if cached["essay_hash"] != example.essay_hash:
            raise ValueError("Cached essay mismatch")
        structure = annotate_structural(example.text, restore_profile(cached["profile"]), lexicons)
        for i, ann in enumerate(structure.annotations):
            sentence_id = f"{example.id}:{ann.sid}"
            if sentence_id in old_ids:
                continue
            prior = structure.annotations[max(0, i - 2):i]
            relations = [c for c in ann.connectives if c["eligible"] and c["kind"] == "EC"]
            embedded = [e for e in ann.final_endings if e["embedded"] and
                        (e["inside_quotation"] or e["followed_by_quotative"] or e["interrogative_or_colloquial"])]
            all_rows.append({"sentence_id": sentence_id, "essay_id": example.id, "genre": example.genre,
                "essay_hash": example.essay_hash, "sid": ann.sid, "paragraph": ann.paragraph,
                "paragraph_initial": not prior or prior[-1].paragraph != ann.paragraph,
                "sentence": ann.text, "previous_sentences": [
                    {"sid": a.sid, "paragraph": a.paragraph, "sentence": a.text} for a in prior],
                "subject_omitted": ann.subject_omitted, "omission_uncertain": ann.omission_uncertain,
                "subject_evidence": ann.subject_evidence,
                "predecessor_id": ann.predecessor_id, "cross_paragraph": ann.cross_paragraph,
                "initial_conjunction": ann.initial_conj, "unambiguous_ec": relations,
                "style": ann.style, "dominant_style": structure.dominant_style,
                "final_ending": ann.final_ending, "final_endings": ann.final_endings,
                "multi_unit": ann.multi_unit, "embedded_or_quoted_ending": bool(embedded),
                "morphemes": [{"token_id": f"M{j + 1}", "form": t.form, "tag": t.tag} for j, t in enumerate(ann.tokens)]})
    rng, selected, used, pools = random.Random(config["phase2c_judge"]["sample_seed"]), [], set(), {}
    def take(label, eligible, n, check):
        values = [r for r in all_rows if r["sentence_id"] not in used and eligible(r)]
        pools[label] = len(values)
        if len(values) < n:
            raise ValueError(f"Insufficient eligible {label}: {len(values)} < {n}")
        chosen = [{**r, "check": check, "subgroup": label} for r in rng.sample(values, n)]
        selected.extend(chosen)
        used.update(r["sentence_id"] for r in chosen)
        return chosen
    positive = take("omitted_true", lambda r: r["subject_omitted"], 40, "omission")
    negative = take("omitted_false", lambda r: not r["subject_omitted"], 40, "omission")
    conj = lambda r: r["initial_conjunction"] and r["initial_conjunction"]["eligible"]
    # A paragraph-initial conjunction must have the previous paragraph available.
    cp = take("paragraph_initial", lambda r: conj(r) and r["cross_paragraph"], 10, "conjunction")
    cn = take("within_paragraph", lambda r: conj(r) and not r["paragraph_initial"], 30, "conjunction")
    word = take("coarse_ec", lambda r: bool(r["unambiguous_ec"]), 50, "word")
    by_essay = {}
    for row in all_rows:
        if row["sentence_id"] not in used:
            by_essay.setdefault(row["essay_id"], []).append(row)
    eligible = [eid for eid, rows in by_essay.items() if len(rows) >= 4 and any(r["embedded_or_quoted_ending"] for r in rows)]
    pools["style_essays_with_embedded_or_quoted_ending"] = len(eligible)
    if len(eligible) < 15:
        raise ValueError("Need 15 essays with four fresh sentences including embedded/quoted endings")
    styles = []
    for eid in rng.sample(eligible, 15):
        candidates = by_essay[eid]
        special = rng.choice([r for r in candidates if r["embedded_or_quoted_ending"]])
        for row in [special] + rng.sample([r for r in candidates if r is not special], 3):
            new = {**row, "check": "style", "subgroup": "final_ef_style"}
            selected.append(new)
            styles.append(new)
            used.add(row["sentence_id"])
    pilot = positive[:3] + negative[:2] + cp[:2] + cn[:3] + word[:5] + [styles[i] for i in (0, 5, 10, 15, 16)]
    pilot_ids = {r["sentence_id"] for r in pilot}
    for row in selected:
        row.update(item_id=row["check"] + ":" + row["sentence_id"], level=LEVELS[row["check"]],
                   pilot=row["sentence_id"] in pilot_ids, llm_judgment_1=None, llm_judgment_2=None,
                   llm_ok=None, human_ok=None)
    assert len(selected) == len(used) == 230 and not used & old_ids
    assert Counter(r["check"] for r in selected) == COUNTS
    assert Counter(r["check"] for r in selected if r["pilot"]) == dict.fromkeys(COUNTS, 5)
    write_jsonl(path, selected)
    summary = {"seed": 31, "input_sha256": input_hash(selected), "view_fingerprint": view_fingerprint(config),
        "excluded_previous_sample_ids": sorted(old_ids), "previous_sample_overlap": 0,
        "checks": dict(Counter(r["check"] for r in selected)), "pools": pools,
        "subgroups": dict(Counter(r["subgroup"] for r in selected)),
        "sample_genres": dict(Counter(r["genre"] for r in selected)),
        "text_essays": len({r["essay_id"] for r in styles}),
        "text_sentences_per_essay": dict(Counter(r["essay_id"] for r in styles)),
        "text_embedded_or_quoted": sum(r["embedded_or_quoted_ending"] for r in styles),
        "context": "Two immediately preceding sentences in document order, including previous paragraphs",
        "views": views, "sample_path": str(path)}
    write_json(output / "preparation.json", summary)
    assert_preserved(config)
    return summary
