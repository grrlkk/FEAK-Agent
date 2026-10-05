"""Frozen Phase 2b samples, using valid-only cached Bareun observations."""

from collections import Counter
import json
import random
import shutil

from .common import file_sha, read_json, sha_text, write_json
from .ko import annotate
from .ko.annotation import read_lexicons
from .ko.levels import relation_eligible
from .phase2 import read_jsonl, restore_profile, write_jsonl
from .view_data import load_episode_examples, prepare_views, view_fingerprint


def preserve_phase2(config):
    """Snapshot before any mutation; later runs must match the original snapshot."""
    output, repo = config["paths"]["phase2b_output"], config["paths"]["repo"]
    output.mkdir(parents=True, exist_ok=True)
    path = output / "preserved_hashes.json"
    if not path.exists():
        relative = ["verak/v3/data/ec_check.jsonl", "verak/v3/data/ec_disagreements.jsonl",
            "verak/v3/outputs/phase2/api_budget.json", "verak/v3/outputs/phase2/calls.jsonl",
            "verak/v3/outputs/phase2/ec_results.json", "verak/v3/outputs/phase1/api_budget.json",
            "imple/reports/V3_PHASE_2.md"]
        write_json(path, {name: file_sha(repo / name) for name in relative})
    for name in ("view_eligibility.json", "view_exclusions.json"):
        backup = output / ("phase2_" + name)
        if not backup.exists():
            shutil.copyfile(config["paths"]["metadata"] / name, backup)
    assert_phase2_unchanged(config)


def assert_phase2_unchanged(config):
    snapshots = read_json(config["paths"]["phase2b_output"] / "preserved_hashes.json")
    for relative, expected in snapshots.items():
        if file_sha(config["paths"]["repo"] / relative) != expected:
            raise ValueError(f"Preserved artifact changed: {relative}")


def input_hash(rows):
    return sha_text(json.dumps([{k: v for k, v in r.items() if not k.startswith("llm_") and k != "human_ok"}
                               for r in rows], ensure_ascii=False, sort_keys=True))


def prepare(config, workers=4):
    preserve_phase2(config)
    output, metadata = config["paths"]["phase2b_output"], config["paths"]["metadata"]
    path = metadata / "structure_check_phase2b.jsonl"
    if path.exists():
        raise ValueError("Phase 2b sample already frozen; do not resample or overwrite judgments")
    views = prepare_views(config, workers, output=output,
        cache_dir=config["paths"]["phase2_output"] / "bareun_profiles",
        frozen_review_path=config["paths"]["phase2_output"] / "preparation.json")
    examples = load_episode_examples(config, "agent_dev")
    old_ids = {row["sentence_id"] for row in read_jsonl(metadata / "ec_check.jsonl")}
    lexicons = read_lexicons()
    all_rows = []
    for example in examples:
        cached = read_json(config["paths"]["phase2_output"] / "bareun_profiles" / f"{example.source_line}.json")
        if cached["essay_hash"] != example.essay_hash:
            raise ValueError("Bareun cache text mismatch")
        structure = annotate(example.text, restore_profile(cached["profile"]), lexicons)
        for i, ann in enumerate(structure.annotations):
            sid = f"{example.id}:{ann.sid}"
            if sid in old_ids:
                continue
            prior = [a for a in structure.annotations[:i] if a.paragraph == ann.paragraph][-2:]
            refs = [e for e in structure.edges if e.src == ann.sid and e.kind == "REF"]
            ec = [c for c in ann.connectives if relation_eligible(c)]
            pm = ann.polarity == "NEG" or ann.modality is not None
            row = {"sentence_id": sid, "essay_id": example.id, "genre": example.genre,
                   "essay_hash": example.essay_hash, "paragraph": ann.paragraph,
                   "sentence": ann.text, "previous_sentences": [{"sid": a.sid, "sentence": a.text} for a in prior],
                   "sid": ann.sid, "multi_unit": ann.multi_unit,
                   "morphemes": [{"token_id": f"M{j + 1}", "form": t.form, "tag": t.tag} for j, t in enumerate(ann.tokens)],
                   "unambiguous_ec": [{"token_id": c.token_id, "form": c.form, "relation": c.candidates[0]} for c in ec],
                   "polarity_modality": {"polarity": ann.polarity, "modality": ann.modality} if pm else None,
                   "reference": {"target": refs[0].dst, "label": refs[0].label, "confidence": refs[0].confidence} if refs else None,
                   "omitted_subject": not ann.subject.realized,
                   "initial_conjunction": {"form": ann.initial_conj.form, "relation": ann.initial_conj.relation} if ann.initial_conj else None,
                   "style": ann.style, "dominant_style": structure.dominant_style,
                   "final_endings": ann.final_endings,
                   "special_ending": any(v["interrogative_or_colloquial"] for v in ann.final_endings)}
            all_rows.append(row)
    # No judgments have been made when these quotas and ordering are frozen.
    rng, selected, used = random.Random(config["structure_judge"]["sample_seed"]), [], set()
    pools = {}
    def take(label, eligible, n, check):
        values = [r for r in all_rows if r["sentence_id"] not in used and eligible(r)]
        pools[label] = len(values)
        if len(values) < n:
            raise ValueError(f"Insufficient eligible {label} items: {len(values)} < {n}")
        chosen = rng.sample(values, n)
        for row in chosen:
            row = {**row, "check": check, "subgroup": label}
            selected.append(row)
            used.add(row["sentence_id"])
        return selected[-n:]
    # WORD gets adequate independent coverage of both requested fields.
    word_rel = take("relation", lambda r: bool(r["unambiguous_ec"]), 40, "WORD")
    word_pm = take("polarity_modality", lambda r: r["polarity_modality"] is not None, 40, "WORD")
    high = take("HIGH", lambda r: r["omitted_subject"] and r["reference"] and r["reference"]["confidence"] == "HIGH", 60, "SENTENCE")
    low = take("LOW", lambda r: r["omitted_subject"] and r["reference"] and r["reference"]["confidence"] == "LOW", 20, "SENTENCE")
    conj = take("conjunction", lambda r: r["initial_conjunction"] is not None, 40, "SENTENCE")
    by_essay = {}
    for row in all_rows:
        if row["sentence_id"] not in used:
            by_essay.setdefault(row["essay_id"], []).append(row)
    eligible_essays = [key for key, rows in by_essay.items() if len(rows) >= 4 and any(r["special_ending"] for r in rows)]
    pools["TEXT_essays_with_special_ending"] = len(eligible_essays)
    if len(eligible_essays) < 20:
        raise ValueError("Need twenty essays with four fresh sentences including a special ending")
    for eid in rng.sample(eligible_essays, 20):
        candidates = by_essay[eid]
        special = rng.choice([r for r in candidates if r["special_ending"]])
        for row in [special] + rng.sample([r for r in candidates if r is not special], 3):
            selected.append({**row, "check": "TEXT", "subgroup": "style"})
            used.add(row["sentence_id"])
    # 5 per check, explicitly mixed so the pilot covers HIGH/LOW/conjunction and
    # both WORD fields; the TEXT pilot spans essays and includes special endings.
    pilot_ids = {r["sentence_id"] for r in word_rel[:3] + word_pm[:2] + high[:2] + low[:1] + conj[:2]}
    text_rows = [r for r in selected if r["check"] == "TEXT"]
    pilot_ids.update(text_rows[i]["sentence_id"] for i in (0, 5, 10, 15, 16))
    for row in selected:
        row.update(item_id=row["check"] + ":" + row["sentence_id"], pilot=row["sentence_id"] in pilot_ids,
                   llm_judgment_1=None, llm_judgment_2=None, llm_ok=None, human_ok=None)
    assert len(selected) == len(used) == 280
    assert Counter(r["check"] for r in selected if r["pilot"]) == {"WORD": 5, "SENTENCE": 5, "TEXT": 5}
    write_jsonl(path, selected)
    summary = {"seed": 29, "input_sha256": input_hash(selected), "view_fingerprint": view_fingerprint(config),
               "excluded_phase2_sentence_ids": sorted(old_ids), "new_sample_overlap_with_phase2": 0,
               "cross_check_duplicate_sentences": 0, "pools": pools,
               "checks": dict(Counter(r["check"] for r in selected)),
               "subgroups": dict(Counter(r["subgroup"] for r in selected)),
               "text_essays": len({r["essay_id"] for r in text_rows}),
               "text_special_endings": sum(r["special_ending"] for r in text_rows),
               "text_sentences_per_essay": dict(Counter(r["essay_id"] for r in text_rows)),
               "word_relation_applicable": sum(bool(r["unambiguous_ec"]) for r in selected if r["check"] == "WORD"),
               "word_pm_applicable": sum(r["polarity_modality"] is not None for r in selected if r["check"] == "WORD"),
               "sample_path": str(path), "views": views}
    write_json(output / "preparation.json", summary)
    assert_phase2_unchanged(config)
    return summary
