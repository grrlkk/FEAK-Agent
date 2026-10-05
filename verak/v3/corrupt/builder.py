"""Balanced, deterministic curriculum construction with private restoration records."""

from collections import Counter, defaultdict
import json
import random

from ..common import read_json, sha_text, write_json
from ..ko import render
from ..phase2 import write_jsonl
from ..view_data import view_fingerprint
from .document import BareunBank, Document, source_document
from .operators import LEVELS, apply, candidates, exact_restoration_satisfies, recovery_target, restore_record
from .sources import select_sources


class BuildStats:
    def __init__(self):
        self.attempts, self.passed, self.failures = Counter(), Counter(), Counter()
        self.ineligible, self.kept = Counter(), Counter()

    def export(self):
        return {"operators": {op: {"attempts": self.attempts[op], "verified": self.passed[op],
            "discarded": self.attempts[op]-self.passed[op],
            "verification_rate": self.passed[op]/self.attempts[op] if self.attempts[op] else None,
            "no_eligible_site": self.ineligible[op], "kept_records": self.kept[op]}
            for op in LEVELS}, "failure_reasons": dict(self.failures)}


def load_sources(config, split, bank):
    examples, stats = select_sources(config, split)
    scores = read_json(config["paths"]["phase3_output"] / f"sources_{split}.json")
    if {e.id for e in examples} != set(scores["rows"]):
        raise ValueError("Finish scoring every selected source before building")
    sources = []
    for example in examples:
        result = scores["rows"][example.id]
        if result["essay_hash"] != example.essay_hash:
            raise ValueError("Stale source selection")
        if result["eligible"]:
            sources.append((example, source_document(config, example, bank), result["score"]))
    return sources


def donor_pool(sources):
    donors = []
    for example, doc, _ in sources:
        for ann in doc.structure().annotations:
            if not ann.multi_unit and 25 <= len(ann.text) <= 150 and ann.style in {"한다", "합니다"}:
                donors.append({"source_id": example.id, "question_hash": example.question_hash,
                    "sid": ann.sid, "text": ann.text, "style": ann.style})
    return donors


def composition(level, rng):
    return {"L1": (0, 1), "L2": (0, rng.choice((2, 3))),
            "L3": (1, rng.choice((0, 1))), "L4": (2, rng.choice((1, 2)))}[level]


def curriculum_attempts(requested, allowed):
    """Try the assigned level first, then feasible alternatives without weakening it.

    A source consisting entirely of multi-units can still receive L3 off-topic
    insertion. Never invent a sentence-internal site or mislabel an L1 as L4.
    """
    order = [requested] + [v for v in ("L3", "L1", "L2", "L4") if v != requested and v in allowed]
    return [(level, retry, fallback) for fallback, level in enumerate(order) for retry in range(40)]


def source_coupled_changes(changes, structure):
    originals = {a.sid: a for a in structure.annotations}
    # A sentence inserted by an earlier G_OFFTOPIC has no source dependency to
    # recover. Its own record already requires removing it.
    return [{**change, "recovery_predecessor_id": originals[change["sid"]].predecessor_id}
            for change in changes if change["sid"] in originals]


def build_episode(example, source, score, *, split, episode_id, level, seed, bank,
                  enabled, donors, vague_cache, stats, level_counts, count_tokens):
    rng = random.Random(seed)
    source_structure = source.structure()
    globals_n, locals_n = composition(level, rng)
    doc, records, used, provisional = source.clone(), [], set(), Counter()
    # Spread operator use within each level, and the three local levels equally.
    for is_global in [True]*globals_n + [False]*locals_n:
        choices = [op for op in enabled if (LEVELS[op] == "GLOBAL") == is_global]
        jitter = {op: rng.random() for op in choices}
        choices.sort(key=lambda op: (0 if is_global else level_counts[LEVELS[op]] + provisional[LEVELS[op]],
                                     stats.kept[op] + sum(r["op"] == op for r in records), jitter[op]))
        accepted = False
        for op in choices:
            variants = candidates(doc, op, donors=donors, vague_cache=vague_cache,
                                  question_hash=example.question_hash)
            variants = [p for p in variants if not used.intersection(p.sids)]
            if not variants:
                stats.ineligible[op] += 1
                continue
            rng.shuffle(variants)
            for proposal in variants[:8]:
                stats.attempts[op] += 1
                try:
                    changed, record = apply(doc, proposal, bank)
                    restored = restore_record(changed, record, bank)
                    if restored.text != doc.text or not exact_restoration_satisfies(restored, record):
                        raise ValueError("Exact restoration oracle failed")
                    stats.passed[op] += 1
                except ValueError as error:
                    stats.failures[f"{op}: {error}"] += 1
                    continue
                record["record_id"] = f"{episode_id}:R{len(records)+1}"
                # Recovery is against the source document, not a placement shifted
                # by an earlier deletion/move. The inverse still undoes this step.
                record["recovery_target"] = recovery_target(source, proposal, source_structure,
                    {u.sid: u.text for u in source.units if u.sid in proposal.sids})
                record["coupled_changes"] = source_coupled_changes(record["coupled_changes"], source_structure)
                record["params"]["seed"] = seed
                doc, accepted = changed, True
                records.append(record)
                used.update(proposal.sids)
                provisional[LEVELS[op]] += 1
                break
            if accepted:
                break
        if not accepted:
            raise ValueError("Curriculum composition unavailable on distinct eligible units")
    view = render(doc.structure(), compact=True)
    tokens = count_tokens(view)
    if tokens > 3000:
        raise ValueError("corrupted_compact_view_exceeds_3000_tokens")
    inverse = doc
    for record in reversed(records):
        inverse = restore_record(inverse, record, bank)
    if inverse.text != source.text:
        raise ValueError("Episode inverse failed")
    if not all(exact_restoration_satisfies(inverse, r) for r in records):
        raise ValueError("Recovery targets must all be satisfied simultaneously by the source")
    return {"schema_version": "phase3_surface_records_v1", "episode_id": episode_id,
        "source_id": example.id, "split": split, "question": example.question, "question_hash": example.question_hash,
        "genre": example.genre, "level": level, "seed": seed, "source_text": source.text,
        "corrupted_text": doc.text, "source_hash": example.essay_hash, "corrupted_hash": sha_text(doc.text),
        "records": records, "source_layout": source.snapshot(), "corrupted_layout": doc.snapshot(),
        "compact_view": view, "compact_tokens": tokens, "q_source": score["mean"], "q_corrupted": None,
        "preexisting_spell_spans": [], "spelling_detection": "not_run_optional",
        "analysis": "frozen_phase2c_on_record_units; Bareun_reanalysis_of_every_changed_surface",
        "supervision_private": ["source_text", "source_layout", "records", "q_source"]}


def build_dataset(config, split, out, *, seed=13, per_essay=2, levels=("L1", "L2", "L3", "L4"),
                  disabled=(), bank=None):
    if per_essay != 2 or not levels or set(levels)-{"L1", "L2", "L3", "L4"}:
        raise ValueError("Phase 3 requires two variants and known curriculum levels")
    bank = bank or BareunBank(config)
    sources = load_sources(config, split, bank)
    donors = donor_pool(sources)  # Same split only; a different question is mandatory.
    vague_path = config["paths"]["phase3_output"] / "vague_cache.json"
    vague = read_json(vague_path) if vague_path.exists() else {}
    enabled = sorted(set(LEVELS)-set(disabled))
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config["paths"]["policy_base"]), local_files_only=True)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    stats, level_counts, rows, exclusions = BuildStats(), Counter(), [], []
    shuffled = list(sources)
    random.Random(seed).shuffle(shuffled)
    for i, (example, source, score) in enumerate(shuffled):
        dominant = source.structure().dominant_style
        nearby_donors = [d for d in donors if d["question_hash"] != example.question_hash and
                          d["style"] == dominant]
        random.Random(seed + example.source_line).shuffle(nearby_donors)
        nearby_donors = nearby_donors[:8]
        pair, hashes = [], set()
        for variant in range(per_essay):
            requested_level = levels[(i + 2*variant) % len(levels)]
            episode_id = f"{split}:{example.source_line}:v{variant+1}"
            attempts = curriculum_attempts(requested_level, levels)
            for level, retry, fallback in attempts:
                try:
                    row = build_episode(example, source, score, split=split, episode_id=episode_id,
                        level=level, seed=seed + example.source_line*1000 + variant*100 + retry + fallback*10000000,
                        bank=bank, enabled=enabled, donors=nearby_donors, vague_cache=vague,
                        stats=stats, level_counts=level_counts, count_tokens=count_tokens)
                    if row["corrupted_hash"] in hashes:
                        raise ValueError("Duplicate variants for one source")
                    row["generation_retry"] = retry
                    row["requested_level"] = requested_level
                    row["curriculum_fallback"] = level != requested_level
                    pair.append(row)
                    hashes.add(row["corrupted_hash"])
                    break
                except ValueError as error:
                    stats.failures[str(error)] += 1
            else:
                exclusions.append({"source_id": example.id, "requested_level": requested_level,
                    "reason": "cannot_build_two_complete_distinct_variants", "attempts": len(attempts)})
                break
        if len(pair) == per_essay:
            rows.extend(pair)
            for row in pair:
                for record in row["records"]:
                    stats.kept[record["op"]] += 1
                    if record["level"] != "GLOBAL":
                        level_counts[record["level"]] += 1
        if (i+1) % 20 == 0 or i+1 == len(shuffled):
            print(f"Build {split} {i+1}/{len(shuffled)}; episodes={len(rows)}; exclusions={len(exclusions)}", flush=True)
            write_jsonl(out, rows)
            write_json(out.with_suffix(".stats.json"), {**stats.export(), "exclusions": exclusions,
                "source_essays": len(sources), "kept_sources": len(rows)//2, "episodes": len(rows),
                "operator_levels": dict(Counter(r["level"] for x in rows for r in x["records"])),
                "curriculum_levels": dict(Counter(r["level"] for r in rows)), "local_level_counts": dict(level_counts),
                "genres": dict(Counter(r["genre"] for r in rows)), "disabled": list(disabled),
                "seed": seed, "view_fingerprint": view_fingerprint(config), "scored": False})
    return rows
