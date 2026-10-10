"""Frozen dev-only gap reconstruction cases; generators never receive hidden answers."""

from collections import Counter, defaultdict
import random

from .common import file_sha, read_json, sha_text, write_json
from .corrupt.document import Document
from .corrupt.instance_policy import back_reference, candidates
from .corrupt.operators import apply
from .data_policy import assert_data_tree_clean
from .phase2 import read_jsonl, restore_profile, write_jsonl
from .view_data import load_episode_examples

GAP = "[MISSING_SENTENCE]"
CONTEXT_KEYS = ("question", "essay_style", "corrupted_paragraph_with_gap",
                "previous_sentence", "next_sentence", "next_cross_paragraph")


def reconstruction_payload(case):
    payload = {key: case[key] for key in CONTEXT_KEYS}
    # Fail closed even when the deleted sentence is repeated elsewhere in context.
    if any(case['deleted_sentence'] in value for value in payload.values() if isinstance(value,str)):
        raise ValueError("Deleted sentence leaked into reconstruction input")
    if payload["corrupted_paragraph_with_gap"].count(GAP) != 1:
        raise ValueError("Exactly one gap marker is required")
    return payload


def labeling_payload(case, reconstruction):
    return {**reconstruction_payload(case), "deleted_sentence": case["deleted_sentence"],
            "reconstruction": reconstruction}


def make_case(example, record, style, *, origin, seed=43):
    if record["op"] != "G_DELETE_SUPPORT" or len(record["sids"]) != 1:
        raise ValueError("Expected one deleted supporting sentence")
    sid = record["sids"][0]
    paragraphs = record["inverse"]["paragraphs"]
    positions = [(p, i, u) for p in paragraphs for i, u in enumerate(p["units"])]
    index = next(i for i, (_, _, u) in enumerate(positions) if u["sid"] == sid)
    paragraph, position, unit = positions[index]
    if position == 0 or index + 1 == len(positions):
        raise ValueError("Deletion must be noninitial with a following sentence")
    next_paragraph, _, following = positions[index + 1]
    if not back_reference(following["text"]):
        raise ValueError("Next sentence must start with an approved back-reference")
    if record["original_text"][sid] != unit["text"]:
        raise ValueError("Hidden answer differs from the recorded deleted text")
    paragraph_text = "".join(u["leading"] + (GAP if u["sid"] == sid else u["text"])
                             for u in paragraph["units"])
    item_id = f"{example.id}:{sid}"
    negative_pool = [u for _, _, u in positions if u["sid"] != sid and u["text"] != unit["text"]]
    if not negative_pool:
        raise ValueError("No distinct same-essay negative")
    negative = random.Random(f"{seed}:{item_id}:negative").choice(negative_pool)
    result = {**example.metadata(), "item_id": item_id, "source_id": example.id,
        "split": "agent_dev", "question": example.question, "essay_style": style,
        "deleted_sid": sid, "deleted_sentence": unit["text"],
        "paragraph_id": paragraph["pid"], "position": position,
        "corrupted_paragraph_with_gap": paragraph_text,
        "previous_sentence": positions[index - 1][2]["text"], "next_sentence": following["text"],
        "next_cross_paragraph": next_paragraph["pid"] != paragraph["pid"],
        "back_reference": back_reference(following["text"]),
        "negative_sid": negative["sid"], "negative_sentence": negative["text"],
        "negative_same_role": False, "negative_label_method": "same_essay_rule_no_judge",
        "origin": origin, "record": record}
    reconstruction_payload(result)
    return result


def round_robin_sites(pool, count, seed):
    """Spread supplemental sites over essays before taking a second site."""
    groups = defaultdict(list)
    for row in sorted(pool, key=lambda r: r["item_id"]):
        groups[row["source_id"]].append(row)
    rng = random.Random(seed)
    keys = sorted(groups)
    rng.shuffle(keys)
    for rows in groups.values():
        rng.shuffle(rows)
    result = []
    while len(result) < count:
        before = len(result)
        for key in keys:
            if groups[key] and len(result) < count:
                result.append(groups[key].pop())
        if before == len(result):
            raise ValueError("Insufficient dev deletion sites; no duplicate padding")
    return result


def prepare_cases(config, output, *, n=200, seed=43):
    if n != 200:
        raise ValueError("Authorized calibration requires exactly 200 deletion cases")
    output.mkdir(parents=True, exist_ok=True)
    metadata = config["paths"]["metadata"]
    sources_path = config["paths"]["phase3_output"] / "sources_agent_dev.json"
    corpus_path = metadata / "corrupt/agent_dev.jsonl"
    inputs = {str(p): file_sha(p) for p in (sources_path, corpus_path,
        metadata / "splits.json", metadata / "view_eligibility.json", config["paths"]["valid"])}
    examples = {e.id: e for e in load_episode_examples(config, "agent_dev")}
    selected_sources = read_json(sources_path)["rows"]
    cache = {}

    def document(source_id):
        if source_id not in cache:
            e = examples[source_id]
            data = read_json(config["paths"]["phase2_output"] / "bareun_profiles" / f"{e.source_line}.json")
            if data["essay_hash"] != e.essay_hash:
                raise ValueError("Stale source profile")
            doc = Document.from_profile(e.text, restore_profile(data["profile"]))
            cache[source_id] = doc, doc.structure().dominant_style
        return cache[source_id]

    existing, exclusions, raw_records = {}, [], 0
    corpus = sorted(read_jsonl(corpus_path), key=lambda r: (len(r["records"]), r["episode_id"]))
    for row in corpus:
        example = examples[row["source_id"]]
        if row["split"] != "agent_dev" or row["source_hash"] != example.essay_hash:
            raise ValueError("Corpus provenance mismatch")
        for record in row["records"]:
            if record["op"] != "G_DELETE_SUPPORT":
                continue
            raw_records += 1
            key = f"{example.id}:{record['sids'][0]}"
            if key in existing:
                continue
            try:
                existing[key] = make_case(example, record, document(example.id)[1], seed=seed,
                    origin={"kind": "final_dev_record", "episode_id": row["episode_id"],
                            "record_id": record["record_id"]})
            except ValueError as error:
                exclusions.append({"item_id": key, "reason": str(error)})
    chosen = list(existing.values())
    if len(chosen) > n:
        chosen = random.Random(seed).sample(sorted(chosen, key=lambda r: r["item_id"]), n)
    used_essays = {row["source_id"] for row in existing.values()}
    supplemental = []
    for source_id, eligibility in sorted(selected_sources.items()):
        if not eligibility["eligible"] or source_id in used_essays:
            continue
        example = examples[source_id]
        if eligibility["essay_hash"] != example.essay_hash:
            raise ValueError("Scored-source hash mismatch")
        doc, style = document(source_id)
        for proposal in candidates(doc, "G_DELETE_SUPPORT"):
            _, record = apply(doc, proposal, None)  # Global deletion needs no new Bareun call.
            try:
                supplemental.append(make_case(example, record, style, seed=seed,
                    origin={"kind": "supplemental_dev_deletion", "quality_selection": "section_4.3"}))
            except ValueError as error:
                exclusions.append({"item_id": f"{source_id}:{proposal.sids[0]}", "reason": str(error)})
    chosen += round_robin_sites(supplemental, n - len(chosen), seed)
    if len({row["item_id"] for row in chosen}) != n:
        raise ValueError("Duplicate calibration sites")
    manifest = {"phase": "4_B", "seed": seed, "n": n, "inputs_sha256": inputs,
        "selection": "existing unique dev sites, then other Section 4.3 dev sources in seeded round-robin",
        "existing_records": raw_records, "existing_unique_sites": len(existing),
        "supplemental_pool": len(supplemental), "exclusions": exclusions,
        "selected": [{k: row[k] for k in ("item_id", "source_id", "question_hash", "essay_hash", "origin")}
                     for row in chosen],
        "unique_essays": len({r["source_id"] for r in chosen}),
        "genres": dict(Counter(r["genre"] for r in chosen)),
        "cross_paragraph_next": sum(r["next_cross_paragraph"] for r in chosen),
        "negative_label_method": "same_essay_rule_no_judge; may contain semantic false negatives"}
    sample_path, manifest_path = output / "cases.jsonl", output / "sample_manifest.json"
    if sample_path.exists() and (read_jsonl(sample_path) != chosen or read_json(manifest_path) != manifest):
        raise ValueError("Refusing to change frozen calibration sample")
    write_jsonl(sample_path, chosen)
    write_json(manifest_path, manifest)
    deny = set(read_json(metadata / "audit_index.json")["train"]["essay_hashes"])
    assert_data_tree_clean(output, deny)
    return chosen, manifest
