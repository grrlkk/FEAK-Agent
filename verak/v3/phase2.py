"""Reproducible new-dev sampling and render artifacts, without gold or test data."""

from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
import random
import tempfile

from verak.src.schemas import Profile, Sentence, Token
from .common import file_sha, read_json, sha_text, write_json
from .data_policy import assert_data_tree_clean, load_examples, stratified_sample
from .ko import KoreanStructure, annotate, render, render_with_budget


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        temp = Path(handle.name)
    temp.replace(path)


def restore_profile(data):
    sentences = [Sentence(**{**sentence, "tokens": [Token(**token) for token in sentence["tokens"]]})
                 for sentence in data["sentences"]]
    return Profile(**{**data, "sentences": sentences})


def ec_row(example, annotation):
    morphs = [{"id": f"M{i + 1}", "form": token.form, "tag": token.tag,
               "span": [token.start - annotation.start, token.end - annotation.start],
               "surface": annotation.text[token.start - annotation.start:token.end - annotation.start],
               "highlight": token.tag == "EC"}
              for i, token in enumerate(annotation.tokens)]
    return {"sentence_id": f"{example.id}:{annotation.sid}", "sid": annotation.sid,
            "essay_id": example.id, "source_line": example.source_line,
            "question_hash": example.question_hash, "essay_hash": example.essay_hash,
            "genre": example.genre, "paragraph": annotation.paragraph,
            "sentence": annotation.text, "span": [annotation.start, annotation.end],
            "morphemes": morphs,
            "highlighted_morphemes": " ".join(
                f"**{m['id']}:{m['form']}/{m['tag']}**" if m["highlight"] else f"{m['id']}:{m['form']}/{m['tag']}"
                for m in morphs),
            "ec_tokens": [{**asdict(conn), "span": [p - annotation.start for p in conn.span]}
                          for conn in annotation.connectives],
            "llm_judgment_1": None, "llm_judgment_2": None, "llm_ok": None, "human_ok": None}


def sample_ec(pool, n=100, seed=23):
    ordered = sorted(pool, key=lambda row: (row["source_line"], int(row["sid"][1:])))
    if len(ordered) < n:
        raise ValueError("Not enough EC sentences in the new dev split")
    return random.Random(seed).sample(ordered, n)


def prepare(config, *, progress=print):
    examples = load_examples(config, "agent_dev")
    output = config["paths"]["phase2_output"]
    cache = output / "bareun_profiles"
    cache.mkdir(parents=True, exist_ok=True)
    ko = KoreanStructure.from_config(config)
    render_examples = stratified_sample(examples, 20, seed=23)
    if sum(example.genre == "논증" for example in render_examples) < 7:
        raise ValueError("Render review requires at least seven argumentative essays")
    review_ids = {example.id for example in render_examples}
    # No essay is truncated to fabricate a 2,500-character token-budget probe.
    probe = min(examples, key=lambda value: (abs(len(value.text) - 2500), value.source_line))
    structures, pool = {}, []
    deny = set(read_json(config["paths"]["metadata"] / "audit_index.json")["train"]["essay_hashes"])
    skipped_training_match = 0
    for i, example in enumerate(examples, 1):
        path = cache / f"{example.source_line}.json"
        if path.exists():
            cached = read_json(path)
            if cached["essay_hash"] != example.essay_hash:
                raise ValueError("Bareun profile cache does not match validation source")
            profile = restore_profile(cached["profile"])
        else:
            profile = ko.analyzer.profile(example.text)
            write_json(path, {"essay_hash": example.essay_hash, "profile": profile.to_dict()})
        structure = annotate(example.text, profile, ko.lexicons)
        if example.id in review_ids or example.id == probe.id:
            structures[example.id] = structure
        for ann in structure.annotations:
            if ann.connectives:
                if sha_text(ann.text) in deny:
                    skipped_training_match += 1
                    continue
                pool.append(ec_row(example, ann))
        if i % 100 == 0 or i == len(examples):
            progress(f"Bareun new-dev {i}/{len(examples)}; EC sentence pool={len(pool)}", flush=True)

    rows = sample_ec(pool, config["ec_judge"]["sample_size"], config["ec_judge"]["sample_seed"])
    ec_path = config["paths"]["metadata"] / "ec_check.jsonl"
    if ec_path.exists():
        existing = read_jsonl(ec_path)
        def base(row):
            return {k: v for k, v in row.items() if k not in {"llm_judgment_1", "llm_judgment_2", "llm_ok", "human_ok"}}
        if [base(row) for row in rows] != [base(row) for row in existing]:
            raise ValueError("Refusing to replace an existing EC sample with different inputs")
        rows = existing
    else:
        write_jsonl(ec_path, rows)

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(config["paths"]["policy_base"]), local_files_only=True)
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    render_dir = output / "renders"
    render_dir.mkdir(exist_ok=True)
    review = []
    for example in render_examples:
        structure = structures[example.id]
        full, compact = render(structure, compact=False), render(structure, compact=True)
        view = render_with_budget(structure, count_tokens)
        prefix = render_dir / str(example.source_line)
        prefix.with_suffix(".txt").write_text(full + "\n", encoding="utf-8")
        prefix.with_suffix(".compact.txt").write_text(compact + "\n", encoding="utf-8")
        write_json(prefix.with_suffix(".json"), structure.to_dict())
        review.append({**example.metadata(), "characters": len(example.text),
                       "sentences": len(structure.annotations), "dominant_style": structure.dominant_style,
                       "full_tokens": count_tokens(full), "compact_tokens": count_tokens(compact),
                       "selected_mode": view["mode"], "over_budget": view["over_budget"],
                       "max_line_chars": max(map(len, full.splitlines())), "render_path": str(prefix.with_suffix(".txt"))})
    probe_full = render(structures[probe.id], compact=False)
    probe_tokens = count_tokens(probe_full)
    # Check the closest actual essay and all reviewed essays of approximately 2,500 chars.
    ask = False  # Resolved by user: compact for every episode, overflow excludes the essay.
    manifest = {"split": "agent_dev", "split_sha256": file_sha(config["paths"]["metadata"] / "splits.json"),
                "valid_sha256": file_sha(config["paths"]["valid"]), "dev_essays": len(examples),
                "ec_pool_sentences": len(pool), "excluded_exact_train_sentence_matches": skipped_training_match,
                "seed": config["ec_judge"]["sample_seed"], "sample_size": len(rows),
                "sample_genres": dict(Counter(row["genre"] for row in rows)),
                "sample_ec_tokens": sum(len(row["ec_tokens"]) for row in rows),
                "sample_ids": [row["sentence_id"] for row in rows],
                "sample_inputs_sha256": sha_text(json.dumps([
                    {key: value for key, value in row.items() if not key.startswith("llm_") and key != "human_ok"}
                    for row in rows], ensure_ascii=False, sort_keys=True)),
                "render_review": review,
                "token_budget_probe": {**probe.metadata(), "characters": len(probe.text),
                    "full_tokens": probe_tokens, "tokenizer": str(config["paths"]["policy_base"]),
                    "approx_2500_characters": 2250 <= len(probe.text) <= 2750,
                    "ask_required": ask},
                "analyzer": "bareun", "analyzer_version": ko.analyzer.backend.version}
    write_json(output / "preparation.json", manifest)
    assert_data_tree_clean(config["paths"]["metadata"], deny)
    return manifest
