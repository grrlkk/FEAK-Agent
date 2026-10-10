"""Validation-only agent input; corpus hashes are used only for leakage checks."""

from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path
import random
import re

from .common import Example, extract_question_essay, file_sha, read_json, sha_text, write_json

GENRES = ("설명", "논증", "정서", "기타")
HEADING_SETS = {
    "설명": {"과제 수행의 충실성", "설명의 명료성", "설명의 구체성", "설명의 적절성",
           "문장의 연결성", "글의 통일성", "어휘의 적절성", "어법의 적절성"},
    "논증": {"과제 수행의 충실성", "주장의 명료성", "근거의 타당성", "주장의 적절성",
           "문장 및 문단의 연결성", "글의 통일성", "어휘 및 문장의 적절성", "어법의 정확성"},
    "정서": {"과제 수행의 충실성", "주제 전달 및 정서 표현의 명료성",
           "주제 전달 및 정서 표현의 구체성", "주제 전달 및 정서 표현의 적절성",
           "문장 및 문단의 연결성", "글의 통일성", "어휘 및 문장의 적절성", "어법의 정확성"},
}


def feedback_headings(assistant: str) -> tuple[str, ...]:
    return tuple(sorted(re.findall(r"(?m)^\s*-\s*([^:\n]+):", assistant)))


def split_questions(question_hashes, *, seed=13, train_fraction=0.8, genres=None):
    values = sorted(set(question_hashes))
    if len(values) < 2 or not 0 < train_fraction < 1:
        raise ValueError("Need two or more questions and a fraction in (0, 1)")
    rng = random.Random(seed)
    groups = {"ALL": values}
    if genres is not None:
        if set(values) != set(genres) or not set(genres.values()) <= set(GENRES):
            raise ValueError("Genre labels must cover every question exactly")
        groups = {genre: [q for q in values if genres[q] == genre]
                  for genre in GENRES if genre in genres.values()}
    train, dev = [], []
    for group in groups.values():
        if len(group) < 2:
            raise ValueError("Need at least two questions in each genre")
        rng.shuffle(group)
        count = max(1, min(len(group) - 1, round(len(group) * train_fraction)))
        train.extend(group[:count])
        dev.extend(group[count:])
    return {"agent_train": sorted(train), "agent_dev": sorted(dev)}


def resplit_by_genre(config):
    """Replace the split only, retaining its original bytes; never read train/test."""
    directory = config["paths"]["metadata"]
    path, backup = directory / "splits.json", directory / "splits_v1.json"
    original = read_json(path)
    examples = load_examples(config, "agent_train") + load_examples(config, "agent_dev")
    labels = read_json(directory / "genres.json")["questions"]
    row_counts = Counter(e.question_hash for e in examples)
    split = split_questions(row_counts, **config["split"],
                            genres={q: labels[q]["genre"] for q in row_counts})
    assert_split_integrity(split, row_counts)
    counts = {name: {"questions": len(values), "essays": sum(row_counts[q] for q in values),
        "genres_questions": dict(Counter(labels[q]["genre"] for q in values)),
        "genres_essays": dict(sum((Counter({labels[q]["genre"]: row_counts[q]}) for q in values), Counter()))}
        for name, values in split.items()}
    if original.get("method") == "genre_stratified_question":
        if not backup.exists() or any(original[key] != split[key] for key in split) or original["counts"] != counts:
            raise ValueError("Existing stratified split or its backup differs")
        return original
    if backup.exists() and backup.read_bytes() != path.read_bytes():
        raise ValueError("Refusing to replace a different splits_v1 backup")
    if not backup.exists():
        with backup.open("xb") as handle:
            handle.write(path.read_bytes())
    manifest = {**original, **split, **config["split"], "schema_version": 2,
        "method": "genre_stratified_question", "genre_order": list(GENRES),
        "previous_split_sha256": file_sha(backup), "counts": counts}
    write_json(path, manifest)
    return manifest


def assert_split_integrity(splits, all_questions=None):
    train, dev = splits["agent_train"], splits["agent_dev"]
    if len(train) != len(set(train)) or len(dev) != len(set(dev)) or set(train) & set(dev):
        raise ValueError("Question leakage or duplicate question hash in agent splits")
    if not train or not dev:
        raise ValueError("Empty agent split")
    if all_questions is not None and set(train) | set(dev) != set(all_questions):
        raise ValueError("Question coverage does not match validation data")


def assert_not_training_essay(text: str, deny_hashes):
    if sha_text(text) in deny_hashes:
        raise ValueError("Exact scorer-training essay found in v3 data")


def assert_data_tree_clean(directory, deny_hashes):
    """Examine all JSON/JSONL string values, not just a preferred essay key."""
    def check(value):
        if isinstance(value, str):
            assert_not_training_essay(value, deny_hashes)
        elif isinstance(value, list):
            for item in value:
                check(item)
        elif isinstance(value, dict):
            for item in value.values():
                check(item)
    for path in Path(directory).rglob("*"):
        if path.is_file() and path.suffix == ".json":
            check(read_json(path))
        elif path.is_file() and path.suffix == ".jsonl":
            with path.open(encoding="utf-8") as handle:
                for line in handle:
                    if line.strip():
                        check(json.loads(line))


def audit_source_hashes(path: Path):
    """Read-only metadata/exclusion audit. Never yields a training/test example."""
    questions, essays, rows = Counter(), set(), 0
    with Path(path).open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            question, essay = extract_question_essay(json.loads(line))
            questions[sha_text(question)] += 1
            essays.add(sha_text(essay))
            rows += 1
    return {"path": str(Path(path).resolve()), "sha256": file_sha(path), "rows": rows,
            "question_counts": dict(sorted(questions.items())), "essay_hashes": sorted(essays)}


def prepare_metadata(config, *, phase0_audit, classify=None):
    """Audit train/test as hashes only; labels and split contents come from valid only."""
    paths = config["paths"]
    directory = paths["metadata"]
    if (directory / "splits.json").exists() and read_json(directory / "splits.json").get("method") == "genre_stratified_question":
        raise ValueError("Phase 1b split is already prepared; refusing legacy preparation")
    previous = read_json(phase0_audit)
    train_path = paths["repo"] / "data/data_jsonl/train.jsonl"
    test_path = paths["repo"] / "data/data_jsonl/test.jsonl"
    training = audit_source_hashes(train_path)
    testing = audit_source_hashes(test_path)
    for name, value in (("train", training), ("test", testing)):
        if value["sha256"] != previous["files"][name]["sha256"]:
            raise ValueError("Source file differs from the approved Phase 0 audit")
    valid_sha = file_sha(paths["valid"])
    if valid_sha != previous["files"]["valid"]["sha256"]:
        raise ValueError("Validation source differs from the approved Phase 0 audit")
    deny = set(training["essay_hashes"])
    questions, headings, row_counts = {}, defaultdict(Counter), Counter()
    with paths["valid"].open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            question, essay = extract_question_essay(row)
            assert_not_training_essay(essay, deny)
            qhash = sha_text(question)
            questions[qhash] = question
            row_counts[qhash] += 1
            headings[qhash][feedback_headings(row.get("assistant", ""))] += 1
    labels, heading_audit = {}, {}
    for qhash in sorted(questions):
        sets = headings[qhash]
        genre = None
        if len(sets) == 1:
            chosen = next(iter(sets))
            if len(chosen) == 8:
                genre = next((key for key, expected in HEADING_SETS.items() if set(chosen) == expected), None)
        method = "offline_feedback_heading_set"
        if genre is None:
            if classify is None:
                raise ValueError("Inconsistent/unknown rubric headings require a budgeted question classifier")
            genre = classify(qhash, questions[qhash])
            method = "gpt_question_classification"
        if genre not in GENRES:
            raise ValueError("Invalid genre label")
        labels[qhash] = {"genre": genre, "method": method, "rows": row_counts[qhash],
                         "distinct_heading_sets": len(sets)}
        heading_audit[qhash] = [{"headings": list(key), "rows": count} for key, count in sorted(sets.items())]
    split = split_questions(questions, **config["split"])
    assert_split_integrity(split, questions)
    counts = {name: {"questions": len(values), "essays": sum(row_counts[q] for q in values),
                    "genres_questions": dict(Counter(labels[q]["genre"] for q in values)),
                    "genres_essays": dict(sum((Counter({labels[q]["genre"]: row_counts[q]}) for q in values), Counter()))}
              for name, values in split.items()}
    exclusions = {}
    for pair, info in previous["feak_split_overlap"].items():
        if "test" not in pair:
            continue
        other = pair.replace("test", "").strip("_")
        for item in info["duplicate_essay_locations"]:
            for number in item["test"]:
                exclusions.setdefault(number, {"row": number, "essay_sha256": item["essay_sha256"], "duplicates_with": []})["duplicates_with"].append(other)
    write_json(directory / "eval_exclusions.json", {
        "schema_version": 1, "test_path": str(test_path), "test_sha256": testing["sha256"],
        "rule": "Phase 9 must skip these 1-based rows; exact essay duplicate with train or valid",
        "valid_exclusions": [], "rows": [exclusions[k] for k in sorted(exclusions)]})
    write_json(directory / "audit_index.json", {
        "usage": "Hashes only, for leakage exclusion and seen-by-scorer metadata; never agent input",
        "train": training, "test": {key: value for key, value in testing.items() if key != "essay_hashes"},
        "valid_sha256": valid_sha})
    provenance = {}
    for name, qcounts in (("valid", row_counts), ("test", testing["question_counts"])):
        provenance[name] = {q: {"seen_by_scorer": q in training["question_counts"], "rows": qcounts[q]}
                            for q in sorted(qcounts)}
    write_json(directory / "question_provenance.json", provenance)
    write_json(directory / "genres.json", {"source": "valid.jsonl only", "source_sha256": valid_sha,
        "allowed_labels": list(GENRES), "questions": labels})
    write_json(directory / "genre_audit.json", {"offline_only": True, "questions": heading_audit,
        "distinct_heading_sets_per_question": dict(Counter(len(x) for x in headings.values()))})
    manifest = {"schema_version": 1, "source": str(paths["valid"]), "source_sha256": valid_sha,
                "question_hash": "sha256(exact extracted UTF-8 question)", **config["split"],
                "counts": counts, **split}
    existing_split = directory / "splits.json"
    if existing_split.exists() and read_json(existing_split) != manifest:
        raise ValueError("Refusing to replace an existing split; use the explicit resplit CLI")
    write_json(existing_split, manifest)
    assert_data_tree_clean(directory, deny)
    return manifest


def load_examples(config, split="agent_dev"):
    if split not in ("agent_train", "agent_dev"):
        raise ValueError("Only validation-derived agent splits are available before Phase 9")
    paths = config["paths"]
    manifest = read_json(paths["metadata"] / "splits.json")
    assert_split_integrity(manifest)
    if paths["valid"].resolve() != Path(manifest["source"]).resolve():
        raise ValueError("Agent source path must be the prepared validation file")
    if file_sha(paths["valid"]) != manifest["source_sha256"]:
        raise ValueError("Validation source changed")
    genres = read_json(paths["metadata"] / "genres.json")["questions"]
    provenance = read_json(paths["metadata"] / "question_provenance.json")["valid"]
    deny = set(read_json(paths["metadata"] / "audit_index.json")["train"]["essay_hashes"])
    chosen, result = set(manifest[split]), []
    with paths["valid"].open(encoding="utf-8") as handle:
        for number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            question, text = extract_question_essay(json.loads(line))
            qhash = sha_text(question)
            if qhash in chosen:
                assert_not_training_essay(text, deny)
                result.append(Example(number, question, text, qhash, sha_text(text),
                    genres[qhash]["genre"], provenance[qhash]["seen_by_scorer"]))
    return result


def stratified_sample(examples, n, seed=13):
    """Fixed-seed round robin across present genres, independent of scorer results."""
    examples = list(examples)
    if not isinstance(n, int) or n < 1 or n > len(examples):
        raise ValueError("Not enough eligible examples")
    groups = defaultdict(list)
    for example in examples:
        groups[example.genre].append(example)
    rng = random.Random(seed)
    for genre in sorted(groups):
        values = groups[genre]
        values.sort(key=lambda item: item.source_line)
        rng.shuffle(values)
    result = []
    while len(result) < n:
        for genre in sorted(groups):
            if groups[genre] and len(result) < n:
                result.append(groups[genre].pop())
    return result
