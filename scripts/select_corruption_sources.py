"""Select high-quality source essays for corruption chain generation.

Implements the selection decided in
feak_tc_docs/중간정리/CORRUPTION_GEN_DECISIONS_2026-07-22.md:
top ~25% by two-grader average, question-stratified, length-filtered,
with a held-out evaluation pool split off before any generation.

The kanana scorer was trained on train.jsonl, so pilot/main pools come
from train.jsonl (in-distribution for the scorer; per-step discard
checks guard measurement validity) while the held-out evaluation pool
comes from test.jsonl — essays the scorer never saw — so the TVM
generalization claim stays clean.

Outputs (data/corruption/, gitignored — rerun with the fixed seed to
reproduce):
  heldout_200.jsonl  scorer-unseen essays reserved for TVM evaluation
  pilot_50.jsonl     pilot chain generation (train.jsonl)
  main_1000.jsonl    main run pool (train.jsonl)
"""

import json
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from feak_tc.diagnose.stub import split_sentences

SEED = 20260722
TRAIN_PATH = Path("data/data_jsonl/train.jsonl")
TEST_PATH = Path("data/data_jsonl/test.jsonl")
STAGE_A_LOG = Path("experiments/results/mvp_stage_a_100_bge_m10.jsonl")
OUT_DIR = Path("data/corruption")
TOP_QUANTILE = 0.25
MIN_SENTENCES = 6
MIN_CHARS, MAX_CHARS = 300, 2500
TRAIN_QUOTAS = {"pilot_50": 50, "main_1000": 1000}
HELDOUT_QUOTA = 200


def load_stage_a_ids() -> set[str]:
    ids = set()
    if STAGE_A_LOG.exists():
        with STAGE_A_LOG.open() as f:
            for line in f:
                ids.add(json.loads(line)["record_id"])
    return ids


def _scores(value) -> list[float]:
    scores = json.loads(value) if isinstance(value, str) else value
    if not isinstance(scores, list) or not scores:
        raise ValueError("bad grader scores")
    return [float(v) for v in scores]


def load_candidates(path: Path, id_prefix: str) -> list[dict]:
    rows, seen = [], set()
    with path.open() as f:
        for idx, line in enumerate(f):
            raw = json.loads(line)
            user = raw.get("user", "")
            if "에세이:" not in user:
                continue
            head, essay = user.split("에세이:", 1)
            question, essay = head.replace("질문:", "").strip(), essay.strip()
            if not essay or essay in seen:
                continue
            seen.add(essay)
            try:
                g1 = _scores(raw["grader_1_scores"])
                g2 = _scores(raw["grader_2_scores"])
            except (KeyError, ValueError, TypeError):
                continue
            if not (MIN_CHARS <= len(essay) <= MAX_CHARS):
                continue
            if len(split_sentences(essay)) < MIN_SENTENCES:
                continue
            rows.append(
                {
                    "record_id": f"{id_prefix}_{idx}",
                    "question": question,
                    "text": essay,
                    "grader_avg": round((sum(g1) / len(g1) + sum(g2) / len(g2)) / 2, 3),
                }
            )
    return rows


def top_quantile_per_question(rows: list[dict]) -> list[dict]:
    by_q = defaultdict(list)
    for row in rows:
        by_q[row["question"]].append(row)
    pool = []
    for rows_q in by_q.values():
        rows_q.sort(key=lambda r: -r["grader_avg"])
        keep = max(1, int(len(rows_q) * TOP_QUANTILE))
        pool.extend(rows_q[:keep])
    return pool


def stratified_draw(pool: list[dict], quota: int, rng: random.Random) -> list[dict]:
    """Round-robin across questions so every topic contributes."""
    by_q = defaultdict(list)
    for row in pool:
        by_q[row["question"]].append(row)
    for rows_q in by_q.values():
        rng.shuffle(rows_q)
    drawn, questions = [], sorted(by_q)
    while len(drawn) < quota and any(by_q[q] for q in questions):
        for q in questions:
            if by_q[q] and len(drawn) < quota:
                drawn.append(by_q[q].pop())
    return drawn


def write_pool(name: str, drawn: list[dict]) -> None:
    path = OUT_DIR / f"{name}.jsonl"
    with path.open("w") as f:
        for row in drawn:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    questions = len({r["question"] for r in drawn})
    avg = sum(r["grader_avg"] for r in drawn) / len(drawn)
    print(f"{path}: {len(drawn)} essays, {questions} questions, grader_avg mean {avg:.2f}")


def main() -> None:
    rng = random.Random(SEED)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Held-out evaluation pool: scorer-unseen essays from test.jsonl.
    test_pool = top_quantile_per_question(load_candidates(TEST_PATH, "test"))
    rng.shuffle(test_pool)
    print(f"test top-{TOP_QUANTILE:.0%} pool: {len(test_pool)}")
    write_pool("heldout_200", stratified_draw(test_pool, HELDOUT_QUOTA, rng))

    # Pilot/main pools: train.jsonl (scorer in-distribution).
    stage_a = load_stage_a_ids()
    pool = top_quantile_per_question(load_candidates(TRAIN_PATH, "train"))
    overlap = {r["record_id"] for r in pool} & stage_a
    # Stage A essays feed data source ② (real LLM edit pairs); keeping them
    # out of every corruption pool avoids any train/eval entanglement later.
    pool = [r for r in pool if r["record_id"] not in stage_a]
    rng.shuffle(pool)
    print(f"train top-{TOP_QUANTILE:.0%} pool: {len(pool)} (Stage A overlap excluded: {len(overlap)})")
    remaining = pool
    for name, quota in TRAIN_QUOTAS.items():
        drawn = stratified_draw(remaining, quota, rng)
        drawn_ids = {r["record_id"] for r in drawn}
        remaining = [r for r in remaining if r["record_id"] not in drawn_ids]
        write_pool(name, drawn)


if __name__ == "__main__":
    main()
