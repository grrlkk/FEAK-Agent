"""Select high-quality source essays for corruption chain generation.

Implements the selection decided in
feak_tc_docs/중간정리/CORRUPTION_GEN_DECISIONS_2026-07-22.md:
top ~25% by two-grader average, question-stratified, length-filtered,
with a held-out evaluation pool split off before any generation.

Outputs (data/corruption/, gitignored — rerun with the fixed seed to
reproduce):
  heldout_200.jsonl  essays reserved for TVM evaluation, never corrupted
  pilot_50.jsonl     pilot chain generation
  main_1000.jsonl    main run pool
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
STAGE_A_LOG = Path("experiments/results/mvp_stage_a_100_bge_m10.jsonl")
OUT_DIR = Path("data/corruption")
TOP_QUANTILE = 0.25
MIN_SENTENCES = 6
MIN_CHARS, MAX_CHARS = 300, 2500
QUOTAS = {"heldout_200": 200, "pilot_50": 50, "main_1000": 1000}


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


def load_candidates() -> list[dict]:
    rows, seen = [], set()
    with TRAIN_PATH.open() as f:
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
                    "record_id": f"train_{idx}",
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


def main() -> None:
    rng = random.Random(SEED)
    stage_a = load_stage_a_ids()
    candidates = load_candidates()
    pool = top_quantile_per_question(candidates)
    overlap = {r["record_id"] for r in pool} & stage_a
    # Stage A essays feed data source ② (real LLM edit pairs); keeping them
    # out of every corruption pool avoids any train/eval entanglement later.
    pool = [r for r in pool if r["record_id"] not in stage_a]
    rng.shuffle(pool)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    remaining = pool
    print(f"candidates after filters: {len(candidates)}")
    print(f"top-{TOP_QUANTILE:.0%} pool: {len(pool)} (Stage A overlap excluded: {len(overlap)})")
    for name, quota in QUOTAS.items():
        drawn = stratified_draw(remaining, quota, rng)
        drawn_ids = {r["record_id"] for r in drawn}
        remaining = [r for r in remaining if r["record_id"] not in drawn_ids]
        path = OUT_DIR / f"{name}.jsonl"
        with path.open("w") as f:
            for row in drawn:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")
        questions = len({r["question"] for r in drawn})
        avg = sum(r["grader_avg"] for r in drawn) / len(drawn)
        print(f"{path}: {len(drawn)} essays, {questions} questions, grader_avg mean {avg:.2f}")


if __name__ == "__main__":
    main()
