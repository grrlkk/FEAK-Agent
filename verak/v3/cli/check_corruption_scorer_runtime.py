"""Check unchanged expected scores after reducing host CPU thread oversubscription."""

import time
from ..common import load_config, read_json, write_json
from ..corrupt.sources import select_sources
from .prepare_corruption_sources import phase3_scorer


def main():
    config = load_config()
    examples, _ = select_sources(config, "agent_dev")
    old = read_json(config["paths"]["phase3_output"] / "sources_agent_dev.json")["rows"]
    selected, seen = [], set()
    for example in examples:
        if example.genre not in seen:
            selected.append(example)
            seen.add(example.genre)
    selected.extend(e for e in examples if e not in selected and len(e.text) >= 1200)
    selected = selected[:6]
    scorer, rows = phase3_scorer(config), []
    try:
        for example in selected:
            before = time.monotonic()
            result = scorer.score(example.question, example.text, use_cache=False)
            expected = old[example.id]["score"]
            row = {"source_id": example.id, "genre": example.genre, "cpu_threads": 1,
                   "seconds": time.monotonic()-before, "expected_identical": result.expected == expected["expected"],
                   "score_line_identical": result.score_line == expected["score_line"]}
            rows.append(row)
            print(row, flush=True)
            if not row["expected_identical"] or not row["score_line_identical"]:
                raise ValueError("Scheduling change failed exact scorer equivalence")
    finally:
        scorer.close()
        write_json(config["paths"]["phase3_output"] / "scorer_runtime_equivalence.json", rows)


if __name__ == "__main__":
    main()
