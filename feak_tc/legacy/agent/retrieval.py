"""Optional train-only exemplar lookup; no index, embeddings or fitting required."""

import json

from feak_tc.diagnose.constants import RUBRIC_KEYS
from feak_tc.mvp.transition import lexical_similarity


class ExemplarStore:
    def __init__(self, path=None, *, top_k=2, min_score=7.0, exclude_similarity=0.8):
        self.rows = []
        self.top_k = top_k
        self.min_score = min_score
        self.exclude_similarity = exclude_similarity
        if path:
            with open(path, encoding="utf-8") as stream:
                for line in stream:
                    if not line.strip():
                        continue
                    row = json.loads(line)
                    if row.get("split") != "train":
                        raise ValueError("Exemplars must all have split=train")
                    if not all(isinstance(row.get(k), str) and row[k].strip()
                               for k in ("essay_id", "source_group", "text")):
                        raise ValueError("Exemplars need essay_id, source_group and text")
                    if not isinstance(row.get("rubrics"), dict) or any(
                        key not in RUBRIC_KEYS or isinstance(value, bool)
                        or not isinstance(value, (int, float)) or not 1 <= value <= 9
                        for key, value in row["rubrics"].items()
                    ):
                        raise ValueError("Exemplars need valid rubric scores on the 1–9 scale")
                    self.rows.append(row)

    def retrieve(self, original, weak_rubrics, *, essay_id=None, source_group=None):
        matches = []
        for row in self.rows:
            if row["essay_id"] == essay_id or row["source_group"] == source_group:
                continue
            similarity = lexical_similarity(original, row["text"])
            if similarity >= self.exclude_similarity:
                continue
            matching = [key for key in weak_rubrics if row["rubrics"].get(key, 0) >= self.min_score]
            if matching:
                matches.append((similarity, row["essay_id"], {
                    "essay_id": row["essay_id"], "text": row["text"], "rubrics": {
                        key: row["rubrics"][key] for key in matching
                    },
                }))
        return [item[2] for item in sorted(matches, key=lambda x: (-x[0], x[1]))[:self.top_k]]
