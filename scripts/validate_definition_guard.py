"""Out-of-sample check of the DOF definition guard on unseen train.jsonl rows."""
import json
import random
import sys
from collections import defaultdict

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parents[1]))
from feak_tc.diagnose.stub import split_sentences
from feak_tc.mvp.targeting import (
    _protected_definition_indices,
    _targeting_config,
    rank_target_spans,
)

random.seed(20260722)

used = set()
with open("experiments/results/mvp_stage_a_100_bge_m10.jsonl") as f:
    for line in f:
        used.add(json.loads(line)["record_id"])

records = []
seen_essays = set()
with open("data/data_jsonl/train.jsonl") as f:
    for idx, line in enumerate(f):
        rid = f"train_{idx}"
        if rid in used:
            continue
        user = json.loads(line)["user"]
        if "에세이:" not in user:
            continue
        head, essay = user.split("에세이:", 1)
        question = head.replace("질문:", "").strip()
        essay = essay.strip()
        if not essay or essay in seen_essays:
            continue
        seen_essays.add(essay)
        records.append({"id": rid, "question": question, "essay": essay})

by_q = defaultdict(list)
for rec in records:
    by_q[rec["question"]].append(rec)
print(f"unseen unique essays: {len(records)}, distinct questions: {len(by_q)}")

sample = []
per_q = max(1, 400 // len(by_q))
for q, rows in sorted(by_q.items()):
    sample.extend(random.sample(rows, min(per_q, len(rows))))
random.shuffle(sample)
sample = sample[:400]
print(f"sampled: {len(sample)} across {len({r['question'] for r in sample})} questions\n")

cfg = _targeting_config()
penalty = float(cfg["definition_penalty"])
n_sent = 0
stats = {"noq": 0, "withq": 0, "flip_noq": 0, "flip_withq": 0}
flagged = []

for rec in sample:
    sents = split_sentences(rec["essay"])
    n_sent += len(sents)
    for mode, q in (("noq", None), ("withq", rec["question"])):
        prot = _protected_definition_indices(sents, q, cfg)
        stats[mode] += len(prot)
        if not prot:
            continue
        if mode == "withq":
            for i in sorted(prot):
                flagged.append((rec["id"], rec["question"], sents[i]))
        # would the pre-guard DOF top-1 have been a protected sentence?
        ranked = rank_target_spans(rec["essay"], "content_3", action_type="DELETE_OR_FOCUS",
                                   limit=len(sents), question=q)
        pre = max(ranked, key=lambda r: (r["score"] + (penalty if r["index"] in prot else 0.0),
                                         -r["index"]))
        if pre["index"] in prot:
            stats["flip_" + mode] += 1

print(f"sentences total: {n_sent}")
print(f"protected sentences  (question=None): {stats['noq']}  ({100*stats['noq']/n_sent:.2f}%)")
print(f"protected sentences  (with question): {stats['withq']}  ({100*stats['withq']/n_sent:.2f}%)")
print(f"essays where guard changes DOF top-1 (question=None): {stats['flip_noq']}/{len(sample)}")
print(f"essays where guard changes DOF top-1 (with question): {stats['flip_withq']}/{len(sample)}")
print(f"\n=== flagged sentences (with-question mode, {len(flagged)}) ===")
for rid, q, s in flagged:
    print(f"[{rid}] Q: {q[:40]}\n    {s[:120]}")
