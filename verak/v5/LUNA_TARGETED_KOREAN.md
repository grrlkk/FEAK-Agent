# Korean-only predecessor-change probe

Read the ten frozen `simple_train10/episodes/three_agent/*/result.json` artifacts.
Use each `original` and **`revision`** field; preserve the existing `final` and
all source files. No Orchestrator/Revision generation or full-essay assembly runs.

Split with the existing deterministic sentence splitter. Match sentences in any
order, maximizing the number of one-to-one pairs with SequenceMatcher similarity
>= 0.9, then their total similarity. Keep surface spaces and punctuation. Select
pairs whose immediately preceding sentence differs, across paragraph boundaries;
an absent predecessor is the empty string. This intentionally includes trivial
predecessor changes. No marker/phi filter or Bareun analysis is added.

Each selected pair gets one independent Luna call with only the old predecessor,
old sentence, current predecessor, current sentence, and the user's question.
Use reasoning none, temperature 0, the existing cumulative $5 ledger and four
concurrent calls. Do not cascade suggestions into later observations. Persist raw
responses and proposals without overwriting the essays; report format errors
without automatically retrying. A response identical to the current sentence is
counted as keep, while retaining the raw answer.

```bash
python -m verak.v5.luna_targeted_korean prepare
python -m verak.v5.luna_targeted_korean run
python -m verak.v5.luna_targeted_korean report
python -m pytest tests/test_luna_targeted_korean.py -q
```

Local artifacts: `verak/v5/outputs/luna_upper_bound_v2/targeted_korean_train10`.
Optional `review.json` contains per-case `assessment`/`reason` from qualitative
inspection, and `interpretation.md` supplies the local findings. These reviews
are Codex assessments, not human labels or additional model-judge API calls.
Reports: `reports/luna_targeted_korean.md` and the full UTF-8 CSV
`reports/luna_targeted_korean_sentences.csv`. Raw essays, results and reviews stay
local. Validation50, final test and training remain untouched.
