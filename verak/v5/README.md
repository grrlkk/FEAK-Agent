# V5 cooperation diagnostic

`cooperate` is an opt-in inference experiment on the frozen diagnostic cohorts.
It does not replace the default v4.6 runner, update prompts, or train adapters.

The integration wrapper sends complete public worker reports to the next worker
and Orchestrator, exposes the current document state, and lets the Orchestrator
review an automatic final Korean pass before FINISH. The original whole-essay
Korean scope, role restrictions, context limits, step budgets, and three-rejection
failure rule remain. Rejected-action feedback is keyed to the current state and
action; a rejection never bans other edits to that sentence.

Run `python -m verak.v5.cooperate prepare` to freeze code, checkpoints, public
inputs, saved FIX1 controls, and the explicitly authorized API contract. `gpu`
requires the existing SEARCH-enabled runtime, idle GPUs, local Bareun, and the
same GPU reference scorer. It generates only the new three repeated runs and
scores both the new and saved comparisons. It does not regenerate controls.
`judge` freezes blinded, full-context requests before using the approved Claude
budget. `report` separates protocol completion from scorer/judge quality. These
commands do not authorize a new budget: this experiment's contract reflects the
user's specific $5 approval and must not be reused for an unrelated experiment.

`clean_conn` consumes an explicit, hash-bound linguistic annotation file. It
creates separate source corpora and filtered SFT exports; it never edits saved
diagnostics, v1 rewards, original datasets, or existing adapters. An affected
corpus input loses stale scores/profiles/inverse snapshots and is marked as
requiring fresh preparation and GPU reference scoring. Old demonstrations on
that corruption episode are excluded across roles; other practice episodes of
the same source remain. Any later stage-3 data derivation must use the filtered
exports or enforce the saved episode exclusions. No retraining is launched here.

Experiment reports, source texts, outputs, annotations, and model artifacts stay
local under the configured repository artifact root and are not committed.

Tests:

```sh
python -m pytest -q verak/v5/tests verak/v4/tests/test_smoke45.py verak/v4/tests/test_fix1.py
python -m pytest -q
```
