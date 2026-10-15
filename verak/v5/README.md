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
`python -m verak.v5.cooperate_eval` waits for the inference completion file using
Linux file notifications, then runs judging and reporting. It does not poll the
GPUs or progress logs, and refuses to judge an incomplete GPU run.

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

For the corrected Luna Orchestrator comparison, use
`python -m verak.v5.orchestrator1 --no-search prepare`, followed by
`--no-search gpu`, `--no-search judge` and `--no-search report` with the configured
inference interpreter. This writes only to `outputs/orchestrator1_no_search`
and `reports/V5_ORCHESTRATOR1_NO_SEARCH.md`. Both the local Orchestrator control
and Luna condition run anew on the same C30 (60 episodes). Search-enabled
COOPERATE1/ORCHESTRATOR1 texts are historical references, not this control.

The controller defers `needs_search=yes` items to attributed writer notes
before worker dispatch. Mixed delegations retain their other tasks; entirely
deferred delegations create no worker stage or fabricated STOP. Search
availability is explicit in both Orchestrator and worker observations, the
backend is disabled, and no index preflight or retrieval runs. Saved episodes
from the opposite SEARCH setting cannot be reused. Every new episode is audited
for empty retrieved evidence and no search-dependent worker tasks before judging.
The actual FINISH stays in `finish_report`; top-level `writer_notes` also contains
controller deferrals, including after failures. Deferred work is unperformed,
even if all executed worker stages report done. Search removal does not validate
the semantics of unsourced EDIT/INSERT actions.

Worker adapters, system prompts, budgets and the final Korean pass are preserved.
Only the Orchestrator provider differs within this no-search pair. The same blind
whole-essay judge and reversed-order sensitivity check are used; no training or
test-cohort expansion. Prior confirmed and unknown API costs count toward the
existing combined $3 ceiling; this correction does not create a fresh $3 budget.

`recovery1` is a separate bounded Korean continuation pilot using the cleaned
SFT training split. Its initial 384 teacher-snapshot probes produced few actual
errors, so the scheduler was paused before training. `recovery1_rollout` adds
48 multi-step Korean stages and 16 full three-agent runs on training sources.
The latter use the student's actual Revision-to-Korean handoffs. Diagnostic,
validation and test sources never enter these collections.

An inherited teacher action is used only on the same document/tasks; conflicting
targets are excluded. Unmatched states require explicit saved Codex annotations
and successful action execution in the replayed real stage. This includes real
UNDO history, and never treats unavailable prior UNDO history as a model failure.
Saved student prefixes, not reconstructed rendering bytes, become training
inputs. This is corrective SFT with student-visited states, not rejection-sampling
RFT. Manual judgments are model annotations, not human gold.

`python -m verak.v5.recovery1_run start` starts a detached, resumable local job:
384 bounded collection seeds, 768 mixed corrective/replay rows, one Korean-only
continuation epoch, and one new run of the same 30 diagnostics. It preserves
the original adapters, all other roles, and COOPERATE1 runtime settings. It
refuses to train with fewer than 16 unique corrective states. The extended data
builder also requires at least eight failure observations. Its correction half
uses 288 failure slots and 96 independently matched student STOP states where
available; the replay half adjusts STOP sampling to retain about 30% STOP overall.
Use `start_rollouts` / `start_full_rollouts` with the owned collection server,
then `validate_reviews`, `build_rollout_data`, and `resume` after all collectors
have exited. The scheduler's `pause` leaves its collection children running.
The filtered
validation subset is evaluated before and after; its loss is not compared to
the different original export as though the inputs were identical.

`recovery1_eval prepare_evaluation` writes an unapproved proposal. Actual calls
require the separately approved `recovery1/api_contract.json`; the previous
experiment's budget cannot authorize them. `follow_start` waits on file events
for all 30 results, then grades anonymous complete essays under the new $3 cap.
Only original/final essays and source evidence reach the judge. Operations,
whole-essay preference, rubric scores, and source support remain separate. The
global rubric is an uncalibrated diagnostic and the new run is not a held-out
quality benchmark.

`orchestrator1` isolates the Orchestrator model on the same COOPERATE1 C30.
Only Orchestrator generation uses `gpt-6-luna` (reasoning none, temperature 0)
through Chat Completions with JSON-object mode for the single-action protocol.
Revision epoch1 and Korean epoch2 remain local, with unchanged prompts, public
observations, SEARCH, action budgets and context admission. Provider token counts
are recorded separately from the frozen Kanana tokenizer; worker template audits
stay strict. The first Orchestrator input must equal the saved control input.

Run `python -m verak.v5.orchestrator1 prepare`, then `gpu`, `judge`, and `report`
with the existing SEARCH interpreter. The new contract records the user's
approval of this experiment and an agent-set $3 ceiling ($1 Luna, $2 Claude),
separate from earlier experiments. The unconstrained format probe is preserved
under `orchestrator1`, and the Responses JSON probe under `orchestrator1_json`.
The full comparison uses `orchestrator1_chat` and deducts all confirmed and
reserved probe costs from the same ceiling. `gpu` owns and stops its local server.
`judge` compares complete anonymous final essays against saved COOPERATE1 run1, providing
the actual retrieved source passages. A reversed-order judgment measures order
sensitivity; it is not a second independent judge. Runs2/3 remain descriptive
controls. The report separates completion, delegation traces, dev10 recovery,
scorer changes and final-text quality. This experiment does not change defaults.

Tests:

```sh
python -m pytest -q verak/v5/tests verak/v4/tests/test_smoke45.py verak/v4/tests/test_fix1.py
python -m pytest -q
```
