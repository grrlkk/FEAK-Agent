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

Tests:

```sh
python -m pytest -q verak/v5/tests verak/v4/tests/test_smoke45.py verak/v4/tests/test_fix1.py
python -m pytest -q
```
