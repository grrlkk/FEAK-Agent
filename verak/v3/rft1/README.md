# RFT round 1 and the subsequent one-shot comparison

This package executes the user-authorized first RFT round from the accepted
GLOBAL/KOREAN SFT epoch-2 adapters. It imports the unchanged v1 environment,
prompts, action protocol, observations and rewards. It does not authorize a
second round, DPO, v2 training or an Orchestrator.

The latest data-boost instruction adds a reference-scoring slot between rollout
completion and training. Ready v1 GLOBAL G_PARA_SWAP/G_SENT_MOVE teacher additions
are GPU-rescored, filtered under the RFT STOP/rejection rules and merged before
export. Insertion v2 data remains separate. Data not ready at that boundary waits
until RFT evaluation ends and cannot change this round's frozen training set.
All targets carry `sft_teacher`, `extra_teacher` or `rft_rollout` source tags.
GLOBAL full-recovery duplication applies per operator only below200 merged unique
train recoveries; L_CONJ duplication remains2x. The report separates operator and
source counts and attributes gains jointly to rollouts and any extra teachers.

Artifacts remain local under `verak/v3/outputs/phase8_rft1`; one-shot artifacts
go under `verak/v3/outputs/oneshot_baseline`. The original repository's pinned
paths are obtained from the existing SFT configuration. No dataset, model,
essay text or experiment report is committed.

## Collection and continuation

Use the `feak_agent` Python environment for:

```text
python -m verak.v3.cli.rft1 prepare
python -m verak.v3.cli.rft1 rollout
python -m verak.v3.cli.rft1 export
```

Collection requires GPU0 serving the accepted SFT epoch-2 pair using the existing
`serve_policy --sft-root` command; the worker owns the frozen GPU1 scorer.
All 1,430 active training episodes receive four samples at temperature .7 and
top_p .95. The frozen design records the corpus, runtime, prompts, adapters,
seed rule and all source-level evaluation exclusions. Episode files and call
caches are immutable. A saved failed generation is not silently resampled.

Selection takes the best eligible sample per essay and role, merges teacher
selections by higher role R, and preserves all SFT validation sources. Eligibility
requires R>=.80 for GLOBAL-record examples and KOREAN; the user-confirmed existing
no-GLOBAL STOP-within2/no-structural-attempt/R_over0 exception replaces GLOBAL's
reward threshold when no GLOBAL record exists. Every selected teacher or rollout
trajectory additionally needs a valid terminal STOP and at most one rejected
action. GLOBAL STOP-only training trajectories are capped at35%. Full recovery
of G_PARA_SWAP/G_SENT_MOVE receives total weight2 only below200 merged unique
training recoveries for that operator; full L_CONJ retains weight2 without
compounding. Data exports use the actual per-turn inference contexts and
mask everything except the current action and end-of-turn target.

The two-GPU trainer is launched with:

```text
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m verak.v3.cli.rft1 train --role global
python -m torch.distributed.run --standalone --nnodes=1 --nproc-per-node=2 \
  -m verak.v3.cli.rft1 train --role korean
```

Policy serving and scorer processes must be stopped first. Each rank places a
full NF4 base on its local GPU and continues the corresponding SFT adapter for
one epoch at lr5e-5. Micro-batch1 and accumulation4 on each rank retain effective
batch8. Target-token counts are summed across ranks and the accumulation window;
loss compensates DDP gradient averaging. Validation shards are disjoint and
token-weighted. Periodic optimizer checkpoints allow `--resume`; the final epoch
adapter and provenance are saved separately. Completed roles cannot be retrained
by this command.

## Evaluation and reports

`evaluate` runs the frozen SFT100dev/30real cohort with both RFT adapters, the
same8192/1024 context contract, deterministic temperature0/top_p1/seed47 and the
unchanged v1 environment. `markers` applies the accepted Sol final-marker check
with cached judgments and a shared A/C cap of$3. `report` writes the requested
RFT report after all130 evaluation attempts. `context-audit` only reads saved
histories and the pinned tokenizer; its proposal is never applied to evaluation.

`oneshot-export` is guarded by A's completion marker. It selects the higher
combined-R saved teacher attempt first, then applies the SFT role quality gates
to that same attempt; failure excludes the essay, without fallback. SFT held-out
sources remain validation-only. A fresh LoRA is trained for one epoch with the
SFT lr1e-4 recipe using `train --role oneshot`, again through two-rank torchrun.

One-shot evaluation uses the existing Phase6 rewrite prompt and record-blind
Hungarian lexical alignment to corrupted input (threshold.35). Whole-essay
generation allows up to4096 output tokens within the8192 context, while agent
actions keep their1024 limit. Combined R uses the existing ONE_SHOT step
convention. Tool validity, tool STOP and intermediate role rewards are N/A;
they are not fabricated. Use `--condition oneshot` with `evaluate`, `markers`
and `report` for C. There are no new teacher calls.

## Durable orchestration

`launch-rollouts` and `launch-controller` are host/background launch commands.
They must run in an execution context that preserves child processes; a transient
sandbox may reap detached children. The controller's `continue` stage waits for
the owned rollout worker, validates all5,720 attempts, exports, trains GLOBAL then
KOREAN, evaluates and reports A, and only then considers C. An explicit
`{"hold": true}` in `phase8_rft1/oneshot_hold.json` makes the controller exit at
`a_complete_c_on_hold` without starting any C subprocess. This hold leaves A
unchanged and remains in force until the user explicitly resumes C. A controller
already waiting for rollouts can load this guard with `restart-controller`, which
restarts only the waiting manager and leaves collection and serving untouched.
Without a hold, it stops its own servers and exits after C.
Ownership requires matching PID, process group and pinned
model command; unrelated GPU processes are never terminated.

`status.json`, `rollout_progress.json`, stage logs and `controller/steps.jsonl`
are the durable progress sources. A failed stage stops advancement and records
the exact error; inspect it before resuming. The independent v2 retry is CPU/API
only and does not share this training pipeline or its budget ledger.

`rft1.coverage.snapshot` reads a frozen list of saved rollout files without calls
or reward recomputation. Its primary denominator contains only essays with all
four samples saved. A success requires every record of that operator to be fully
recovered within one sample; separate fields report main recovery and complete
recovery including coupled changes. Unobserved rewards remain unknown with
coverage bounds, and immutable file hashes identify the snapshot.
