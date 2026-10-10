# v4.2 scale collection

These commands prepare and collect B2/B3 only. They do not start training, a CPU
quality scorer, a GPU scorer, or an automatic post-A job. Old v1/v4 outputs and
the active corpus remain immutable. The shared environment and policy prefixes
are the exact frozen `v4.2_editors_20261010` contract used by B1.

## Launch gate and budgets

Do not launch until the parent explicitly approves collection after B1 PASS.
Both `content-freeze` and `corruption-freeze` also fail closed unless
`scale2/B1/gate.json` passes, its completion/metrics hashes agree, and both smoke
ledgers have no pending or reserved calls. Contracts freeze those proofs.

- B2 freezes a cap of $50 minus the confirmed v4.1 and v4.2 smoke costs. Planner,
  two teacher attempts, and paired judging share this one durable ledger.
- B3 has a separate $30 durable ledger. It uses Luna only, with record-derived
  delegations and no new planner or quality-judge calls.
- Both use pinned Luna low with two independent unseeded attempts. Ordering seeds
  are recorded separately and do not claim to seed model sampling.
- Responses, request identities, per-turn partial traces, final attempts and
  failures are preserved. A saved failed/uncertain request is not sent again.
  Budget exhaustion stops new episode dispatch and drains in-flight work. All
  planned denominators and uncollected attempts remain in reports.

From this checkout, after the explicit launch authorization:

```bash
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v4.scale2 content-freeze
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v4.scale2 corruption-freeze
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v4.scale2 content-run
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v4.scale2 corruption-run
```

Run the last two commands in separate supervised processes. The CLI hides CUDA,
limits library threads to one, uses nice 19 and disjoint CPU affinities: 104–107
for B2 and 108–111 for B3. Bareun cache misses share the existing cross-process
lock and at most one new request per second. Cached profiles are reused.

The first 20 actually started teacher attempts of each collection use at most
two episode workers. Healthy saved API outcomes allow the default B2=6/B3=4
ramp, at most ten in-flight paid calls overall. A rate-limit/timeout outcome
holds that component at two. Each worker makes sequential calls within an
episode; changing concurrency never changes context, turn order or sampling.

The parent may atomically write `scale2/dispatch_policy.json` to pause new
episodes when A needs Bareun, or reduce workers without interrupting a request:

```json
{
  "pause_new_episodes": false,
  "paused_components": {"B2": false, "B3": false},
  "max_workers": {"B2": 6, "B3": 4},
  "disable_ramp": false
}
```

`--root PATH` uses an isolated output root with its own B1 proof. `--limit N`
processes only the first N frozen sources/practices for a bounded authorized
invocation; it neither changes the full denominator nor publishes a final B3
GPU-ready manifest. Resume the same command without the limit to continue.

## B2 data and export

The sample contains 1,000 new train essays, with genre quotas 334/333/333.
Sampling uses the mean of all 16 stored human-grader values on the 1–5 scale,
the lower two-thirds cutoff within each genre, and recorded ordering seeds.
Prior PREP1/2/3/map/smoke, active v3 train/dev and extra-teacher sources are
excluded by ID and normalized-text hash. The question inventory separately
records the 62 canonical test-split questions, the 42 present in the frozen
100-test cohort, and 23 additional v3-dev questions: 85 conservative exclusions,
with zero missing canonical questions. Counts, cutoffs and every prior manifest
hash are frozen before collection. A failed source is not replaced.

Each essay is normalized once before its Sol plan is generated. Planner, teacher
and judge receive the same prepared baseline. The frozen Dv3 procedure is reused:
Revision delegations of at most two tasks, six actions per delegation, at most
two successful INSERTs per attempt, followed by the Korean whole-essay pass with
14 actions. The standard 8,192 context and 1,024 teacher output limit remain.

Selection uses the six Dv3 judgment conditions: no invented specifics or
experiences, preserved meaning, better essay, at least half the assigned items
fully addressed, no introduced repetition, no introduced awkwardness. Both
passing attempts are retained. No additional terminal-STOP gate is imposed.
Only valid action targets receive loss; public observations/tool outputs/notices
are masked, and privileged feedback/scores never enter the editor prefix.

Outputs are under `scale2/B2`: `sample.json`, `essays/`, `content/items/`,
`content/attempts/`, `content/partial/`, `content/judgments/`, `content/export/`,
`metrics.json`, `report.md`, and `collection_finished.json`.

## B3 projection and final GPU reward

All 1,430 active practices (2,860 attempts) retain their original records and
source IDs. Stable IDs are privately mapped to aliases in current reading order
before the policy sees the corrupted input, hiding original order/positions.
Public tasks contain locations, action names and instructions only, never
record IDs, original answers, recovery targets or the inverse mapping.
No-GLOBAL essays receive an explicit whole-essay structure review; the Korean
agent always receives its whole-essay pass.

Original and normalized layouts, token offsets, initial character ranges,
dynamic split lineage, actual initial/stage-1/final text and raw actions are
saved. Complete, adjacent, in-order fragment groups can be projected exactly
to their original stable sentence. Partial deletion, separated fragments,
reversed fragments and cross-paragraph groups have **unknown** reward and cannot
be selected. All practices remain in the denominator. No approximate recovery
definition or zero-filled unknown reward is introduced.

The normalized source and corrupted baseline keep automatic whitespace changes
out of agent `R_over`. Records recovered by normalization are reported per
operator separately from subsequent model recovery. Revision EDIT split attempts
are structural attempts even if rejected or undone. Raw v4 actions stay intact;
a separate private copy maps DELETE/INSERT and stable IDs to v1 attribution.

Collection saves mechanical recovery and over-edit terms separately, with Q,
total reward and selection null. It never calls a CPU quality scorer. Once raw
collection has stopped, `gpu_manifest.json`/`gpu_ready.json` freeze all raw and
prepared hashes and the minimal distinct actual normalized Q inputs. They are
ready files only; they dispatch no work. Exact saved GPU cache hits are allowed,
but old raw-corpus Q values are never silently substituted.

The parent schedules GPU reference scoring **after A evaluation** and writes:

```json
{
  "manifest_sha256": "sha256 of B3/gpu_manifest.json",
  "fingerprint": "the frozen reference GPU scorer fingerprint",
  "responses_root": "/absolute/path/to/responses",
  "slot": "post_A_evaluation",
  "scheduled_by": "root"
}
```

at `scale2/B3/gpu_rescore/complete.json`. Each `{key}.json` response contains
`execution_device: "gpu_reference"`, that same `fingerprint`, and
`result: {cache_key: key, mean: ...}` from the unchanged scorer. The pair key is
the existing exact question/text key. Missing/failed/incompatible responses
remain unknown; there is no CPU fallback.

Only then explicitly run the file-only consumer:

```bash
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v4.scale2 corruption-gpu-finalize
```

It verifies the frozen reward code/weights, input hashes and reference identity,
uses unchanged v1 arithmetic, and chooses the best attempt per role and practice
at R≥0.80, valid terminal STOP and at most one rejected action. The existing
no-GLOBAL exception is retained: STOP within two steps, no structural attempt,
R_over=0. No RFT 35% STOP cap, 2x operator duplication or new source cap applies.
`gpu_selection.json` and action-only `export/` are immutable and idempotent.
Original source IDs remain on exports for any later source-grouped split.

Final B3 reports give GPU-only role rewards and per-operator own-role main
recovery, observed/unknown/uncollected coverage, selected trajectories/sources,
and normalization-only recovery. `content-report` and `corruption-report` are
file-only progress commands; neither starts collection or scoring.

## Cost estimate and verification

`cost-estimate` reads the saved 100-source Dv3 run, which included plans, 200
teacher attempts and paired judgments. Its $4.39749 extrapolates to about
$43.97 for B2 before smoke costs. The prior v1 two-stage 2,860-attempt collection
cost about $17.73 including reused runs; B3's v4.2 action count can differ.
These are planning estimates. Durable reservation-aware caps take precedence.

Offline tests exercise public/private IDs through the actual Dv3 teacher loop,
split/UNDO/handoff, all nine active operators against unchanged v1 rewards,
unknown projections, exact normalized GPU endpoints, no CPU fallback, action
masking, source exclusions, budget gates, bounded draining and immutable resume.
They use fake services or saved local tokens and make no model/GPU calls.
