# Weak-GLOBAL teacher expansion

This data-only pipeline supplements G_PARA_SWAP and G_SENT_MOVE while RFT round 1
continues. It never starts training, policy serving, or a GPU scorer. The KOREAN
prompt, operators, permissions, observations and hand-off are the existing v1
implementation. KOREAN runs as the second teacher stage, but only GLOBAL
trajectories are selected.

Source eligibility is exactly Phase 3b: view-eligible `agent_train`, genre human
score upper quartile and 500–2,500 characters, with an existing valid source
score. Sources used by either active corpus or either fixed evaluation cohort
are excluded by ID and essay hash. Each practice has one record. Distinct
corruption variants, within or across operators, may use the same unused source;
the report distinguishes source diversity from practice counts. Each added batch
contains at most 25 practices per operator, selected in balanced source order.

Generation calls the existing Phase 3b proposal, apply, inverse and view checks.
Sol QC uses the unchanged high-reasoning prompt/schema and requires
`damage_real && original_is_fix`. Every passing practice receives two independent
pinned-Luna low attempts, with order seeds 71 and 72 and no model sampling seed.
The v1 8,192/1,024 token contract and no-CHECK environment remain unchanged.

Because rewards are invisible without CHECK, the original v1 action loop can
finish before scoring. Raw generation files remain immutable; separately saved
measured files add the exact v1 GLOBAL reward from corrupted/stage-1 scores on
the shared frozen NF4 CPU scorer. KOREAN and combined rewards are unmeasured;
their final text is not scored for this GLOBAL-only selection. CPU arithmetic
has separate cache provenance and is compared with saved
GPU scores. No fabricated quality score is used. Selection imports the actual
SFT `absolute_selection` rule and takes the best qualifying GLOBAL attempt per
practice; ties prefer attempt 1. Source IDs are retained for future validation
grouping. This task does not train or make a validation split.

Exact saved GPU score-cache hits may be reused after the entire frozen scorer
fingerprint matches; the background task never invokes a GPU. The CPU service
preserves the frozen NF4/double-quantized weights and records its arithmetic
contract separately. The initial FP32 execution showed a material discrepancy
and is not canonical. Final selection waits for a validated
`cpu_scorer/selection_approval.json`, uses that fingerprint only, and saves
`measured_<fingerprint-prefix>` outputs without reusing earlier FP32 measurements.
Each score records its execution device and fingerprint. `launch-prefetch` queues only corrupted/stage-1 states
of already saved episodes while teacher collection continues, sharing the
idempotent file queue with the insertion boost.

All Sol and Luna sends use one SQLite reservation ledger with a $12 aggregate
cap. Completed calls replay on restart; interrupted/failed sent calls are never
silently regenerated. Proven pre-send environment/DNS failures may be archived
and attempted after preflight succeeds. Raw requests, API retries, confirmed
cost and uncertain reservations are retained.

The initial source/teacher plans are immutable. Expansion batches live under
`batches/batch_NNN/` and share the initial ledger. Before each batch, conservative
observed costs reserve projected funds for both Luna attempts, assuming every new
QC candidate passes. QC reservations also protect those funds. Batch-specific
interruption reconciliation never settles a concurrent original-teacher call.
The hard per-call reservation cap applies even if actual usage exceeds a cost
projection. Reports distinguish a projection/cap stop from source supply.

The worker hides CUDA, runs at nice 19 on CPUs 104–111, and uses one teacher
worker. Both boosts share cache-first Bareun access at no more than one new
request per second. Busy/timeout or new RFT Bareun errors suspend background
cache misses. A separate CPU model service serves both boosts sequentially.

Run from the isolated worktree:

```bash
python -m verak.v3.cli.global_boost prepare
python -m verak.v3.cli.global_boost launch
python -m verak.v3.cli.global_boost audit-variants
python -m verak.v3.cli.global_boost launch-expansion
python -m verak.v3.cli.global_boost launch-finalizer
```

Individual resumable stages are `qc`, `teacher`, `measure`, and `report`.
`continue` runs those stages sequentially. `restart` reloads the data controller
only after its locked ledger confirms no pending API request; saved responses
replay and no GPU process is touched. Artifacts are under
`verak/v3/outputs/data_boost/global` in the configured original repository.
`best_global_trajectories.json`, `component_metrics.json` and
`component_report.md` feed the combined `V3_DATA_BOOST.md` report. No data,
credentials, model weights or generated report is committed.

The finalizer waits for all generation batches and the approved scorer, measures
GLOBAL only, and writes aggregate artifacts plus `complete.json` with their
hashes. That marker requires no live paid calls, accounted terminal outcomes,
completed available measurements, and either all requested teacher slots or an
explicit budget stop. A code PR or a launched process is not task completion.
