# Weak-GLOBAL teacher expansion

This data-only pipeline supplements G_PARA_SWAP and G_SENT_MOVE while RFT round 1
continues. It never starts training, policy serving, or a GPU scorer. The root
controller later obtains GPU reference scores in an authorized RFT gap; this
component consumes those saved results without making GPU calls. The KOREAN
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
measured files add provisional v1 GLOBAL rewards from corrupted/stage-1 scores on
the shared frozen NF4 CPU scorer. KOREAN and combined rewards are unmeasured;
their final text is not scored for this GLOBAL-only selection. CPU arithmetic
has separate cache provenance and is compared with GPU scores on at least 200
distinct source essays. CPU rewards and eligibility remain internal. Every
teacher episode subsequently gets GPU-reference quality inputs; final rewards
and selection use GPU values only. No fabricated quality score is used.
Selection imports the actual
SFT `absolute_selection` rule and takes the best qualifying GLOBAL attempt per
practice; ties prefer attempt 1. Source IDs are retained for future validation
grouping. This task does not train or make a validation split.

Exact saved GPU score-cache hits may be reused after the entire frozen scorer
fingerprint matches; the background task never invokes a GPU. The CPU service
preserves the frozen NF4/double-quantized weights and records its arithmetic
contract separately. FP32 execution is allowed for provisional measurement only.
`cpu_ready.json` freezes the raw episodes, provisional measurements and quality
input manifest. The root schedules GPU scoring after rollouts and before RFT
training if GLOBAL is ready, otherwise after RFT evaluation. `gpu-finalize`
consumes `gpu_rescore/global_complete.json` and immediately writes GPU-only
`gpu_selection.json` for the root's conditional RFT merge. It does not wait for
the >=200-essay audit; the final report does. CPU and GPU measurements use distinct
source/fingerprint directories. An explicitly failed GPU input remains an error
and never falls back to CPU.
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

The worker hides CUDA, runs at nice 19 on CPUs 104–111, and uses two independent
teacher workers. Existing frozen single-worker manifests remain intact; runtime
concurrency is recorded separately. Both boosts share cache-first Bareun access at no more than one new
request per second. Busy/timeout or new RFT Bareun errors suspend background
cache misses, and elevated recent latency downgrades GLOBAL to one worker. A
separate CPU model service serves both boosts sequentially.

Run from the isolated worktree:

```bash
python -m verak.v3.cli.global_boost prepare
python -m verak.v3.cli.global_boost launch
python -m verak.v3.cli.global_boost audit-variants
python -m verak.v3.cli.global_boost launch-expansion
python -m verak.v3.cli.global_boost launch-finalizer
```

Individual resumable stages are `qc`, `teacher`, `measure`, and `report`.
`continue` runs generation and provisional CPU measurement only; it exports no
CPU-based final selection/report. `restart` reloads the data controller
only after its locked ledger confirms no pending API request; saved responses
replay and no GPU process is touched. Artifacts are under
`verak/v3/outputs/data_boost/global` in the configured original repository.
`best_global_trajectories.json`, `component_metrics.json` and
`component_report.md` feed the combined `V3_DATA_BOOST.md` report. No data,
credentials, model weights or generated report is committed.

The finalizer freezes CPU readiness, consumes the GPU pass, waits for the
>=200-essay audit, and writes aggregate artifacts plus `complete.json` with their
hashes. That marker requires no live paid calls, accounted terminal outcomes,
completed available measurements, and either all requested teacher slots or an
explicit budget stop. A code PR or a launched process is not task completion.

When question, rendered essay, and scorer contract serialize to identical input
bytes, the provisional quality delta is exactly zero without scoring. Absolute
before/after Q stay null, and identity hashes prove the shortcut. The GPU manifest
still includes those inputs. All non-quality reward terms and SFT gates are
unchanged.
# Approved v4 preparation continuation

The cumulative GLOBAL cap is now $40, including all historical spending. The old
42-source expansion was drained under its ledger lock; raw trajectories and excess
practices remain archived. `v4-compare` reuses the exact saved92 Sol/Luna cohort.
Sol exceeds Luna by23.08pp on G_SENT_MOVE only, enabling one Sol-low GLOBAL rescue
when both completed Luna attempts failed full recovery of that operator.

`v4-prepare` freezes new Phase3b practices from all eligible agent_train sources,
including active-corpus sources, while excluding frozen SFT/evaluation holdouts and
every saved structural position. New source coverage is prioritized. The400/operator
planning quota counts historical practices only up to4/source/operator; excess raw
history is reported separately. Every new practice includes a versioned source/position
proof and its immutable prior-position index hash.

`v4-launch` runs the new balanced QC/two-Luna-attempt batches and then conditional
rescue. It uses the existing atomic ledger, shared low-rate Bareun access, CPU affinity
and hidden CUDA setting. Sol rescue uses4096 API output tokens for GLOBAL only;
KOREAN remains Luna low/1024. The policy still observes the original v1 prompt/context
and learns action JSON only under8192 context/1024 target tokens.

The existing `cpu-ready`, `gpu-finalize`, and `launch-finalizer` interfaces remain.
Final selections require GPU R>=.80, valid terminal STOP and<=1 rejected action;
best attempts are chosen per practice, then a deterministic GPU-R ranking retains
at most4 practices/source/operator. The published GPU selection and final report
both use that same capped set. CPU scores never rank this cap or enter final rewards.
Completion waits for root-owned GPU rescoring and the>=200-source compatibility audit.
`v4/component_metrics.json` and `v4/component_report.md` contain the A report snippet;
the existing root component paths and completion marker remain compatible.
