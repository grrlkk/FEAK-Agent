# Weak-GLOBAL teacher expansion

This data-only pipeline supplements G_PARA_SWAP and G_SENT_MOVE while RFT round 1
continues. It never starts training, policy serving, or a GPU scorer. The KOREAN
prompt, operators, permissions, observations and hand-off are the existing v1
implementation. KOREAN runs as the second teacher stage, but only GLOBAL
trajectories are selected.

Source eligibility is exactly Phase 3b: view-eligible `agent_train`, genre human
score upper quartile and 500–2,500 characters, with an existing valid source
score. Sources used by either active corpus or either fixed evaluation cohort
are excluded by ID and essay hash. Each practice has one record; distinct
operators may use the same unused source. Insufficient source supply is reported
without changing the selection policy.

Generation calls the existing Phase 3b proposal, apply, inverse and view checks.
Sol QC uses the unchanged high-reasoning prompt/schema and requires
`damage_real && original_is_fix`. Every passing practice receives two independent
pinned-Luna low attempts, with order seeds 71 and 72 and no model sampling seed.
The v1 8,192/1,024 token contract and no-CHECK environment remain unchanged.

Because rewards are invisible without CHECK, the original v1 action loop can
finish before scoring. Raw generation files remain immutable; separately saved
measured files add exact v1 reward calculations from the shared frozen NF4 CPU
scorer. CPU arithmetic has separate cache provenance and is compared with saved
GPU scores. No fabricated quality score is used. Selection imports the actual
SFT `absolute_selection` rule and takes the best qualifying GLOBAL attempt per
practice; ties prefer attempt 1. Source IDs are retained for future validation
grouping. This task does not train or make a validation split.

All Sol and Luna sends use one SQLite reservation ledger with a $12 aggregate
cap. Completed calls replay on restart; interrupted/failed sent calls are never
silently regenerated. Proven pre-send environment/DNS failures may be archived
and attempted after preflight succeeds. Raw requests, API retries, confirmed
cost and uncertain reservations are retained.

The worker hides CUDA, runs at nice 19 on CPUs 104–111, and uses one teacher
worker. Both boosts share cache-first Bareun access at no more than one new
request per second. Busy/timeout or new RFT Bareun errors suspend background
cache misses. A separate CPU model service serves both boosts sequentially.

Run from the isolated worktree:

```bash
python -m verak.v3.cli.global_boost prepare
python -m verak.v3.cli.global_boost launch
```

Individual resumable stages are `qc`, `teacher`, `measure`, and `report`.
`continue` runs those stages sequentially. Artifacts are under
`verak/v3/outputs/data_boost/global` in the configured original repository.
`best_global_trajectories.json`, `component_metrics.json` and
`component_report.md` feed the combined `V3_DATA_BOOST.md` report. No data,
credentials, model weights or generated report is committed.
