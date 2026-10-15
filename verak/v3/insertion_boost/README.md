# GLOBAL insertion boost

This independent `config v2` experiment keeps the original retry, v2 and v1
artifacts immutable. Its source plan is exactly the 200 remaining dependency-
feasible agent_train sources in the frozen retry plan. Teacher inputs contain the
78 prior passing train records plus newly passing train records. The revised
operator gate is prior-plus-new QC passes / judged records >= 30%.

The original Sol label/QC contracts are reused. Paid incomplete outcomes remain
unknown and are never sampled again. A separate aggregate reservation ledger
covers all new Sol/Luna calls with a $6 cap. Only proven pre-send DNS failures
with an empty ledger and no request files can be archived for a same-input retry.

Luna uses the pinned low model, two independent unseeded attempts, original v2
two-stage observations/actions, 8192/1024 context and no CHECK. KOREAN executes
normally but only GLOBAL rewards and best per-source GLOBAL selections are
produced. Selection uses the original SFT R >= .80 rule, not later RFT gates.

The fast FP32 CPU approximation is **provisional only**. Its initial material
drift and the closer but slower BF16-wrapper engineering probe remain separate.
The later user requirement mandates GPU reference scoring for every teacher
episode before final rewards or selections. CPU and GPU provenance never share
a cache fingerprint. A frozen 200-source audit uses existing GPU trajectories:
600 states, genre-proportional quotas, seeded selection, with a declared near-.80
stress stratum. It reports mean/max Q error, generated and argmax digit agreement
separately, and both role threshold changes. Empirical errors are not universal
bounds. All final GLOBAL rewards use GPU reference values, even if the audit
has no eligibility flips; missing GPU values are excluded without CPU fallback.

`cpu_ready.json` identifies the immutable `gpu_rescore_manifest.json` after paid
collection and provisional CPU measurements. Only the root controller starts
GPU scoring, in its authorized RFT boundary or after evaluation. `gpu-finalize`
reads those local GPU results and writes `gpu_selection.json` immediately,
without waiting for the CPU audit. Final report completion additionally requires
both GPU components and the >=200-source audit; the shared approval artifact is
published once and hash-verified. These v2 insertion selections remain separate
from round 1. No GPU model, GPU service call or training occurs in this module.

Both data tasks share one CPU score queue and a cache-first Bareun lock. Misses
are serialized at at most one per second, with a pause throughout task A
inference after a new busy/timeout or task-A Bareun error. Teacher execution uses
one insertion worker, nice19 and disjoint CPU affinity. GLOBAL CPU requests have
priority over insertion and audit work so they can feed the authorized RFT gap.
Root-owned RFT settings
and process priorities are never changed.

Entry point: `python -m verak.v3.cli.insertion_boost STAGE --config v2`.
Stages: `prepare`, `qc`, `teacher`, `score`, `gpu-finalize`, `report`, `serve_cpu`,
`schedule_cpu`, `prepare_audit_200`, `audit_200`, `continue`. Engineering probe
stages remain available for provenance. Paid stages additionally
require `--paid-approval Proceed --max-api-calls N`; the approval flag records
the authorization already supplied by the user, rather than requesting another
confirmation. `continue` launches a resumable host supervisor; run it outside
an ephemeral sandbox. `report --final` requires no requests in flight.

Local outputs live under `outputs/data_boost/insertion`; the component report is
`component_report.md` and selected trajectories are `best_role_trajectories.json`.
The root agent owns the combined `imple/reports/V3_DATA_BOOST.md` report. Data,
API records, results and model weights are never committed.
