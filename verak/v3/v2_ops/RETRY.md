# Dependency-first G_DEL_LINK retry

The separate `config_v2_retry.yaml` overlay is still method version `v2`.
The accepted v1 environment and initial v2 experiment stay unchanged. All retry
artifacts use `verak/v3/outputs/v2_ops_retry/`; passing corpus exports use
`verak/v3/data/corrupt_ops_v2_retry/`. The initial v2 output is read only.

The source pool, paragraph-first/last-paragraph sites, source-level SFT evaluation
exclusions and 300 train / 80 dev request are retained. Before Sol role filtering,
cached Bareun tokens require an immediate-successor dependency hint. Supported
hints are the requested anaphors, `그/이 + noun`, `그렇다면/그러면`, an explicitly
introduced ordinal, or a question followed by an answer-like next paragraph.
The last two use conservative, documented lexical tests; independent unchanged
Sol damage/repair/recoverability QC still decides whether the deletion is usable.
Cached role labels and identical deletion judgments are reused, including rejects.
The chosen deletion never depends on its prior QC verdict.

L_FUSE is disabled in the retry: 338 of its 377 previously judged merges were
acceptable alternatives. The 30% yield denominator remains all 380 requested
sources. Role-filter failures and source shortages count as unusable; missing
API judgments stay unknown and give lower/upper yield bounds.

```bash
python -m verak.v3.cli.v2_ops_retry prepare --config v2
# Current task explicitly authorized the new, separate $10 paid budget:
python -m verak.v3.cli.v2_ops_retry qc --config v2 --max-api-calls 20000 --paid-approval Proceed
python -m verak.v3.cli.v2_ops_retry teacher --config v2 --max-api-calls 20000 --paid-approval Proceed
python -m verak.v3.cli.v2_ops_retry report --config v2
# Optional detached continuation waits for an already-owned QC PID, resumes safely,
# and schedules the teacher only when the yield lower bound reaches 30%:
python -m verak.v3.cli.v2_ops_retry continue --config v2 --wait-pid PID --max-api-calls 20000 --paid-approval Proceed
# Final checks; no paid calls:
python -m verak.v3.cli.v2_ops_retry audit --config v2
python -m verak.v3.cli.v2_ops_retry report --config v2 --final
```

The CLI hides CUDA, uses nice 10 or lower priority, limits execution to the last
eight available CPU cores, and sets math-library threads to one. The shared retry
reservation ledger allows at most two paid requests. New Bareun requests wait
while task A's top-level status is `rollouts` or `evaluating*`; otherwise they run
one at a time at no more than one request per second. Cached analyses and API
calls can continue independently. No retry command controls GPU services.

When the gate passes, teacher generation uses the same pinned Luna low, no-CHECK
two-stage policy, 8,192/1,024 context/output contract, two independent unseeded
attempts and best-per-essay/role selection. No GPU scorer is available in this
CPU/API-only command. If a future retained run needs quality scores, exact CPU
scorer consistency must be established separately before final role selection;
missing quality scores never become zero or change the selection threshold.

Reports go to `imple/reports/V3_V2_OPS_RETRY.md`. Missing recovery/quality scores
are unknown, not zero. No v2 training command exists. Saved inputs, API outputs,
failures, reservations and attempts remain immutable on resume.
