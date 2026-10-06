# Phase 6: decisions and baselines before training

This package implements the authorized Phase 6 experiments only. It neither selects
an agent architecture nor trains a model. Read `imple/VERAK_V3_SPEC_ADDENDUM_1.md`
before every phase; §9 constrains every role to content already in the essay.

Run with the `feak_agent` environment and runtime environment credentials:

```bash
python -m verak.v3.cli.phase6 --stage filter --max-api-calls 260
python -m verak.v3.cli.phase6 --stage prepare --max-api-calls 0
python -m verak.v3.cli.phase6 --stage run --max-api-calls 12000
python -m verak.v3.cli.phase6 --stage summarize --max-api-calls 0
python -m verak.v3.cli.phase6 --stage validate --max-api-calls 0
python -m verak.v3.cli.phase6 --stage report --max-api-calls 0
```

`run --limit 1` performs an integration check on the first frozen item of each
requested experiment. A subsequent unrestricted `run` reuses those results and
cached model responses. `--conditions` can select `two_stage`, `single`,
`check_once`, `real`, `rewrite_sol`, `rewrite_kanana`, or `local_link`.

- The original corpora remain unchanged. Every deletion record is judged once by
  Sol/high. Any false result excludes the whole essay without replacement. The
  active copies live under `data/corrupt_recoverable/`.
- `design.json` freezes corpus/prompt hashes, 60 distinct sources (15 per level,
  seed 53), 30 CHECK cases, 30 non-corruption real essays (10 per genre), and 100
  local-analysis sources. Gold answers never enter an agent or rewrite prompt.
- Sol/low supplies teacher actions and one-shot rewriting; separate Sol/high
  requests judge real-essay fabrication. CHECK-once is a prompt treatment, not an
  enforced action or extra hidden tool call. Real mode has no reward.
- Four API workers share a transactional SQLite ledger, a $35 cap and cost
  reservations. HTTP 5xx without usage cost zero; timeouts retain reservations.
  Retries are separate entries. Completed requests replay from cache. Local DNS
  resolution is checked before reserving a request; network-blocked attempts
  are documented separately from requests that reached the API transport.
- One dedicated GPU1 thread owns the frozen scorer, its SQLite connection and
  the calibrated BGE similarity model. GPU0 serves the untrained pinned Kanana
  base via the existing vLLM endpoint. No adapter is trained or loaded.
- One-shot outputs are aligned to the **corrupted input only** with lexical
  Hungarian assignment (threshold .35), then evaluated using the unchanged
  Phase 4 single-mode reward. New/deleted/split sentences can be hard to align;
  per-unit matches are saved. One generation counts as one step. No cleanup LLM,
  best-of selection, answer-based alignment, or synthetic action log is used.
- Changes are observations, not automatic semantic error labels. Changed-sentence
  share covers original units that were edited/deleted/reordered; insertion counts
  are separate. Explicit subject insertions require a new Bareun-marked phrase in
  an insertion diff. Conjunction and polarity/modality differences are likewise
  structural measurements. Raw per-action notices remain in trajectories.
- Confidence intervals resample **paired essays** within L1–L4 (10,000 samples,
  seed 61). A linguistic-level recovery is undefined, rather than zero, on an
  essay without records at that level. Local operators are applied independently
  to unmodified sources, only where their existing pattern applies.

Each experiment writes `metrics.json`, `report.md`, and local raw evidence under
`outputs/phase6/`. The phase report belongs at `imple/reports/V3_PHASE_6.md`.
Do not commit essays, judgments, request logs, credentials, or reports.
