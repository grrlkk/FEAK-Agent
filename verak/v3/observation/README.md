# Observation design comparison

This is an explicitly bounded design comparison, not a training or held-out evaluation
entry point. It uses the accepted Pilot-2 92 **agent_train** corruptions and 30 Phase-6
real agent_dev essays. The prior Luna-low corruption runs are immutable setting (a).

Run with `OPENAI_API_KEY` and `BAREUN_API_KEY` in the process environment:

```bash
python -m verak.v3.cli.observation_test prepare --max-api-calls 0
python -m verak.v3.cli.observation_test graphs --max-api-calls 12000
python -m verak.v3.cli.observation_test agents --max-api-calls 12000
python -m verak.v3.cli.observation_test spot --max-api-calls 12000
python -m verak.v3.cli.observation_test markers --max-api-calls 12000
python -m verak.v3.cli.observation_test report --max-api-calls 0
```

All paid stages share one SQLite reservation ledger and the **$8** cap. The call ceiling
does not override the cost cap. Cached successful or failed episode files are not silently
regenerated. New teacher actions use Luna low / 1,024 output tokens and the shared
8,192 context scheme; graph extraction and Sol structured judgments have their own
larger response caps. No bulk generation, SFT, or RFT is exposed here.

- `current`: unchanged prompts, profile, notices and hand-off.
- `text_only`: hide profile and notices; keep only GLOBAL actions in the hand-off.
- `graph`: frozen Luna discourse edges, stable public IDs, dynamically rebuilt Bareun
  clause spans/marker edges, graph-change notices. Invalid discourse graphs block the
  corresponding graph episode. They are not repaired with another generation.

Role names use a fixed seven-label vocabulary. Clause nodes are surface spans ending at
Bareun EC/ETM/ETN/JKQ tokens, not a full dependency parse. Cross-paragraph edges and
isolated nodes are facts, not error labels. A deleted sentence's outgoing edges disappear;
incoming edges become dangling. UNDO restores the frozen edges when their nodes return.

Quality checks use active dev corruptions with relevant GLOBAL records, plus two
independent extractions for 100 dev sources and Sol judgments for 50 dev sources. The
Responses API does not expose a seed argument; the manifest records that stability is
two independent requests with identical prompts, not controlled model sampling seeds.

Final marker cases are reconstructed from saved full/partial observations. Every valid
structural action remains in the denominator; targets come from actual predecessor
changes, insertions and internal Bareun notices regardless of policy visibility. Sol sees
only the final predecessor and target sentence. Missing/incomplete outcomes stay unknown.

Outputs and the report remain local under `outputs/observation_test/` and
`imple/reports/V3_OBSERVATION_TEST.md`. No essay data or result artifacts belong in Git.
