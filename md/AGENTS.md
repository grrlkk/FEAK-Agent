# AGENTS.md

## Project Identity

This repository implements **FEAK-TC: FEAK-guided Transition Control for Open-ended Korean Writing Agents**.

This is an **Agent research implementation**, not a web application and not a FEAK-web refactor.

The project studies how a Writing Agent can control an iterative revision trajectory by deciding:

- what revision action to try
- whether the revision transition is beneficial
- whether the revision damages other parts of the essay
- whether to reject, rollback, or stop

## Core Research Principle

The main contribution is **trajectory-level control**, not feedback generation.

Do not turn this into:

- a chatbot UI
- a frontend product
- a classroom dashboard
- a generic LLM writing assistant
- a direct modification of the legacy UKTA_v2 web service

## Method Focus

The method must remain centered on the following loop:

```text
Observe -> Propose -> Simulate -> Evaluate -> Control -> Repeat
```

Where:

- **Observe**: FEAK Diagnoser produces diagnostic signals.
- **Propose**: LLM generates stratified action candidates.
- **Simulate**: each action is applied as a patch candidate.
- **Evaluate**: a Transition Value Model or heuristic scorer evaluates the transition.
- **Control**: controller accepts, rejects, rolls back, or stops.

## No Feature Creep Rule

Do not add new modules, signals, or models unless they directly support one of these:

- action selection
- step-level transition evaluation
- regression detection
- rollback
- stopping

Do not add RAG, NLI, long-term memory, multi-agent teams, web search, style/voice models, or full semantic content-unit extraction unless explicitly requested.

## Compact Transition Feature Rule

For the initial method, transition evaluation must use a compact feature set:

- action_type
- target_rubric
- target_gain
- target_gap_reduction
- non_target_drop
- evidence_match
- edit_ratio

These are the approved transition features for v0/v1.

## Legacy Code Rule

The old `UKTA_v2` repository is legacy reference only.

- Do not modify `UKTA_v2` directly.
- Do not copy the entire old backend.
- Do not reuse old MongoDB credentials.
- Only wrap or port the FEAK analysis components that are needed.

Expected legacy components:

- feature extraction
- rubric scoring
- elite benchmark gap computation
- priority feature selection
- evidence sheet generation

## Database Rule

Use MongoDB, not SQLite.

Use database name:

```text
feak_tc
```

Expected collections:

- essays
- feak_observations
- expert_action_labels
- agent_runs
- agent_steps
- candidates
- transition_pairs
- tvm_training_examples

Raw AI-Hub JSON files should remain on disk under `data/raw`. MongoDB should store normalized records and experiment logs.

## Data Rule

Normalize AI-Hub essay records into a stable schema before using them.

Expected normalized fields:

- essay_id
- prompt
- topic
- grade
- purpose
- text
- features
- rubric_scores_raw
- rubric_scores_mean
- expert_feedback
- rubric_definitions
- raw_path

## Expert Feedback Rule

Expert feedback is **not the face of the method**.

Use expert feedback only as:

- auxiliary supervision
- calibration signal
- validation anchor

Do not frame the project as simply imitating expert feedback.

## Revision Patch Rule

Do not rewrite whole essays unless explicitly requested.

All candidate revisions must be represented as patches:

- operation
- target_span
- before
- after
- reason

This is required for rollback and trajectory logging.

## LLM Output Rule

All LLM-generated outputs must be valid JSON and must be schema-validated.

Never assume LLM output is valid.

## Testing Rule

Every new module should have a smoke test or pytest test.

Do not mark a task complete unless the relevant command runs.

## Reporting Rule

After each implementation task, report:

- files created or changed
- commands run
- tests run
- assumptions made
- unresolved issues
