# CURRENT_TASK.md

## Task

Implement Phase 0 and Phase 1 skeleton.

## Goal

Create a clean FEAK-TC research repository with MongoDB connection, base schemas, and AI-Hub JSON ingestion.

## Implement These Files

- `pyproject.toml`
- `.env.example`
- `configs/default.yaml`
- `configs/rubric_schema.yaml`
- `configs/action_taxonomy.yaml`
- `feak_tc/db/mongo.py`
- `feak_tc/db/indexes.py`
- `feak_tc/db/repositories.py`
- `feak_tc/schemas/essay.py`
- `feak_tc/data/aihub_loader.py`
- `scripts/01_ingest_aihub.py`
- `tests/test_aihub_loader.py`

## MongoDB

Use environment variables:

- `MONGO_URI`
- `MONGO_DB`

Default database:

```text
feak_tc
```

Create indexes for collections:

- essays
- feak_observations
- expert_action_labels
- agent_runs
- agent_steps
- candidates
- transition_pairs
- tvm_training_examples

## AI-Hub Essay Normalization

Normalize each raw JSON into:

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

## Constraints

Do not implement:

- FEAK analyzer
- action proposer
- patch simulator
- transition value model
- controller loop
- frontend
- FastAPI
- model training

This task is only for clean project setup and data ingestion.

## Acceptance Criteria

- `pytest` passes
- `python scripts/01_ingest_aihub.py --input data/raw --limit 3` runs
- normalized records are inserted into MongoDB
- indexes are created
- ingestion script has useful CLI arguments
- loader handles missing optional fields gracefully

## After Implementation

Report:

- files changed
- commands run
- test results
- assumptions about AI-Hub JSON structure
- unresolved issues
