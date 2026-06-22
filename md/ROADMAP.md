# ROADMAP.md

## Phase 0. Clean Project Setup

### Goal

Create a clean Python research repo for FEAK-TC.

### Tasks

- create package structure
- create config files
- create MongoDB connection module
- create Pydantic schemas
- create tests

### Success Criteria

- project imports correctly
- pytest runs
- MongoDB connection can be initialized

## Phase 1. AI-Hub Data Ingestion

### Goal

Load AI-Hub Korean writing JSON records into MongoDB.

### Tasks

- parse raw JSON files
- normalize essay prompt, text, features, rubric scores, expert feedback, rubric definitions
- store normalized records in `essays`
- create MongoDB indexes

### Success Criteria

- `scripts/01_ingest_aihub.py --input data/raw --limit 3` runs
- inserted records are visible in MongoDB
- tests pass

## Phase 2. FEAK Diagnoser Wrapper

### Goal

Wrap the legacy FEAK analysis components as a clean tool.

### Tasks

- create `FEAKObservation` schema
- create legacy adapter
- normalize legacy output to canonical schema
- cache observations in `feak_observations`

### Success Criteria

- text input returns FEAKObservation JSON
- observation can be saved and retrieved by text hash

## Phase 3. Transition Feature v0

### Goal

Compute compact transition features for a candidate revision.

### Tasks

- define action schema
- define patch schema
- define transition feature schema
- compute target_gain
- compute target_gap_reduction
- compute non_target_drop
- compute evidence_match
- compute edit_ratio

### Success Criteria

- before/after essay pair produces a valid transition feature vector
- transition features are logged

## Phase 4. One-step Heuristic Controller

### Goal

Run one-step revision control without learned TVM.

### Tasks

- generate action candidates
- create patch candidates
- apply patches
- re-run FEAK observation
- compute transition features
- score candidates with heuristic scorer
- accept/reject/stop

### Success Criteria

- one essay produces multiple candidates
- one selected candidate or stop decision
- candidates are saved in MongoDB

## Phase 5. TVM Training Data Construction

### Goal

Construct preference pairs for Transition Value Model training.

### Tasks

- generate counterfactual candidate revisions
- identify positive transitions
- mine hard negative transitions
- create pairwise examples
- store in `transition_pairs` or `tvm_training_examples`

### Success Criteria

- dataset contains preferred/rejected transition pairs
- hard negative categories are logged

## Phase 6. Learned Transition Value Model

### Goal

Train a compact learned TVM.

### Tasks

- implement simple ranker model
- train using pairwise ranking loss
- evaluate pairwise accuracy
- compare against heuristic scorer

### Success Criteria

- learned TVM beats heuristic scorer on held-out transition pairs

## Phase 7. TVM-guided Trajectory Controller

### Goal

Replace heuristic scorer with learned TVM in the revision loop.

### Tasks

- integrate TVM into controller
- run multi-step trajectories
- log rollback and stop decisions
- compare baselines

### Success Criteria

- agent_runs, agent_steps, and candidates are populated
- trajectory logs can be exported

## Phase 8. Evaluation

### Baselines

- Original FEAK
- Repeated FEAK
- LLM self-refinement
- Heuristic FEAK-TC
- Learned TVM FEAK-TC

### Metrics

- final quality improvement
- transition pairwise accuracy
- regression rate
- over-edit ratio
- rollback correctness
- stop appropriateness
