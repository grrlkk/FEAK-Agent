# METHOD_SPEC.md

## Method Name

**FEAK-TC: FEAK-guided Transition Control**

## Core Thesis

FEAK-TC evaluates revision at the level of **state transitions**, not at the level of final essays or feedback sentences.

The central question is:

```text
Did this action move the essay from the current state to a better next state?
```

## High-level Loop

```text
Current essay x_t
  -> FEAK Diagnoser
  -> Stratified Counterfactual Action Proposal
  -> Patch Simulator
  -> Transition Value Model / Heuristic Scorer
  -> Checkpointed Controller
  -> Next essay x_{t+1}
```

## Module 1. FEAK Diagnoser

### Role

Convert the current essay into diagnostic observation.

### Input

- current essay text

### Output

```json
{
  "rubric_scores": {},
  "features": {},
  "elite_gaps": {},
  "weak_rubrics": [],
  "selected_features": [],
  "evidence_sheet": []
}
```

### Rule

The Diagnoser does not write to the database directly. Repository modules handle persistence.

## Module 2. Stratified Counterfactual Action Proposal

### Purpose

Address B1: state-conditioned action selection.

### Idea

Do not let the LLM freely choose only additive feedback. Instead, generate candidates across explicit action types.

Initial action types:

- ADD_DETAIL
- DELETE_OR_FOCUS
- COMPRESS
- RESTRUCTURE
- STYLE_REFINE
- STOP

### Output Schema

```json
{
  "action_type": "DELETE_OR_FOCUS",
  "target_rubric": "organization_2",
  "target_span": "김치와 축구 선수 관련 문장",
  "instruction": "한글의 우수성과 직접 관련 없는 내용을 삭제하거나 축소한다"
}
```

## Module 3. Patch Simulator

### Purpose

Convert each candidate action into a patch-based candidate revision.

### Patch Schema

```json
{
  "operation": "replace",
  "target_span": "sentence_4",
  "before": "...",
  "after": "...",
  "reason": "..."
}
```

### Rule

Whole-essay rewrite is not allowed in MVP. Patch-based editing is required for rollback and edit-ratio computation.

## Module 4. Transition Feature Extractor

### Purpose

Compute compact transition features for each candidate revision.

### Approved Compact Transition Features

```text
phi(x_t, a, x_prime) = [
  action_type,
  target_rubric,
  target_gain,
  target_gap_reduction,
  non_target_drop,
  evidence_match,
  edit_ratio
]
```

### Feature Definitions

#### action_type

One of the predefined revision action categories.

#### target_rubric

The rubric the action is intended to improve.

#### target_gain

Change in target rubric score:

```text
target_gain = score_after[target_rubric] - score_before[target_rubric]
```

#### target_gap_reduction

Change in distance from elite benchmark for target features:

```text
gap_reduction = abs(z_before) - abs(z_after)
```

#### non_target_drop

Maximum score drop among non-target rubrics:

```text
non_target_drop = max(score_before[r] - score_after[r]) for r != target_rubric
```

#### evidence_match

Whether the patch targets the evidence span/rubric/feature indicated by FEAK or auxiliary feedback.

#### edit_ratio

How much of the text changed:

```text
edit_ratio = changed_tokens / total_tokens
```

## Module 5. Transition Value Model

### Purpose

Address B2: step-level action utility evaluation.

### v0: Heuristic Scorer

Use this only for MVP and baseline.

```text
score = target_gain + target_gap_reduction + evidence_match - non_target_drop - edit_ratio
```

### v1: Learned Transition Value Model

Train a model:

```text
V_theta(phi) -> transition_value
```

The model should learn which transition is better using pairwise ranking.

### Pairwise Loss

```text
L = -log sigmoid(V_theta(phi_preferred) - V_theta(phi_rejected))
```

### Positive Transition Criteria

A positive transition should satisfy most of:

- target rubric improves
- target feature gap decreases
- evidence_match is high
- non_target_drop is low
- edit_ratio is not excessive
- auxiliary expert feedback direction is not contradicted

### Hard Negative Transition Criteria

A hard negative transition may satisfy one surface criterion but fail another:

- target_gain is positive but non_target_drop is large
- target_gain is positive but evidence_match is low
- edit_ratio is excessive
- action type contradicts the diagnosed issue
- expression is polished but the target issue remains unresolved

## Module 6. Checkpointed Trajectory Controller

### Purpose

Address B3: rollback and stopping.

### Decisions

- accept
- reject
- rollback
- stop

### Controller Rules for MVP

- Select the candidate with highest transition value.
- Reject candidates with evidence_match below threshold.
- Reject candidates with non_target_drop above threshold.
- Reject candidates with edit_ratio above threshold.
- Stop if all candidates are below minimum value.
- Stop if recent target_gain is below threshold.
- Rollback if current checkpoint is worse than previous checkpoint.

## Novelty Claims

The method's novelty is not FEAK reuse itself.

The novelty is:

1. Evaluating revision at the transition level rather than final-output level.
2. Generating counterfactual action-type-stratified candidates from the same state.
3. Learning a Transition Value Model from compact FEAK-based transition signals.
4. Using checkpointed control for accept/reject/rollback/stop.

## Out of Scope for MVP

Do not implement:

- learned tool router
- RAG
- full semantic drift model
- NLI verifier
- content-unit extraction
- style/voice preservation model
- full RL or offline RL
- multi-agent architecture
