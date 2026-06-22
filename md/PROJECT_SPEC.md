# PROJECT_SPEC.md

## Project Name

**FEAK-TC: FEAK-guided Transition Control for Open-ended Korean Writing Agents**

## One-line Summary

FEAK-TC uses FEAK diagnostic signals to control iterative writing revision trajectories: it generates multiple candidate revision actions, evaluates the transition caused by each action, and decides whether to accept, reject, rollback, or stop.

## What This Project Is

This is a research implementation of an **open-ended Writing Agent**.

The system should support:

- FEAK-based diagnostic observation of the current essay
- stratified candidate action generation
- patch-based candidate revision simulation
- transition-level evaluation
- checkpointed trajectory control
- rollback and stopping
- trajectory logging for later learning and evaluation

## What This Project Is Not

This project is not:

- a frontend application
- a chatbot UI
- a classroom product
- a full rewrite of FEAK-web
- a direct continuation of the old UKTA_v2 web code
- a generic LLM feedback generator

## Research Motivation

LLM-based writing systems are moving from one-shot generation and one-shot feedback toward Writing Agents that repeatedly intervene in the writing process.

However, the key bottleneck is not simply generating revisions. The real problem is controlling revision trajectories.

An open-ended Writing Agent must decide:

- which revision action to try
- whether the action improves the current essay state
- whether the action damages other rubrics or causes regression
- whether the action should be rejected
- whether a previous checkpoint should be restored
- whether further revision should stop

## Core Problem

Prior Writing Agent research has advanced separate components:

- planning and action exploration
- evaluation and reward signals
- intent, evidence, and grounding

But these components are not sufficiently integrated into a single closed-loop revision controller:

```text
action selection -> step-level verification -> regression tracking -> rollback -> stopping
```

This project addresses that gap.

## FEAK's Role

The original FEAK system performs one-shot Korean writing diagnosis and feedback:

```text
measurement -> feedback
```

FEAK-TC reuses FEAK diagnostic signals for revision control:

```text
measurement -> transition evaluation -> control
```

FEAK diagnostic signals include:

- 8 rubric scores
- 29 linguistic features
- elite benchmark gaps
- weak rubrics
- selected priority features
- evidence sheet

FEAK is an observation tool. It is not the controller and not the final reward by itself.

## Expert Feedback's Role

Expert feedback data should not dominate the framing of the paper.

Expert feedback is used as auxiliary signal for:

- validating action direction
- constructing positive/hard-negative transition pairs
- calibrating the Transition Value Model

The main contribution remains FEAK-guided transition control.

## Main Method Idea

At each revision step, FEAK-TC does:

1. **Observe**: run FEAK Diagnoser on the current essay.
2. **Propose**: generate action-type-stratified candidates.
3. **Simulate**: apply each action as a patch to create candidate revised essays.
4. **Evaluate**: compute compact transition features and score each transition.
5. **Control**: accept, reject, rollback, or stop.
6. **Log**: store the entire trajectory.

## Initial Technical Scope

The initial implementation should focus on:

- clean repository setup
- MongoDB-backed data ingestion
- AI-Hub JSON normalization
- FEAK observation schema
- transition feature schema
- one-step controller skeleton

Do not implement frontend, full RL, RAG, long-term memory, or deployment in the initial phase.
