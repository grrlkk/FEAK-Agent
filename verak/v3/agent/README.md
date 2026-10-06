# Phase 5 environment and runtime

`RevisionEnv` in `verak/v3/env/` supports `two_stage` and `single`. The two-stage
runner starts GLOBAL, builds a factual hand-off when its stage terminates, and
starts KOREAN with a fresh role conversation. KOREAN cannot undo GLOBAL actions.
There is no policy training, orchestrator model, verifier, or output guardrail.

The initial observation assigns visible S/P IDs in the current document order.
The environment retains private source IDs for reward computation. Text, IDs,
the frozen Phase 2c compact annotation view, genre rubric names, and remaining
budgets are visible. Source answers and corruption records never enter prompts.
New insertions use N IDs; single-mode splits retain the first ID and use letter
suffixes for additional units. IDs are not recycled after UNDO.

EDIT/MOVE are atomic. Invalid arguments and role-forbidden actions consume one
step and return an error observation. Each role has its own undo stack, step
budget, CHECK budget, and three-consecutive-error limit. GLOBAL and KOREAN use
14 steps / 2 checks each; single uses 24 / 4. Reaching a CHECK budget ends that
stage, as does STOP or the step/error limit.

Bareun is the only live analyzer. Changed paragraphs are reanalyzed and cached
by paragraph text plus neighboring context hash. Edited replacement fragments
are also analyzed to establish their sentence boundaries before applying the
role restriction. Frozen annotation rules are recomputed from current tokens;
the feedback contains observations, never an antecedent identity or a semantic
correctness decision. It covers changed units, adjacent units, and edge-linked
units in severity order. The compact annotation view is never truncated.

CHECK exposes eight expected scores, Q, the stage-start delta, the previous
CHECK delta (null for the first CHECK), and the genre noise floor. Terminal
training rewards also need starting, stage-1, and final scores; these private
bookkeeping calls reuse scorer caches and are not new agent CHECK actions.
Real-essay mode has no terminal reward and performs scoring only when CHECK
needs it. No score is used as an action acceptance gate.

`TeacherBackend` extends the existing environment-only OpenAI adapter with
multi-turn Responses requests, using gpt-6.1-sol / low. Prior messages and the
role system prompt are passed verbatim. JSON parsing gets one retry; semantic
action errors go directly to the environment. Every API attempt, including
transport retries, reserves a shared Phase 5 ledger entry. Confirmed usage,
cache reads/writes, reasoning tokens, and timeout reservations are recorded.
An upper bound for the next request prevents passing the $10 smoke cap.

`PolicyBackend` uses the local vLLM chat endpoint with the tokenizer's template,
temperature 0, seed 47, no LoRA, and no JSON-constrained decoder. If necessary,
policy history keeps the system prompt, first observation, last eight message
turns, and a code-built action diary. Teacher histories are never rewritten.

Run the local policy server in the separate `verak_vllm` conda environment:

```bash
/home/chanwoo/anaconda3/envs/verak_vllm/bin/python -m verak.v3.cli.serve_policy
```

The server binds only 127.0.0.1:8030, GPU0, with multi-LoRA support enabled
(two slots, rank 64). It loads local snapshot
`c963a5f4f6496c749f94064a20b33028b0db9f19`. vLLM 0.8.5 / torch 2.6.0 / CUDA 12.4
is used for compatibility with the installed driver; feak_agent stays unchanged.
The minimal serving pins are in `requirements-vllm.txt`; install them only in
the separate environment.
The frozen 4-bit scorer and selected embedding run on GPU1.

Smoke commands (credentials must already be in the process environment):

```bash
python -m verak.v3.cli.run_episodes --backend teacher --mode two_stage --limit 20 --seed 47 --max-api-calls 1600 --max-cost-usd 10 --out verak/v3/outputs/phase5/teacher_two_stage
python -m verak.v3.cli.run_episodes --backend teacher --mode single --limit 10 --seed 47 --max-api-calls 1600 --max-cost-usd 10 --out verak/v3/outputs/phase5/teacher_single
python -m verak.v3.cli.run_episodes --backend policy --mode two_stage --limit 5 --seed 47 --max-api-calls 0 --out verak/v3/outputs/phase5/policy_two_stage
```

Run the two teacher commands sequentially; they share the same cost/call ledger.
Completed episodes are
reused; interrupted episodes replay exact cached teacher requests. A changed
manifest stops resumption. `events.jsonl` is flushed after every turn/action;
`episodes.jsonl` stores messages, observations, hand-off, final text, role
rewards, termination reasons, steps/checks, usage, cost, and timing. The smoke
CLI accepts only the approved dev corpus and never opens scorer train or test.
