> 과거 2축 RV 구현 기록입니다. 현재 실행은 [RV 파일럿 안내](RV_PILOT.md)를 따릅니다.

# FEAK-Agent

Research code for **FEAK-TC**, a writing agent that revises Korean essays iteratively while
managing two things at once: whether each revision achieves what it was asked to do, and
whether it preserves the content that had to stay.

FEAK-TC is the successor to [FEAK](#relation-to-feak) (SAC'26), which turns rubric-linked
linguistic features into evidence for LLM feedback. FEAK stops at the feedback; FEAK-TC
asks what comes after it — once feedback becomes an actual edit, how do we verify the edit
and decide whether to keep it, retry, roll back, or stop?

> **Work in progress.** This repository accompanies a paper in preparation, targeted for
> conference submission in 2026. Interfaces and data formats are still moving.

## Overview

Verification is split into two levels. The **Revision Verifier (RV)** judges a single
revision. The **Trajectory Guard** checks whether the essay as a whole degrades as edits
accumulate.

| Step | What happens | Output |
|---|---|---|
| 1. Diagnose, retrieve, plan | FEAK locates weaknesses; RAG supplies traits of relevant high-quality essays | A concrete revision request and its preservation constraints |
| 2. Generate | An LLM produces several candidate revisions for the fixed request | Revised text and its change set |
| 3. Verify (RV) | Each candidate is judged on goal attainment and content preservation | Two-axis verdict |
| 4. Trajectory guard | The full essay and edit history are checked for accumulated damage | Flow-level problems with evidence |
| 5. Control | FEAK quality change and verdicts are combined | Accept, replan, restore, or stop |

RV takes `(task, essay before, revision request, essay after)` and emits two independent
axes, each `pass` / `partial` / `fail`:

- `target_fulfillment` — was the requested problem actually solved?
- `preservation` — were the claims, conditions, and evidence that had to stay kept intact?

The axes are judged separately: a revision can hit its goal and still destroy content, or
miss its goal while leaving the original intact. Preservation is not string identity —
errors the request asks to fix and additions it permits are allowed to change.

The current prototype requires **no new training**. Diagnosis uses the existing trained
Kanana scorer and independent FEAK features. Planning, patch generation, the two-axis RV
and the trajectory guard share a general local LLM. Individual modules can be trained or
replaced later, after the complete loop has been evaluated.

## Status

The training-free loop under `feak_tc/agent/` runs diagnosis, a concrete revision request,
multiple local patches, two-axis RV checks, selection against keeping the current text,
checkpointing, trajectory checks, rollback, replanning and bounded stopping. It saves the
original, every candidate, verdict reasons and the final essay. The default local model is
the cached `kakaocorp/kanana-1.5-8b-instruct-2505` in 4-bit inference; no external LLM API
is used. Only the diagnoser uses the trained scorer adapter; all other roles use general Kanana.

Optional exemplar retrieval accepts a curated train-only JSONL. Without a configured
corpus, the planner runs without exemplars. The earlier one-step heuristic MVP and
four-axis RV data pilot remain available. No RV training or quality benchmark is implied
by a successful execution smoke test.

Experiment records and the current state of collected data live in `feak_tc_docs/`.

## Repository layout

```text
feak_tc/diagnose/     FEAK / Kanana / stub scorer bindings
feak_tc/mvp/          one-step revision, patching, quality scoring, heuristics
feak_tc/agent/        training-free local revision loop and two-axis runtime RV
feak_tc/web/          local visual console, history and candidate comparison
feak_tc/rv/           RV pilot data tooling
feak_tc/corruption/   corruption generation, inspection, analysis
feak_tc/data/         AI-Hub JSON normalization
feak_tc/schemas/      source data schemas
src/apps/             Korean linguistic analysis and scoring
scripts/              entry points for runs and data reuse
configs/              run configuration
tests/                regression tests
feak_tc_docs/         specifications and experiment records
data/                 source corpora (not tracked)
experiments/results/  generated data and experiment output (not tracked)
```

## Setup

Pick the dependency set that matches your runtime.

```bash
pip install -r requirements.txt          # core / dev
pip install -r requirements-kanana.txt   # Kanana scorer
pip install -r requirements-legacy.txt   # legacy UKTA / KoBERT scorer
```

API keys are read from `.env`.

## Usage

Open the local visual console:

```bash
python scripts/run_agent_web.py --port 8765
```

Visit `http://localhost:8765` (forward port 8765 if using a remote server). Enter a task
and essay, then watch diagnosis, revision plans, candidate edits, two-axis RV decisions
and rollback. Compare versions, revisit saved runs or download the full JSON. The default
is a real Kanana run; the explicitly labeled demo mode does not load models. The server
binds to loopback and is intended for personal use, not public deployment.

Run the same loop from the CLI (existing trained Kanana assets and cached base weights required):

```bash
python scripts/run_agent.py \
  --text-file examples/local_agent_essay.txt \
  --question "학교에서 휴대전화 사용에 관한 자신의 주장과 이유를 쓰시오." \
  --output experiments/results/local_agent.json
```

The default allocation is trained Kanana on GPU 1 and general Kanana on GPU 2. Override with
`--kanana-device 1 --llm-device cuda:2`. All thresholds, budgets and model settings are in
[`configs/agent_local.yaml`](configs/agent_local.yaml). Use `--offline-smoke` for a wiring
check without models. See [the local agent guide](docs/TRAINING_FREE_AGENT.md) for setup,
output interpretation, optional retrieval and limitations.

Run the tests:

```bash
python -m pytest -q
```

Run the one-step MVP against the stub diagnoser, with no model download:

```bash
HF_HUB_OFFLINE=1 python scripts/run_mvp.py \
  --diagnoser stub \
  --text "인권은 인간이 가지는 기본적인 권리이다. 우리는 서로의 권리를 존중해야 한다." \
  --proposer-mode deterministic \
  --patcher-mode deterministic \
  --surface-normalizer off
```

This exercises the heuristic loop, not a trained RV or the full controller. For wiring a
real scorer, see [Diagnoser Integration](docs/DIAGNOSER_INTEGRATION.md).

## Relation to FEAK

FEAK is the preceding system and has its own repository at
[grrlkk/FEAK](https://github.com/grrlkk/FEAK). FEAK-TC reuses its scorers and feature
computation as the diagnostic layer and adds revision verification and trajectory control
on top.

> **From Evaluation to Feedback: A Feature-Based and LLM-Constrained Tool for Korean Writing Assessment**
> Chanwoo Jang, Ganghee Go, Jinyong Yun, Seokho Ahn, Myungsun Shin, Ho-Hyun Kil,
> Sungmin Chang, Do-Guk Kim, Young-Duk Seo.
> *The 41st ACM/SIGAPP Symposium on Applied Computing (SAC'26)*, Thessaloniki, Greece.
> [doi:10.1145/3748522.3780021](https://doi.org/10.1145/3748522.3780021)

FEAK dynamically identifies rubric-linked linguistic features with low values and uses them
as evidence for LLM-based feedback generation, grounding generated feedback in measurable
diagnostic signals so that it stays interpretable and verifiable.

Cite FEAK when referring to that diagnostic layer:

```bibtex
@inproceedings{jang2026feak,
  title     = {From Evaluation to Feedback: A Feature-Based and LLM-Constrained Tool
               for Korean Writing Assessment},
  author    = {Jang, Chanwoo and Go, Ganghee and Yun, Jinyong and Ahn, Seokho and
               Shin, Myungsun and Kil, Ho-Hyun and Chang, Sungmin and Kim, Do-Guk and
               Seo, Young-Duk},
  booktitle = {Proceedings of the 41st ACM/SIGAPP Symposium on Applied Computing (SAC '26)},
  year      = {2026},
  publisher = {ACM},
  address   = {New York, NY, USA},
  doi       = {10.1145/3748522.3780021}
}
```

## License

MIT. See [LICENSE](LICENSE) and [NOTICE](NOTICE) for third-party components.
