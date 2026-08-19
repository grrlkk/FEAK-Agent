# FEAK-Agent
A revision agent that judges each edit as a transition (s→s′) using a value model trained from diagnostically corrupted essays, without human-labeled edit pairs.

FEAK-TC experiment repository for Korean essay revision-control research.

The active MVP path runs one local revision step:

```text
essay
  -> diagnose
  -> propose action-stratified candidates
  -> apply reversible patches
  -> re-diagnose
  -> compute transition features
  -> heuristic accept/reject/stop
```

This repository intentionally excludes the previous web frontend, FastAPI routers,
server runtime files, private keys, API keys, raw spreadsheets, and model weights.

## Layout

```text
feak_tc/mvp/             One-step FEAK-TC MVP loop
feak_tc/diagnose/        Stub, Kanana, and legacy FEAK diagnoser adapters
feak_tc/data/            AI-Hub JSON normalization utilities
feak_tc/db/              MongoDB skeleton for later ingestion/log storage
src/apps/                 Core analysis modules kept compatible with existing imports
scripts/                  Experiment entry points
experiments/configs/      Experiment configuration files
experiments/results/      Generated outputs, ignored by Git
data/                     Local datasets, ignored by Git
```

## MVP Smoke Run

Offline deterministic run:

```bash
python scripts/run_mvp.py \
  --text "인권은 인간이 가지는 기본적인 권리이다. 우리는 서로의 권리를 존중해야 한다." \
  --proposer-mode deterministic \
  --patcher-mode deterministic
```

Batch JSONL logging:

```bash
python scripts/run_mvp_batch.py \
  --input data/data_jsonl/train.jsonl \
  --output experiments/results/mvp_batch.jsonl \
  --diagnoser stub \
  --min-chars 150 \
  --proposer-mode deterministic \
  --patcher-mode deterministic
```

The batch input can be a `.txt`, `.jsonl`, `.json`, or a directory of `.txt`
files. Each JSONL output row contains the original record, candidates,
transition features, heuristic scores, and final decision.

True Kanana execution uses the sibling `essay_scoring_llm` package:

```bash
python scripts/run_mvp.py \
  --diagnoser kanana \
  --device-id 3 \
  --question "인권의 뜻과 특징에 대해 서술하세요" \
  --text-file sample_essay.txt
```

## Local Data

Place experiment inputs under `data/`. The default scripts expect:

```text
data/UKTA_1128_total_result.xlsx
```

You can override paths without editing code:

```bash
FEAK_INPUT_FILE=/path/to/input.xlsx FEAK_OUTPUT_FILE=/path/to/output.xlsx python scripts/run_final_scoring.py
```

## Secrets

Do not commit API keys. For Bareun, either set `BAREUN_API_KEY_PATH` or place a local
untracked key file at:

```text
secrets/bareun_api.txt
```

For OpenAI experiments, create a local `.env` with:

```text
OPENAI_API_KEY=...
```

## Model Weights

Essay scoring expects the GRU checkpoint locally at:

```text
src/apps/cohesion/essay_scoring/model/not_topic_model.pth
```

Model weights are ignored by Git. Store them locally or document an external download
location for reproducibility.
