# FEAK-Agent

Clean experiment repository for FEAK/KorCAT essay analysis research.

This repository intentionally excludes the previous web frontend, FastAPI routers,
server runtime files, private keys, API keys, raw spreadsheets, and model weights.

## Layout

```text
src/apps/                 Core analysis modules kept compatible with existing imports
scripts/                  Experiment entry points
experiments/configs/      Experiment configuration files
experiments/results/      Generated outputs, ignored by Git
data/                     Local datasets, ignored by Git
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
