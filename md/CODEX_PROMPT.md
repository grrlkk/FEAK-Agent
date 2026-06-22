# CODEX_PROMPT.md

Paste this into Codex CLI from the repository root.

```text
Read AGENTS.md, PROJECT_SPEC.md, METHOD_SPEC.md, ROADMAP.md, and CURRENT_TASK.md.

Implement only CURRENT_TASK.md.

Before editing, summarize:
1. files you will create or modify
2. how MongoDB will be accessed
3. how AI-Hub JSON will be normalized
4. what tests you will run

Do not implement FEAK analyzer, action proposer, patch simulator, transition value model, controller loop, frontend, FastAPI, or model training.

After implementation:
- run pytest
- run the ingest script on up to 3 sample JSON files if available
- report changed files
- report test results
- report assumptions
- report unresolved issues
```
