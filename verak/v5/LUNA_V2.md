# Luna v2 preparation experiment

The current authorized scope is the user's 14-point review of the first preparation
episode: rerun the **same ten train essays**, with quality checks disabled and
enabled, then produce the complete notice annotation table. Do not run the selected
50 validation essays. The main entry point checks the explicit human authorization
file and currently refuses evaluation. No training or final-test loading occurs.

Use the existing `feak_agent` interpreter. The local Bareun service and the frozen
reference scorer must be available. A separate scorer service uses physical GPU1's
remaining memory; do not resume or terminate unrelated paused training jobs.

```sh
CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 \
  /home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v5.luna_v2_score
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v5.luna_v2_r2
/home/chanwoo/anaconda3/envs/feak_agent/bin/python -m verak.v5.luna_v2_r2_report
LUNA_R2_NATIVE_TEST=1 /home/chanwoo/anaconda3/envs/feak_agent/bin/python \
  -m pytest tests/test_luna_v2.py tests/test_luna_v2_r2.py -q
```

Artifacts remain local under `verak/v5/outputs/luna_upper_bound_v2/`. The original
preparation, prompts, and API responses are retained separately. The two current
arms are `prep_r2_discovery/B` and `prep_r2_checked/B`; the legacy `B` directory name
does not mean discovery quality checks were enforced. Meaning/scorer checks in
discovery are observations used for threshold calibration, not adoption vetoes.

`reports/luna_upper_bound.md` summarizes both arms. The `_prep_r2_notices.csv` and
`.md` files contain **every emitted notice**, including rejected attempts. Human
validity fields remain empty. `_prep_r2_essays.md` contains all original and final
texts, and `_prep_r2_logs/` contains all observations, actions and checks.

The current prompt contract extracts the student's actual thesis, disallows new
claims/solutions, and limits structural deletion to 30% of original sentences.
Rewriting notices target only moved sentences and untouched paragraph seams.
Reference/conjunction detection uses Bareun POS; topic selection is explicitly a
surface heuristic, not a solved dependency or antecedent parser. Unchanged fix
outputs become keep; edits outside the morphological scope become blocked and
are not applied. Both before/after repair strings and requested/effective actions
are logged. Whole paragraph updates and neighboring repairs are transactional.

Modification is normalized Unicode-character Levenshtein distance, including
whitespace. The threshold is the 90th percentile among meaning-preserving
discovery candidates; the score noise floor is the 90th percentile of absolute
whole-essay score changes from ten surface paraphrases rechecked under the revised
meaning policy. Neither threshold uses evaluation outputs. Sentence alignment
remains diagnostic and supports 1:2/2:1/1:3/3:1 splits/merges with uncertainty.

Every paid call shares the existing durable reservation ledger and a **$5 total**
cap, including preparation, calibration and the previously submitted training-only
Claude cost pilot. Secrets are read from the existing environment, not saved in
artifacts. The user approved keeping Sonnet 5.5 and omitting its unsupported
temperature setting; stored feedback authorship is explicitly unverified.

The main-comparison scaffolding (`luna_v2_main`, Batch handling, paired statistics)
is not a completed evaluation and must not be used to claim any A/B/C/D result.
The old v2 prompts and the reviewed preparation prompts are distinct contracts;
a future main run must first receive authorization and explicitly adopt/freeze the
reviewed contract. Raw essays, feedback, outputs and human judgments are not
committed. Config/ID-hash manifests contain no essay bodies.
