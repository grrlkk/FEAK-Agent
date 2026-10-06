# Phase 4 reward interface

Read `imple/VERAK_V3_SPEC_ADDENDUM_1.md` before starting another phase. The reward
module runs after an episode for training/evaluation; it is not an action gate,
orchestrator, or verifier inside the agent loop.

`total.rewards` accepts the hidden `source`, the starting `corrupted` document,
and the `final` document as `corrupt.document.Document` instances with stable
sentence/paragraph IDs and current Bareun tokens. The two-stage mode additionally
requires the actual stage-1 document, its score, and each role's action log.
The Phase 1 scorer owns score caching; request stage-1 Q once with its default
cache enabled. Do not average scores or generate scorer feedback.

```python
from verak.v3.reward import rewards

result = rewards(
    source, corrupted, final, records,
    mode="two_stage", stage1=stage1,
    stage1_actions=global_actions, stage2_actions=korean_actions,
    genre=genre, q_corrupted=q_start, q_stage1=q_middle, q_final=q_end,
    config=config["reward"], similarity=embedding_cosine,
    tau=config["similarity"]["tau"],
    preexisting_spell_spans=episode["preexisting_spell_spans"],
)
```

Return keys are `global`, `korean`, `combined`, and `mode`. Every populated reward
has `R`, `R_rec`, `R_q`, `R_over`, `R_step`, raw/weighted components, and per-record
details. GLOBAL recovery is evaluated at stage 1, using GLOBAL main recovery.
KOREAN recovery is evaluated at the final state, using each local record's main
recovery plus one coupled mean for each GLOBAL record that has coupled damage.
All local levels have equal record weights. Coupled CONJ/DEP weights are 1/.5.
Combined uses the per-record .7/.3 main/coupled weighting. Empty recovery sets
return zero. In single mode, use `actions`; role breakdowns are `None` because
there is no stage boundary from which to attribute work.

An action log entry should have `action`, `valid`, `args`, and preferably
`changed_sids` from the environment. Over-edit compares each role's own start
and end states on unreferenced source sentences. It counts `(form, tag)` edit
distance, excludes supplied spelling spans, and requires attribution for every
net changed unreferenced sentence. UNDO is accounted for by actual endpoints.
All actions, including CHECK/STOP and invalid attempts, count toward step cost.
Combined over-edit compares source to final. Spelling spans may use document
offsets or `{sid, start, end}` sentence-local offsets.

Recovery uses `recovery_predecessor_id`, not a previous corruption's intermediate
predecessor. Known off-topic insertions cannot fill deletion targets; a known bad
subject insertion cannot earn DEP repair credit. A new subject is credited only
with explicit marked-subject evidence, not merely `subject_omitted=false`.
Conjunction alternatives must change the expression and restore the source
coarse class; this structural criterion is not a semantic appropriateness claim.
Antecedent identity is never used.

Deletion credit uses the source paragraph and source position ±1, with a restored
ID or a newly inserted ID. Existing neighboring sentences cannot count as newly
restored support. Exact source restoration is 1; other candidates use the
calibrated cosine with zero below τ−.05, one above τ+.05, and linear interpolation.
The embedding model and τ are fitted on the union of the reconstruction sources.

`total.real_reward` implements the reference-free training reward from §9.3.
Its cohesion term is a structural-change proxy: WORD relation/polarity/modality,
SENTENCE conjunction/positional DEP, and TEXT off-style observations. The obsolete
antecedent-identity term is absent. It is not a semantic error label. Real-essay
trajectory collection and two-stage runtime integration belong to later phases.

Offline dev reward audit (no GPT/Bareun calls):

```bash
python -m verak.v3.cli.verify_rewards
```

Calibration commands require explicit `--max-api-calls`; all paid requests share
the Phase 4 ledger including model discovery and retries. The API code only reads
credentials from the process environment. Configure those in the launcher.
`cheap_model` selects the account-confirmed Luna ID; `role_teacher` records the
new Sol/low teacher for later phases. The old `teacher` key remains Phase 1
reproduction metadata and must not be used for new trajectory generation.
