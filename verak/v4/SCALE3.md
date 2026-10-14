# v4.3 B2/B3 bounded collection

This is a new `scale3` artifact namespace. Earlier v1/v4.1/v4.2 code, saved
episodes, and selections are unchanged. This code does not train or dispatch a
GPU job. Reference scoring is owned by the parent after one-shot evaluation.

## Before the gate

These commands read existing files and freeze source identities only. They do
not open API clients, initialize a scorer, or call Bareun:

```sh
python -m verak.v4.scale3 content-freeze
python -m verak.v4.scale3 corruption-freeze
python -m verak.v4.scale3 cost-estimate
```

B2 uses 1,000 new train sources: argumentative 333, explanatory 333, emotional
334. The filter is the mean of **grader_1/2**'s sixteen 1–5 values, selecting the
lower two-thirds within genre (ties retained). Rubric feedback is LLM-written,
not human feedback. Earlier preparation/smoke/corruption/extra-teacher source
IDs and normalized text duplicates are excluded. The question inventory
distinguishes the canonical 62 test-split questions, its frozen100-test-cohort
42-question subset, and 23 additional v3-dev questions (85 total). Every input
and prior cohort has a SHA; failed sources are never replaced.

B3 freezes the existing 1,430 active practices (741 original sources) without
new source exclusions or corruption generation. Both components make exactly
two independent unseeded teacher attempts where preparation/planning permits.
Uncollected and error attempts remain explicit in the planned denominator.

## Paid execution only after the same20 v4.3 smoke passes

`B1/gate.json` must pass with the immutable prompt contract and settled ledger.
The collection entry point validates gate/complete/metrics hashes before
opening any client. B2's cap is $50 minus **all three** smoke ledgers: prior
v4.1 $0.441109825, v4.2 $0.11151045, and actual v4.3 smoke spending. B3 has a
separate $30 cap. Confirmed plus reserved/unknown costs count toward caps.

Use the existing local-search virtualenv (feak_agent plus pinned local index
dependencies), from this code worktree:

```sh
/home/chanwoo/FEAK-Agent/verak/v4/outputs/scale2/search/venv/bin/python -m verak.v4.scale3 content-run
/home/chanwoo/FEAK-Agent/verak/v4/outputs/scale2/search/venv/bin/python -m verak.v4.scale3 corruption-run
```

The CLI hides CUDA, sets BLAS/tokenizer threads to one, nice19, and separates
B2 cores104–107 from B3 cores108–111. The shared Bareun lock allows at most one
new request/second across both components. Each starts with two episode
workers. After 20 started attempts without timeout/rate-limit evidence, maxima
are B2=6 and B3=4 (at most ten synchronous API requests across both). A file
`scale3/dispatch_policy.json` may pause new episodes or lower worker limits;
in-flight responses are saved and drained, never cancelled and regenerated.

```json
{"pause_new_episodes": false, "paused_components": {"B2": false, "B3": false}, "max_workers": {"B2": 6, "B3": 4}, "disable_ramp": false}
```

Ledger fingerprints include model/effort/request/prompt/schema and attempt
identity. Raw responses and per-turn state are flushed. Saved failed requests
are not silently regenerated. Only a newly authorized change may retry them.

## B2 content and search supervision

Sol supplies at most four located tasks, at most one INSERT task. A missing
relation between existing essay contents remains an ordinary INSERT task. An
actionable public fact/example request may instead be a Revision INSERT task
with `needs_search=yes`; writer experiences/opinions cannot be supplied by
search. The editor receives public task aliases and the exact v4.3 prefix,
without privileged LLM feedback, scores, recovery answers, or hidden suffixes.

The local frozen Wikipedia index supports authorized SEARCH only. A sourced
INSERT must cite a passage actually retrieved in the current delegation; at
most one sourced INSERT succeeds per essay, including after UNDO. The complete
retrieved passage, current cited sentence, title/passage ID, and retrieval
provenance are saved. The existing single Sol pair-judgment is extended with
support-by-cited-passage and paraphrasing; no second support API call is added.
Retrieval is local; its passage results enter the authorized Luna/Sol model
requests as teacher observations and judgment evidence.
Both passing attempts are retained under all six Dv3 conditions plus these
support gates. No corruption STOP gate is added to B2.

Sol raw plans remain in `content/items`; feedback-free Orchestrator target
payloads remain in `orchestrator_targets`. They are not tokenized using an
invented Orchestrator prompt. Editor export contains valid canonical single
action JSON only, with observations/tools/notices masked and exactly the
inference prefix. Full raw teacher outputs and trimming diagnostics remain in
saved attempts. Trimming is teacher-only; student parsing is strict.

## B3 exact reward and GPU handoff

Revision receives structure tasks; Korean receives local tasks and always
reviews the whole essay. Public IDs reflect current corrupted reading order.
The original stable IDs, record targets, normalization/lineage/tokens, and
inverse mapping remain private. No SEARCH task is assigned to B3.

Initial whitespace/segmentation changes are environment preparation, not agent
over-edit. Initial record recovery is reported separately. Only states that
project exactly to the original v1 units receive unchanged v1 rewards;
fragment interleaving, partial deletion, or cross-paragraph separation that
cannot project exactly is **unknown**, not zero. Quality inputs are the actual
normalized initial/Revision-final/Korean-final texts. Old raw-corpus Q is never
substituted silently. There is no provisional CPU quality scorer.

Collection writes `B3/gpu_manifest.json` and `gpu_ready.json` without launching
anything. Parent reference responses must match input key, GPU fingerprint,
and the frozen manifest; completion must say `scheduled_by=root` and
`slot=post_oneshot_evaluation`. After parent scheduling and completion:

```sh
python -m verak.v4.scale3 corruption-gpu-finalize
```

This command is file/CPU only, recomputes v1 arithmetic from saved reference
scores, and publishes immutable/idempotent `gpu_selection.json`. Best eligible
attempt is selected per practice and role: role R≥0.80, valid terminal STOP,
at most one rejection. The inherited no-GLOBAL exception remains: no structural
attempt, STOP within two steps, R_over=0. A rejected/undone splitting EDIT is
still a structural attempt. No RFT35% STOP cap, 2× duplication, or source cap is
added. Reports separate own-role operator recovery, unknowns, initial
environment recovery, and selected trajectory/source counts. No training runs.
