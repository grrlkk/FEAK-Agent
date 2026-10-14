# Three Luna roles: preparation-only simplification

This implements the user's follow-up to the v2 preparation review: first check
whether three Luna roles can improve the writing, with notices and the inspection
machinery removed. The same ten **train** preparation essays are the only allowed
inputs. No validation50 run, final-test access, training, GPU use, Bareun calls,
search, quality rejection, score-based acceptance, or candidate retries.

The one-pass baseline receives the question, original, and cached automatic
scores. The three-role arm receives the same inputs for its plan, then rewrites
the **whole essay once**, then proofreads the whole current essay once. All roles
use `gpt-6-luna`, reasoning `none`, temperature 0. Unresolvable writer notes are
withheld from the writer. Content preservation is a prompt constraint, not an
acceptance filter. Only executable JSON shape is checked; raw responses are saved
before parsing. A JSON string with a literal newline can be decoded without
changing the candidate or dispatching a new request.

This intentionally differs from the v2 paragraph-specific observation/action
format and does not validate a future 8B training format. The intermediate writer
output is retained to measure the Korean role's contribution.

After generation, Claude Sonnet 5.5 independently compares one-pass vs final and
writer vs final, in both orders. Provider-facing prompts and item IDs hide source
IDs, conditions, plans, and scores. Neither generation nor judging receives stored
feedback or human scores. Judgments require exact evidence quotes. Order
disagreement is explicit; the two views are not independent essays. Safe
improvement requires both judgments to report improvement, preserved meaning,
and no added content. Errors flagged by either view are conservative alerts,
not human-confirmed errors.

An evaluation-only recovery was added during the run after one judge cited text
absent from its candidate: preserve the invalid response and repeat the exact
blinded request once. Quality or preference never triggers retries. Both calls
are billed and reported, and the interruption is included in wall-clock time.
This does not regenerate or filter any writing candidate.

Commands (run from the worktree with the existing feak_agent Python):

```bash
python -m verak.v5.luna_simple initialize
python -m verak.v5.luna_simple generate
python -m verak.v5.luna_simple judge
python -m verak.v5.luna_simple report
python -m pytest tests/test_luna_simple.py -q
```

Config, prompts, source hashes, observations, raw responses, usage, elapsed time,
anonymous mappings, judgments and aggregates are saved beneath
`verak/v5/outputs/luna_upper_bound_v2/simple_train10` in the original repository.
Both providers share the existing durable v2 ledger and **$5 cumulative cap**.
Unknown sent outcomes retain their reservations; they are not blindly retried.
Completed calls and episodes resume from storage.

Reports are local-only `reports/luna_three_agent_simple.md` and
`reports/luna_three_agent_simple_samples.md`; raw essays and outputs are never
committed. The frozen config contains only IDs, hashes and protocol metadata.
