# v4.3 local SEARCH and source support

These modules extend only the explicitly versioned v4.3 environment. They do
not modify v1/v4.2, call an API, rebuild the index, or train anything.

`search_v43.SearchSession` composes with the editor environment. The environment
owns role/delegation lifecycle, action parsing, steps, delegated locations, and
the existing two-successful-INSERT limit. The session owns the stricter rule:
one successful sourced INSERT per essay, even after UNDO.

The environment exposes public task aliases (`D1T1`, `D1T2`, etc.) and keeps
private source/feedback item IDs out of policy observations and action targets.
`SEARCH` has exactly `action`, `item_id` (public alias), and `query` fields.
It is permitted only for a current Revision task marked `needs_search="yes"`.
Korean cannot SEARCH. Missing `needs_search` means `no`, preserving the existing
smoke plans without inventing a search task.

`SearchSession.activate(public_tasks, delegation)` starts a delegation.
`search(action, role)` returns all text and attribution metadata for at most
three passages, or an explicit `no_results`. SEARCH consumes one environment
step and has no document/undo effect. An INSERT's `source` is null, or exactly
`{"title": "...", "passage_id": "..."}`. `validate_source` accepts only an
exact title/ID pair actually retrieved during the current search-authorized
delegation; arbitrary IDs and additional URLs are rejected before editing.

Before a document mutation, the environment snapshots citation state with
`snapshot()`. After a successful sourced INSERT it calls `commit_insert`, after
sentence splits it calls `inherit`, and after deletions it calls `prune`.
UNDO uses `restore` alongside the document undo stack; search history and the
successful sourced-INSERT count are not refunded. A Korean EDIT preserves the
sentence ID, so its citation remains attached and uses the edited current text.
On a rejected action neither document nor citation state may change.

`observation`/`export` accept `{sid: text}`, sentence dictionaries, or document
units. They retain full retrieved passages, the query/public task/delegation,
current cited sentence text, original insertion text, exact title/ID, Wikipedia
URL, and license. `citation_text` renders `[출처: title | passage_id]` next to a
sentence for review; it does not silently change the essay text. The environment
also attaches the structured source to the sentence view. These observations
are the same teacher/student/inference observations, with tool outputs masked
in action-target training exports.

The immutable index is the existing 20261001 kowiki dump, Kiwi/BM25 artifact
manifest SHA256
`860cda61615d665a5282fecb143147771885042fcca353992d3e6414b528ca78`.
`verify_index(verify_hashes=True)` checks its frozen files before paid work.
Normal retrieval is fully local. A worker lazily opens one read-only engine
per thread, with one Kiwi worker and one BM25 retrieval thread; SQLite
connections are never shared across threads. No search task means no index or
Kiwi load. Runtime dependencies are already in the isolated local-search venv.

`support_judge_v43` extends the existing B2 Sol pair judgment in the same API
request and ledger; it does not make another call. Add `JUDGE_APPEND`, use
`extend_schema`, attach `payload_for_attempt` to each attempted revision, and
validate both the unchanged quality fields (`strip_support` for that validator)
and `validate_support`. Preserve the complete judgment in storage.

Every surviving cited sentence gets `supported_by_cited_passage: yes/no` and
`paraphrased: yes/no`. Only the exact retrieved text is evidence, not outside
knowledge. Supported sourced facts do not count as invented specifics;
unsourced/unsupported facts and invented writer experiences still do. The
existing meaning, better-than-original, half-items-addressed, no-redundancy,
and no-new-awkwardness gates stay intact. Merge the `support_selection`
criteria into those gates. Missing/malformed source judgments fail closed.
Attempts without surviving citations are explicitly `none` with an empty
source-support list; reports keep applicable/none denominators separate.

The original texts/feedback and retrieved passages sent to the authorized Sol
editing/judgment API are not "fully local evaluation". Only the SEARCH backend
is local; the existing teacher/judge API boundary still applies. Preserve this
distinction in reports.
