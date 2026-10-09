# v4 preparation

`imple/FEAK_AGENT_METHOD.md` and the user's approved preparation scope define
this pilot. This package starts no model training or GPU scorer. It uses the
existing durable API accounting and low-priority local Bareun queue.

- `python -m verak.v4.data`: freeze 300 train/100 question-disjoint test sources,
  deduplicate source text, and materialize stable paragraph/sentence IDs.
- `python -m verak.v4.runner launch B`: Sol feedback-item collection ($5).
- `python -m verak.v4.runner launch D`: wait for B, then collect two Luna
  Revision-to-Korean attempts on 100 low/middle-human-score train sources and
  paired Sol judgments ($15). No SEARCH or scorer calls.
- `python -m verak.v4.report`: regenerate the local report from available
  artifacts, explicitly distinguishing pending GPU selections from final data.

Data live under the main experiment repository's `verak/v4/outputs/prep` and
are not included in Git. Every paid request has a durable input identity,
raw outcome, cost reservation, and separate B/C/D budget. Provider sampling is
unseeded; attempt identities create independent requests.

D uses v4 action JSON and a public state rebuilt from the current essay. The
private feedback message is appended only for teacher calls and is never in
the exported policy prefix. The export loss covers only the current valid
assistant action, using the actual pinned policy tokenizer/chat template.
Stop/step-limit endings are reported; D's selection gate is precisely the
four requested quality conditions, not the corruption teacher's reward gate.

The extra GLOBAL source and GPU-handoff adapters are separate from the v1
runtime. Only complete, immutable teacher collections may enter the existing
GPU slot after all RFT rollouts finish. Missing provisional CPU comparisons
are unknown; training selection uses GPU-reference values exclusively.
