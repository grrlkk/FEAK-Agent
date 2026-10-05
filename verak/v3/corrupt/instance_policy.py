"""Phase 3b candidate rules, separate from the reproducible Phase 3 catalog."""

from dataclasses import replace
import re

from .builder import BuildPolicy, build_dataset
from .document import BareunBank
from .operators import LEVELS, candidates as phase3_candidates

ACTIVE_LEVELS = {op: level for op, level in LEVELS.items() if op not in {"L_TRANSLATIONESE", "G_VAGUE"}}
POLARITY_PAIRS = {"수밖에 없다": "수밖에 있다", "수밖에 없습니다": "수밖에 있습니다",
                 "않을 수 없다": "않을 수 있다", "않을 수 없습니다": "않을 수 있습니다"}
BACK_REFERENCES = ("이처럼", "이러한", "이와 같이", "따라서", "그래서", "이 때문에")


def back_reference(text):
    for form in BACK_REFERENCES:
        pattern = r"\s*".join(map(re.escape, form.split()))
        if re.match(r"[\s\"'“‘(\[]*" + pattern + r"(?=$|\s|[,，:;])", text):
            return form
    return None


def candidates(doc, op, **kwargs):
    if op not in ACTIVE_LEVELS:
        raise ValueError("Operator removed from Phase 3b")
    proposals = phase3_candidates(doc, op, **kwargs)
    if op == "L_POLARITY":
        return [p for p in proposals if POLARITY_PAIRS.get(p.params["old"]) == p.replacement]
    if op == "G_DELETE_SUPPORT":
        units = doc.units
        successors = {a.sid: b for a, b in zip(units, units[1:])}
        kept = []
        for p in proposals:
            following = successors.get(p.sids[0])
            form = back_reference(following.text) if following else None
            if form:
                kept.append(replace(p, params={**p.params, "next_sentence_id": following.sid,
                    "next_back_reference": form, "next_cross_paragraph":
                    doc.locate(p.sids[0])[0] != doc.locate(following.sid)[0]}))
        return kept
    return proposals


POLICY = BuildPolicy(operators=tuple(ACTIVE_LEVELS), candidate_factory=candidates,
    per_essay=3, variant_stride=1, episode_prefix="phase3b:",
    schema_version="phase3b_instance_candidates_v1", use_vague_cache=False)


def candidate_bank(config):
    return BareunBank(config, cache_dir=config["paths"]["phase3b_output"] / "bareun_units",
                      read_cache_dirs=(config["paths"]["phase3_output"] / "bareun_units",))


def build_candidates(config, split):
    output = config["paths"]["phase3b_output"]
    if (output / "pilot_preregistration.json").exists():
        raise ValueError("Candidate pool is frozen for the pilot; do not overwrite")
    return build_dataset(config, split, output / f"candidates_{split}.jsonl",
        per_essay=3, seed=config["corruption_pilot"]["seed"], policy=POLICY, bank=candidate_bank(config))
