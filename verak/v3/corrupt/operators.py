"""Small surface tables, record-driven global moves, and Bareun application checks."""

from collections import Counter
from dataclasses import dataclass, field
import itertools
import re

from ..common import sha_text
from ..ko.coarse import CLOSED_CONJUNCTIONS
from .document import Document, MARKER, Unit, positional_changes, protected

LEVELS = {**dict.fromkeys(("G_PARA_SWAP", "G_SENT_MOVE", "G_DELETE_SUPPORT", "G_OFFTOPIC", "G_VAGUE"), "GLOBAL"),
          **dict.fromkeys(("L_CONN", "L_POLARITY", "L_SPACING"), "WORD"),
          **dict.fromkeys(("L_CONJ", "L_SUBJ_INSERT", "L_TRANSLATIONESE"), "SENTENCE"),
          "L_REGISTER": "TEXT"}
EC_FORMS = {"면": ("CONDITION", ""), "으면": ("CONDITION", "으"),
            "니까": ("CAUSE", ""), "으니까": ("CAUSE", "으"),
            "므로": ("CAUSE", ""), "으므로": ("CAUSE", "으"),
            "지만": ("ADVERSATIVE", ""), "려고": ("PURPOSE", ""), "으려고": ("PURPOSE", "으")}
EC_REPLACEMENTS = {"CONDITION": ("면", "으면"), "CAUSE": ("니까", "으니까"),
                   "ADVERSATIVE": ("지만", "지만"), "PURPOSE": ("려고", "으려고")}
POLARITY = {"수밖에 없다": "수밖에 있다", "수밖에 없습니다": "수밖에 있습니다",
            "않을 수 없다": "않을 수 있다", "않을 수 없습니다": "않을 수 있습니다"}
# Explicit surface pairs: never make forms such as 살아야 -> 살을 수.
for _before, _after in (("해야", "할"), ("먹어야", "먹을"), ("읽어야", "읽을"),
                        ("받아야", "받을"), ("살아야", "살"), ("가야", "갈"),
                        ("와야", "올"), ("봐야", "볼"), ("써야", "쓸"),
                        ("지켜야", "지킬"), ("아껴야", "아낄"), ("줄여야", "줄일")):
    POLARITY[_before + " 한다"] = _after + " 수 있다"
    POLARITY[_before + " 합니다"] = _after + " 수 있습니다"
REGISTER = {"한다": "합니다", "된다": "됩니다", "있다": "있습니다", "없다": "없습니다",
            "이다": "입니다", "했다": "했습니다", "됐다": "됐습니다", "었다": "었습니다",
            "았다": "았습니다", "였다": "였습니다", "겠다": "겠습니다", "좋다": "좋습니다",
            "싶다": "싶습니다", "같다": "같습니다", "많다": "많습니다", "적다": "적습니다",
            "크다": "큽니다", "작다": "작습니다", "높다": "높습니다", "낮다": "낮습니다",
            "간다": "갑니다", "온다": "옵니다", "본다": "봅니다", "쓴다": "씁니다",
            "준다": "줍니다", "난다": "납니다", "안다": "압니다", "든다": "듭니다"}
TRANSLATIONESE = {"노력해야 한다": "노력하는 것이 필요하다", "노력해야 합니다": "노력하는 것이 필요합니다",
                  "참여해야 한다": "참여하는 것이 필요하다", "해결해야 한다": "해결하는 것이 필요하다",
                  "관심이 있다": "관심을 가지고 있다", "관심이 있습니다": "관심을 가지고 있습니다",
                  "책임이 있다": "책임을 가지고 있다", "의미가 있다": "의미를 가지고 있다"}
SPACING = {"할 수": "할수", "될 수": "될수", "수 있다": "수있다", "수 없다": "수없다",
           "수 있습니다": "수있습니다", "수 없습니다": "수없습니다", "것이다": "것 이다",
           "것입니다": "것 입니다", "그렇기 때문에": "그렇기때문에", "예를 들어": "예를들어"}
SUBJECTS = ("그는", "그것은", "이것은", "나는")


@dataclass
class Proposal:
    op: str
    sids: list[str]
    params: dict = field(default_factory=dict)
    span: tuple | None = None
    replacement: str | None = None


def patches(op, ann, table):
    for old, new in table.items():
        for match in re.finditer(re.escape(old), ann.text):
            if not protected(ann.text, *match.span()):
                yield Proposal(op, [ann.sid], {"old": old, "new": new}, match.span(), new)


def local_candidates(doc, op):
    structure = doc.structure()
    for ann in structure.annotations:
        if ann.multi_unit:
            continue
        if op in {"L_POLARITY", "L_TRANSLATIONESE", "L_SPACING"}:
            yield from patches(op, ann, {"L_POLARITY": POLARITY, "L_TRANSLATIONESE": TRANSLATIONESE,
                                         "L_SPACING": SPACING}[op])
        elif op == "L_SUBJ_INSERT" and ann.subject_omitted:
            for subject in SUBJECTS:
                yield Proposal(op, [ann.sid], {"inserted_subject": subject}, (0, 0), subject + " ")
        elif op == "L_CONJ" and ann.initial_conj and ann.initial_conj["eligible"]:
            pattern = r"\s*".join(map(re.escape, ann.initial_conj["form"].split()))
            match = re.match(r"[\s\"'“‘(\[]*(" + pattern + r")(?=$|\s|[,，:;])", ann.text)
            if match and not protected(ann.text, *match.span(1)):
                for form, cls in CLOSED_CONJUNCTIONS.items():
                    if cls != ann.initial_conj["coarse_class"]:
                        yield Proposal(op, [ann.sid], {"coarse_before": ann.initial_conj["coarse_class"],
                            "coarse_after": cls, "old": match[1], "new": form}, match.span(1), form)
        elif op == "L_CONN":
            for ec in ann.connectives:
                if not ec["eligible"] or ec["kind"] != "EC" or ec["form"] not in EC_FORMS:
                    continue
                end = ec["span"][1] - ann.start
                # Bareun spans can overlap a restored stem. Require a literal surface ending.
                form = ec["form"]
                start = end - len(form)
                if start < 1 or ann.text[start:end] != form or protected(ann.text, start, end):
                    continue
                stem_end = start
                stem = ann.text[:stem_end]
                # No unconstrained conjugation. Only a realized Hangul stem and fixed allomorphs.
                if not stem or not ('가' <= stem[-1] <= '힣'):
                    continue
                coda = (ord(stem[-1]) - ord('가')) % 28
                for cls, forms in EC_REPLACEMENTS.items():
                    if cls == ec["coarse_class"]:
                        continue
                    replacement = forms[bool(coda)]
                    yield Proposal(op, [ann.sid], {"coarse_before": ec["coarse_class"],
                        "coarse_after": cls, "old": form, "new": replacement,
                        "token_id": ec["token_id"]}, (start, end), replacement)
        elif op == "L_REGISTER" and ann.style == structure.dominant_style and ann.style in {"한다", "합니다"}:
            table = REGISTER if ann.style == "한다" else {v: k for k, v in REGISTER.items()}
            # Final surface suffix only. Embedded EF and non-final occurrences are never sites.
            for old in sorted(table, key=lambda s: (-len(s), s)):
                match = re.search(re.escape(old) + r"(?=[.!?。！？…\s]*$)", ann.text)
                if match and ann.final_ending and not protected(ann.text, *match.span()):
                    yield Proposal(op, [ann.sid], {"style_before": ann.style,
                        "style_after": "합니다" if ann.style == "한다" else "한다",
                        "dominant_style": structure.dominant_style, "old": old, "new": table[old]},
                        match.span(), table[old])
                    break


def global_candidates(doc, op, *, donors=(), vague_cache=None, question_hash=None):
    annotations = {a.sid: a for a in doc.structure().annotations}
    if op == "G_PARA_SWAP" and len(doc.paragraphs) >= 3:
        for i, j in itertools.combinations(range(len(doc.paragraphs)), 2):
            yield Proposal(op, [u.sid for p in (doc.paragraphs[i], doc.paragraphs[j]) for u in p.units],
                           {"paragraph_indices": [i, j]})
    elif op == "G_SENT_MOVE" and len(doc.paragraphs) >= 3:
        for pi, p in enumerate(doc.paragraphs):
            for si, u in enumerate(p.units):
                if len(p.units) < 2 or annotations[u.sid].multi_unit:
                    continue
                for dest, target in enumerate(doc.paragraphs):
                    if dest != pi:
                        yield Proposal(op, [u.sid], {"from_paragraph": pi, "from_position": si,
                            "to_paragraph": dest, "to_position": len(target.units)})
    elif op == "G_DELETE_SUPPORT":
        for pi, paragraph in enumerate(doc.paragraphs):
            for si, unit in enumerate(paragraph.units[1:], 1):
                if not annotations[unit.sid].multi_unit:
                    yield Proposal(op, [unit.sid], {"paragraph": pi, "position": si,
                        "support_status": "noninitial_candidate_semantics_checked_by_QC"})
    elif op == "G_OFFTOPIC":
        dominant = doc.structure().dominant_style
        for donor in donors:
            if donor["question_hash"] == question_hash or donor["style"] != dominant:
                continue
            sid = "I_" + sha_text(donor["source_id"] + ":" + donor["sid"] + ":" + donor["text"])[:12]
            if any(u.sid == sid for u in doc.units):
                continue
            for pi, p in enumerate(doc.paragraphs):
                yield Proposal(op, [sid], {"donor": donor, "paragraph": pi, "position": len(p.units)})
    elif op == "G_VAGUE":
        for unit in doc.units:
            if annotations[unit.sid].multi_unit:
                continue
            cached = (vague_cache or {}).get(sha_text(unit.text))
            if cached and cached["original"] == unit.text and cached.get("valid"):
                yield Proposal(op, [unit.sid], {"cache_key": sha_text(unit.text)}, (0, len(unit.text)), cached["vague"])


def candidates(doc, op, **kwargs):
    return list(local_candidates(doc, op) if LEVELS[op] != "GLOBAL" else global_candidates(doc, op, **kwargs))


def local_verified(proposal, before, after):
    """Inspect observations at the edited site, not an unrelated occurrence elsewhere."""
    if after.multi_unit or before.text == after.text:
        return False
    op, params = proposal.op, proposal.params
    a, b = proposal.span
    new_end = a + len(proposal.replacement)
    changed = [t for t in after.tokens if t.start-after.start < new_end and t.end-after.start > a]
    if op == "L_CONN":
        return any(c["kind"] == "EC" and c["eligible"] and
            c["coarse_class"] == params["coarse_after"] and c["span"][1]-after.start == new_end
            for c in after.connectives)
    if op == "L_CONJ":
        return bool(after.initial_conj and after.initial_conj["eligible"] and
                    after.initial_conj["coarse_class"] == params["coarse_after"])
    if op == "L_SUBJ_INSERT":
        return not after.subject_omitted and any(t.tag in {"JX", "JKS"} for t in changed)
    if op == "L_REGISTER":
        return bool(after.final_ending and after.style == params["style_after"] and
                    before.final_ending != after.final_ending)
    if op == "L_POLARITY":
        return any(t.form == "있" and t.tag in {"VV", "VA", "VX"} for t in changed)
    if op == "L_TRANSLATIONESE":
        return any(t.tag in {"VV", "VA", "VX", "NNB"} for t in changed)
    if op == "L_SPACING":
        return before.text.replace(" ", "") == after.text.replace(" ", "") and bool(changed)
    return True


def apply(doc, proposal, bank):
    old, current = doc.structure(), doc.clone()
    op, params = proposal.op, proposal.params
    original_units = {u.sid: u.text for u in doc.units if u.sid in proposal.sids}
    verification = "position_record"
    if proposal.span is not None:
        _, _, unit = current.locate(proposal.sids[0])
        ann = next(a for a in old.annotations if a.sid == unit.sid)
        if ann.multi_unit or protected(unit.text, *proposal.span):
            raise ValueError("Protected marker or multi-unit target")
        start, end = proposal.span
        changed = unit.text[:start] + proposal.replacement + unit.text[end:]
        if MARKER.findall(changed) != MARKER.findall(unit.text):
            raise ValueError("Anonymization markers changed")
        unit.text, unit.tokens = changed, bank.tokens(changed)
        new_ann = next(a for a in current.structure().annotations if a.sid == unit.sid)
        if op != "G_VAGUE" and not local_verified(proposal, ann, new_ann):
            raise ValueError("Bareun did not verify the intended local change")
        if op == "G_VAGUE" and (new_ann.multi_unit or new_ann.style != ann.style or
                                not .9 <= len(changed)/len(ann.text) <= 1.1):
            raise ValueError("Vague variant changed register, unit count, or length beyond 10 percent")
        verification = "bareun_reanalysis"
    elif op == "G_PARA_SWAP":
        i, j = params["paragraph_indices"]
        current.paragraphs[i], current.paragraphs[j] = current.paragraphs[j], current.paragraphs[i]
    elif op == "G_SENT_MOVE":
        pi, si, unit = current.locate(proposal.sids[0])
        current.paragraphs[pi].units.pop(si)
        current.paragraphs[pi].units[0].leading = ""
        dest = current.paragraphs[params["to_paragraph"]]
        unit.leading = " " if params["to_position"] else ""
        dest.units.insert(params["to_position"], unit)
    elif op == "G_DELETE_SUPPORT":
        pi, si, _ = current.locate(proposal.sids[0])
        if si == 0:
            raise ValueError("Cannot delete a paragraph's first sentence")
        current.paragraphs[pi].units.pop(si)
    elif op == "G_OFFTOPIC":
        donor = params["donor"]
        dest = current.paragraphs[params["paragraph"]]
        dest.units.insert(params["position"], Unit(proposal.sids[0], donor["text"], bank.tokens(donor["text"]), " "))
    else:
        raise ValueError("Unsupported operator")
    if current.text == doc.text:
        raise ValueError("No surface change")
    new = current.structure()
    corrupted_units = {u.sid: u.text for u in current.units if u.sid in proposal.sids}
    target = recovery_target(doc, proposal, old, original_units)
    record = {"op": op, "level": LEVELS[op], "sids": proposal.sids,
        "original_text": original_units, "corrupted_text": corrupted_units,
        "recovery_target": target,
        "coupled_changes": positional_changes(old, new) if op in {"G_PARA_SWAP", "G_SENT_MOVE"} else [],
        "params": {**params, "span": list(proposal.span) if proposal.span is not None else None},
        "verification": verification, "before_sha256": sha_text(doc.text), "after_sha256": sha_text(current.text),
        "inverse": doc.snapshot()}
    return current, record


def recovery_target(doc, proposal, structure, originals):
    op, sid, p = proposal.op, proposal.sids[0], proposal.params
    target = {"implementation_phase": 4, "sids": proposal.sids}
    if op == "G_PARA_SWAP":
        target.update(kind="paragraph_order", paragraph_ids=[v.pid for v in doc.paragraphs])
    elif op in {"G_SENT_MOVE", "G_DELETE_SUPPORT", "G_VAGUE"}:
        pi, si, _ = doc.locate(sid)
        target.update(kind="placement" if op == "G_SENT_MOVE" else "same_place_high_similarity",
            paragraph=doc.paragraphs[pi].pid, position=si, original=originals[sid])
        if op != "G_SENT_MOVE":
            target["similarity_threshold"] = "calibrate_in_phase4"
    elif op == "G_OFFTOPIC":
        target.update(kind="inserted_sentence_absent", inserted_id=sid)
    elif op in {"L_CONN", "L_CONJ"}:
        target.update(kind="coarse_class_at_site", coarse_class=p["coarse_before"],
                      site=list(proposal.span), channel="EC" if op == "L_CONN" else "initial_conjunction")
    elif op == "L_SUBJ_INSERT":
        target.update(kind="inserted_subject_absent", inserted_subject=p["inserted_subject"])
    elif op == "L_REGISTER":
        target.update(kind="final_EF_style", style=structure.dominant_style)
    else:
        target.update(kind="original_string_at_site", site=list(proposal.span), original=p["old"])
    return target


def restore_record(doc, record, bank):
    if sha_text(doc.text) != record["after_sha256"]:
        raise ValueError("Restore records in reverse application order")
    restored = Document.restore(record["inverse"], bank)
    if sha_text(restored.text) != record["before_sha256"]:
        raise ValueError("Inverse record is corrupt")
    return restored


def exact_restoration_satisfies(doc, record):
    """Phase 3 sufficient-condition oracle, NOT Phase 4 semantic recovery/reward.

    Exact canonical restoration at the recorded site satisfies any future relaxed
    criterion. No similarity threshold, partial credit, or quality reward here.
    """
    target, ids = record["recovery_target"], record["sids"]
    if target["kind"] == "paragraph_order":
        return [p.pid for p in doc.paragraphs] == target["paragraph_ids"]
    if target["kind"] == "inserted_sentence_absent":
        return all(u.sid != target["inserted_id"] for u in doc.units)
    if any(sid not in {u.sid for u in doc.units} for sid in ids):
        return False
    if target["kind"] in {"placement", "same_place_high_similarity"}:
        pi, si, unit = doc.locate(ids[0])
        return doc.paragraphs[pi].pid == target["paragraph"] and si == target["position"] and unit.text == target["original"]
    ann = next(a for a in doc.structure().annotations if a.sid == ids[0])
    if target["kind"] == "coarse_class_at_site":
        if target["channel"] == "initial_conjunction":
            return bool(ann.initial_conj and ann.initial_conj["coarse_class"] == target["coarse_class"])
        return any(c["coarse_class"] == target["coarse_class"] and
                   c["span"][1] - ann.start == target["site"][1] for c in ann.connectives)
    if target["kind"] == "final_EF_style":
        return ann.style == target["style"]
    if target["kind"] == "inserted_subject_absent":
        return ann.text == record["original_text"][ids[0]]
    a, b = target["site"]
    return ann.text[a:b] == target["original"]
