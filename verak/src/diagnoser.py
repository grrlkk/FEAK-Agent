"""Raw Kanana score/feedback parsing and one evidence-checked goal."""

import re
import time

from .schemas import AllowedChange, Evidence, Goal, GoalDraft, GoalResponse, RUBRICS

SYSTEM_PROMPT = (
    "에세이 채점기. 8개 루브릭(과제충실성, 설명명료성, 설명구체성, 설명적절성, "
    "문장연결성, 글통일성, 어휘적절성, 어법적절성)별 1-9점 채점 후 피드백을 작성한다."
)
FEEDBACK_ALIASES = dict(zip(
    ("과제 수행의 충실성", "설명의 명료성", "설명의 구체성", "설명의 적절성",
     "문장의 연결성", "글의 통일성", "어휘의 적절성", "어법의 적절성"), RUBRICS))


def user_prompt(question, draft):
    # User override: no keyword extraction, annotation, or keyword input line.
    return f"질문: {question}\n에세이: {draft}"


def parse_diagnosis(raw):
    lines = raw.strip().splitlines()
    if not lines or not re.fullmatch(r"[1-9](?:[\s,]+[1-9]){7}", lines[0].strip()):
        raise ValueError("Expected exactly eight integer scores from 1 to 9 on the first line")
    numbers = [int(n) for n in re.findall(r"[1-9]", lines[0])]
    marker = re.search(r"(?m)^###\s*Feedback:\s*", raw)
    if marker is None:
        raise ValueError("Missing Feedback section")
    body = raw[marker.end():]
    names = (*RUBRICS, *FEEDBACK_ALIASES)
    pattern = r"(?m)^\s*(?:(?:[-*]|\d+[.)])\s*)?(" + "|".join(map(re.escape, names)) + r")\s*[:：]\s*"
    headers = list(re.finditer(pattern, body))
    feedback = {}
    for i, match in enumerate(headers):
        key = FEEDBACK_ALIASES.get(match.group(1), match.group(1))
        if key in feedback:
            raise ValueError("Duplicate rubric feedback")
        end = headers[i + 1].start() if i + 1 < len(headers) else len(body)
        feedback[key] = body[match.end():end].strip()
    if set(feedback) != set(RUBRICS) or any(not value for value in feedback.values()):
        raise ValueError("Expected nonempty feedback for all eight rubrics")
    return {"scores": dict(zip(RUBRICS, numbers)), "feedback": feedback}


class Diagnoser:
    def __init__(self, generate, on_record=None):
        self.generate = generate
        self.on_record = on_record or (lambda row: None)

    def diagnose(self, question, text):
        payload = user_prompt(question, text)
        for attempt in range(3):
            started = time.monotonic()
            record = {"role": "diagnose", "attempt": attempt, "system": SYSTEM_PROMPT, "input": payload}
            try:
                result = self.generate(SYSTEM_PROMPT, payload)
                if isinstance(result, str):
                    raw, usage = result, None
                else:
                    raw, usage = result["raw"], result.get("usage")
                record.update(raw=raw, usage=usage)
                diagnosis = parse_diagnosis(raw)
                record["status"] = "completed"
                return diagnosis
            except Exception as exc:
                record.update(status="error", error_type=type(exc).__name__)
                if attempt == 2:
                    raise RuntimeError("Kanana diagnosis failed after three attempts") from exc
            finally:
                record["elapsed_s"] = time.monotonic() - started
                self.on_record(record)


class LocalKanana:
    """Reuse the installed FT loader; preserve raw generated scores, without RF conversion."""

    def __init__(self, config):
        self.config = config
        self.loaded = None

    def __call__(self, system, user):
        if self.loaded is None:
            from feak_tc.diagnose.kanana import _ensure_package_importable
            _ensure_package_importable(self.config["package_path"])
            from essay_scoring_llm.config import ScoringConfig
            from essay_scoring_llm.soft_sc import load_model_and_tokenizer
            cfg = ScoringConfig(**self.config["inference"])
            model, tokenizer, _ = load_model_and_tokenizer(cfg)
            self.loaded = model, tokenizer, cfg
        import torch
        model, tokenizer, cfg = self.loaded
        prompt = tokenizer.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt", add_special_tokens=False, truncation=False)
        if inputs["input_ids"].shape[1] > cfg.max_seq_length:
            raise ValueError("Kanana input exceeds context limit; refusing to truncate the essay")
        inputs = {key: value.to(model.device) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=cfg.feedback_max_new_tokens,
                do_sample=True, temperature=cfg.temperature, top_k=cfg.top_k,
                pad_token_id=tokenizer.pad_token_id, eos_token_id=tokenizer.eos_token_id)
        count = inputs["input_ids"].shape[1]
        generated = output[0, count:]
        return {"raw": tokenizer.decode(generated, skip_special_tokens=True),
                "usage": {"input_tokens": count, "output_tokens": len(generated),
                          "total_tokens": count + len(generated)}}


def validate_goal(draft, question, text, profile, selected_rubric):
    if draft.rubric != selected_rubric:
        raise ValueError("Goal must target the selected lowest-scored rubric")
    ids = [sentence.id for sentence in profile.sentences]
    if any(sid not in ids for sid in draft.target_sents):
        raise ValueError("Unknown target sentence")
    indexes = [ids.index(sid) for sid in draft.target_sents]
    if indexes != list(range(indexes[0], indexes[-1] + 1)):
        raise ValueError("Target sentences must be unique, ordered and contiguous")
    for evidence in draft.evidence:
        source = question if evidence.source == "question" else text
        if not evidence.quote.strip() or evidence.quote not in source:
            raise ValueError("Goal evidence must quote its named source exactly")
    for change in draft.allowed_changes:
        if not change.scope_quote.strip() or change.scope_quote not in text:
            raise ValueError("Allowed-change scope must quote the draft")
        if not change.expected.strip() or not change.reason.strip():
            raise ValueError("Allowed changes need an expectation and justification")
    start, end = profile.sentences[indexes[0]].start, profile.sentences[indexes[-1]].end
    return Goal(**draft.model_dump(), span_before=[start, end])


def evidence_catalog(question, profile):
    return {"question": {"source": "question", "quote": question},
            **{f"draft:{s.id}": {"source": "draft", "quote": s.text} for s in profile.sentences}}


def expand_goal_range(selection, profile, question):
    """Resolve chosen endpoints and evidence IDs into exact source spans and quotations."""
    ids = [sentence.id for sentence in profile.sentences]
    if selection.first_sentence not in ids or selection.last_sentence not in ids:
        raise ValueError("Unknown target sentence endpoint")
    first, last = ids.index(selection.first_sentence), ids.index(selection.last_sentence)
    if first > last:
        raise ValueError("Target sentence endpoints are reversed")
    catalog = evidence_catalog(question, profile)
    if any(key not in catalog for key in selection.evidence_ids):
        raise ValueError("Unknown source evidence id")
    sentences = {sentence.id: sentence for sentence in profile.sentences}
    allowed_changes = []
    for change in selection.allowed_changes:
        if change.scope_sentence not in sentences:
            raise ValueError("Unknown allowed-change scope sentence")
        allowed_changes.append(AllowedChange(scope_quote=sentences[change.scope_sentence].text,
                                             expected=change.expected, reason=change.reason))
    return GoalDraft(rubric=selection.rubric, intent=selection.intent, target_sents=ids[first:last + 1],
                     evidence=[Evidence(**catalog[key]) for key in dict.fromkeys(selection.evidence_ids)],
                     allowed_changes=allowed_changes)


def set_goal(question, text, diagnosis, profile, *, llm, prompt):
    scores = diagnosis["scores"]
    if set(scores) != set(RUBRICS) or any(type(v) is not int or not 1 <= v <= 9 for v in scores.values()):
        raise ValueError("Goal selection requires eight native integer rubric scores")
    selected = min(RUBRICS, key=lambda key: scores[key])
    def validate(response):
        if response.goal is not None:
            validate_goal(expand_goal_range(response.goal, profile, question), question, text, profile, selected)
    response = llm.request(GoalResponse, prompt, {
        "question": question, "draft": text, "rubric_feedback": diagnosis["feedback"], "selected_rubric": selected,
        "profile": profile.to_dict(),
        "source_evidence": evidence_catalog(question, profile),
    }, role="goal", retries=0, validate=validate)
    goal = (validate_goal(expand_goal_range(response.goal, profile, question), question, text, profile, selected)
            if response.goal is not None else None)
    return goal, response.reason
