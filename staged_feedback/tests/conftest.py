"""Synthetic sentences only; no evaluation essay is a test fixture."""
import re

import pytest

from staged_feedback.observations import Observer


def sentences(text):
    boundaries = [m.end() for m in re.finditer(r'[.!?。！？][\"\'”’」』)\]]*(?=\s|$)|\r?\n', text)]
    result, start = [], 0
    for end in [*boundaries, len(text)]:
        a, b = start, end
        while a < b and text[a].isspace():
            a += 1
        while b > a and text[b - 1].isspace():
            b -= 1
        if a < b:
            result.append(dict(id=f"s{len(result)+1}", text=text[a:b], start=a, end=b))
        start = end
    return result


@pytest.fixture
def observer():
    return Observer(sentence_spans=sentences, marker_pattern=re.compile(r"#@[^#\r\n]+#"),
                    placeholder_pattern=re.compile(r"__ANON_[A-Za-z0-9]+_[A-Za-z0-9]+__"))


class Anonymizer:
    """Boundary stand-in; any call to a quality validator fails the test."""
    def __init__(self, *args):
        self.mapping = {}

    def mask_data(self, value):
        if isinstance(value, str):
            def replace(match):
                token = f"__ANON_test_{len(self.mapping):06d}__"
                self.mapping[token] = match.group()
                return token
            return re.sub(r"#@[^#\r\n]+#", replace, value)
        if isinstance(value, dict):
            return {k: self.mask_data(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.mask_data(v) for v in value]
        return value

    def restore_data(self, value):
        if hasattr(value, "model_dump"):
            value = value.model_dump()
        if isinstance(value, str):
            for key, text in self.mapping.items():
                value = value.replace(key, text)
            return value
        if isinstance(value, dict):
            return {k: self.restore_data(v) for k, v in value.items()}
        if isinstance(value, list):
            return [self.restore_data(v) for v in value]
        return value

    def validate_revision(self, *args):
        raise AssertionError("Revision validator must never run")


class Diagnostic:
    def __init__(self):
        self.scored, self.extracted = [], []

    def __call__(self, text, prompt, essay_id):
        self.scored.append(text)
        return dict(scores={"내용": 5.0}, features={"length": len(text)},
                    kb_items=[], score_metadata={"n": 32}, mock=True)

    def features(self, text):
        self.extracted.append(text)
        return dict(features={"length": len(text)}, kb_items=[], mock=True)


def issue(identifier="i1", quote="읽을수"):
    from staged_feedback.models import Issue
    return Issue(id=identifier, level="micro", description="띄어쓰기를 고칩니다.",
                 evidence_quotes=[quote], priority=1)


class ScriptedLLM:
    def __init__(self, *, count=1, turns=1, revisions=None, assessments=None):
        self.calls, self.count, self.turns = [], count, turns
        self.revisions, self.assessments = revisions or [], assessments

    def request(self, role, payload, schema, *, context):
        self.calls.append(dict(role=role, payload=payload, context=context))
        turn = context["turn"]
        if role == "coordinator_initial":
            problems = [issue(f"i{i+1}") for i in range(self.count)]
            forced = payload.get("forced_initial_budget")
            budget = forced or self.turns
            response = dict(difficulty=dict(level="easy", revision_amount="small", reason="표기를 다듬습니다.",
                diagnostic_evidence=[dict(source="kanana", key="내용", value=5.0, explanation="측정 점수")],
                evidence_quotes=["읽을수"]), strategy=dict(name="single_pass" if budget == 1 else "front_first", reason="표기 수정"),
                turn_plan=[dict(turn=t, direction="표기 수정", issue_ids=[p.id for p in problems], feedback_count=self.count)
                           for t in range(1, budget + 1)], issues=[p.model_dump() for p in problems],
                turn_budget=budget, reason="표기 문제", student_tasks=[])
        elif role == "feedback":
            response = dict(items=[dict(issue_id=p["id"], target_quote=p["evidence_quotes"][0],
                what="띄어쓰기", how="읽을 수로 고치세요.", student_explanation="의존 명사 앞을 띄웁니다.", feature_evidence=[])
                for p in payload["candidates"]], student_tasks=[])
        elif role == "reviser" and payload.get("mode") == "single_call_staged":
            response = dict(stages=[dict(feedback=["띄어쓰기"], revised_text=payload["current_text"].replace("읽을수", "읽을 수"))
                                   for _ in range(2)], student_tasks=[])
        elif role == "reviser":
            revision = self.revisions[turn-1] if len(self.revisions) >= turn else lambda t: t.replace("읽을수", "읽을 수")
            response = dict(revised_text=revision(payload["current_text"]) if callable(revision) else revision, student_tasks=[])
        elif role == "coordinator_review":
            assessments = self.assessments if self.assessments is not None else [dict(issue_id=i["issue_id"],
                status="partial" if turn < self.turns else "applied", reason="현재 문장을 직접 읽고 판단했습니다.")
                for i in payload["feedback"]["items"]]
            response = dict(assessments=assessments, reopened_issue_ids=[], new_issues=[], priorities=[], dispositions=[],
                strategy=None, turn_plan=None, turn_budget=self.turns, continue_work=turn < self.turns,
                reason="다음 턴에서 남은 내용을 검토합니다." if turn < self.turns else "반영을 확인했습니다.", student_tasks=[])
        else:
            raise AssertionError(role)
        return schema.model_validate(response)
