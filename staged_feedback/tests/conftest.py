"""Synthetic sentences only: no development/evaluation essay is a test fixture."""
import re

import pytest

from staged_feedback.edits import Executor


def sentences(text):
    # Mirror the native punctuation/newline contract, without a legacy dependency.
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
def executor():
    return Executor(sentence_spans=sentences,
        pos=lambda text: [(word, "NNG") for word in re.findall(r"[가-힣]+", text)],
        marker_pattern=re.compile(r"#@[^#\r\n]+#"),
        placeholder_pattern=re.compile(r"__ANON_[A-Za-z0-9]+_[A-Za-z0-9]+__"))


def issue(identifier="i1", source="읽을수", expected="읽을 수"):
    from staged_feedback.models import Issue
    return Issue(id=identifier, level="micro", description="띄어쓰기를 고칩니다.",
        evidence_quotes=[source], priority=1,
        test=dict(kind="span_replaced", source_quote=source, expected=expected,
                  checklist=f"‘{source}’를 ‘{expected}’로 고쳤나요?"))


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


class ScriptedLLM:
    def __init__(self, *, extend=False, broken=False, omit_retry=False):
        self.calls, self.extend, self.broken, self.omit_retry = [], extend, broken, omit_retry

    def request(self, role, payload, schema, *, context):
        self.calls.append(dict(role=role, payload=payload, context=context))
        turn = context["turn"]
        problems = [issue()]
        if self.extend:
            problems.append(issue("i2", "할수", "할 수"))
        if role == "coordinator_initial":
            forced = payload.get("forced_initial_budget")
            budget = forced or 1
            response = dict(difficulty=dict(level="easy", revision_amount="small", reason="표기 수정이 필요합니다.",
                diagnostic_evidence=[dict(source="kanana", key="내용", value=5.0, explanation="측정된 점수입니다.")],
                evidence_quotes=["읽을수"]), strategy=dict(name="front_first" if forced else "single_pass", reason="앞에서부터 표기를 고칩니다."),
                turn_plan=[dict(turn=i, direction="표기 수정", issue_ids=[p.id for p in problems]) for i in range(1, budget + 1)],
                issues=[p.model_dump() for p in problems], commitments=[dict(statement="말투와 주장을 유지합니다.")],
                turn_budget=budget, reason="작은 표현 오류를 다룹니다.", student_tasks=[])
        elif role == "feedback":
            problem = problems[min(turn - 1, len(problems) - 1)]
            response = dict(items=[dict(issue_id=problem.id, target_quote=problem.evidence_quotes[0],
                what="띄어쓰기", how="단어 사이를 띄우세요.", student_explanation="의존 명사 앞을 띄어 써 보세요.",
                test=problem.test.model_dump(), feature_evidence=[dict(source="feature", key="length",
                    value=payload["current_features"]["features"]["length"], explanation="현재 글의 길이를 확인했습니다.")])], student_tasks=[])
        elif role == "reviser" and payload.get("mode") == "single_call_staged":
            response = dict(stages=[dict(feedback=["표기를 고치세요."], checklist=["띄어쓰기를 확인했나요?"],
                revised_text=payload["current_text"].replace("읽을수", "읽을 수")) for _ in range(2)], student_tasks=[])
        elif role == "reviser":
            problem = problems[min(turn - 1, len(problems) - 1)]
            quote = "없는 인용" if self.broken else problem.evidence_quotes[0]
            ops = [dict(id="op1", issue_ids=[problem.id], op="replace", quote=quote, replacement=problem.test.expected)]
            response = dict(operations=[] if self.omit_retry and "retry" in payload else ops, student_tasks=[])
        elif role == "coordinator_review":
            more = self.extend and turn == 1 and payload["max_turns"] > 1
            budget = 2 if more else payload["current_turn_budget"]
            response = dict(resolved_issue_ids=[p.id for p in problems], new_issues=[], priorities=[], dispositions=[],
                turn_budget=budget, new_commitments=[], continue_work=more, reason="다음 표기를 고칩니다." if more else "검토를 마쳤습니다.",
                strategy=dict(name="front_first", reason="남은 표현을 다음 턴에 다룹니다.") if more else None,
                turn_plan=[dict(turn=1, direction="첫 표기 수정", issue_ids=["i1"]),
                           dict(turn=2, direction="다음 표기 수정", issue_ids=["i2"])] if more else None,
                student_tasks=[])
        else:
            raise AssertionError(role)
        return schema.model_validate(response)
