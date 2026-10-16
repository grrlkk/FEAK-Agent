"""Strict contracts shared by structured LLM responses and the controller."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class TestSpec(Strict):
    kind: Literal["absent", "count_at_most", "span_replaced", "sentence_contains", "macro"]
    expression: str | None = None
    limit: int | None = Field(default=None, ge=0)
    source_quote: str | None = None
    expected: str | None = None
    sentence_quote: str | None = None
    target_quotes: list[str] = Field(default_factory=list)
    question: str | None = None
    checklist: str = Field(min_length=1)

    @model_validator(mode="after")
    def fields_for_kind(self):
        required = {
            "absent": [self.expression],
            "count_at_most": [self.expression, self.limit],
            "span_replaced": [self.source_quote, self.expected],
            "sentence_contains": [self.sentence_quote, self.expression],
            "macro": [self.question, self.target_quotes],
        }[self.kind]
        if any(value is None or value == "" or value == [] for value in required):
            raise ValueError(f"Missing fields for {self.kind} test")
        return self


class Issue(Strict):
    id: str = Field(min_length=1)
    level: Literal["macro", "micro"]
    description: str = Field(min_length=1)
    evidence_quotes: list[str] = Field(min_length=1)
    priority: int = Field(ge=1)
    test: TestSpec

    @model_validator(mode="after")
    def matching_level(self):
        if (self.level == "macro") != (self.test.kind == "macro"):
            raise ValueError("Issue level must match its test kind")
        return self


class Commitment(Strict):
    statement: str = Field(min_length=1)
    quote: str | None = None


class StudentTask(Strict):
    issue_id: str | None = None
    instruction: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class DiagnosticEvidence(Strict):
    source: Literal["kanana", "feature"]
    key: str
    value: float | None
    explanation: str = Field(min_length=1)


class Difficulty(Strict):
    level: Literal["easy", "moderate", "hard"]
    revision_amount: Literal["small", "moderate", "large"]
    reason: str = Field(min_length=1)
    diagnostic_evidence: list[DiagnosticEvidence] = Field(min_length=1)
    evidence_quotes: list[str] = Field(min_length=1)


class Strategy(Strict):
    name: Literal["front_first", "macro_first", "single_pass"]
    reason: str = Field(min_length=1)


class TurnDirection(Strict):
    turn: int = Field(ge=1, le=4)
    direction: str = Field(min_length=1)
    issue_ids: list[str]


class InitialPlan(Strict):
    difficulty: Difficulty
    strategy: Strategy
    turn_plan: list[TurnDirection] = Field(min_length=1)
    issues: list[Issue]
    commitments: list[Commitment]
    turn_budget: int = Field(ge=1, le=4)
    reason: str = Field(min_length=1)
    student_tasks: list[StudentTask] = Field(default_factory=list)


class FeedbackItem(Strict):
    issue_id: str = Field(min_length=1)
    target_quote: str = Field(min_length=1)
    what: str = Field(min_length=1)
    how: str = Field(min_length=1)
    student_explanation: str = Field(min_length=1)
    test: TestSpec
    feature_evidence: list[DiagnosticEvidence] = Field(default_factory=list)

    @model_validator(mode="after")
    def features_only(self):
        if any(item.source != "feature" for item in self.feature_evidence):
            raise ValueError("Feedback evidence uses current features, not new Kanana scoring")
        return self


class Feedback(Strict):
    items: list[FeedbackItem]
    student_tasks: list[StudentTask] = Field(default_factory=list)


class Position(Strict):
    kind: Literal["before", "after", "paragraph_start", "paragraph_end"]
    anchor_quote: str | None = None
    paragraph: int | None = Field(default=None, ge=1)


class Operation(Strict):
    id: str = Field(min_length=1)
    issue_ids: list[str] = Field(min_length=1)
    op: Literal["replace", "move", "merge", "split", "connect"]
    quote: str | None = None
    replacement: str | None = None
    sentence_a: str | None = None
    sentence_b: str | None = None
    sentences: list[str] | None = None
    position: Position | None = None
    connector: str | None = None


class Revision(Strict):
    operations: list[Operation]
    student_tasks: list[StudentTask] = Field(default_factory=list)


class MacroVerdict(Strict):
    issue_id: str
    passed: bool
    reason: str = Field(min_length=1)
    evidence_quotes: list[str]


class MacroVerdicts(Strict):
    results: list[MacroVerdict]


class PriorityUpdate(Strict):
    issue_id: str
    priority: int = Field(ge=1)
    reason: str


class Disposition(Strict):
    issue_id: str
    action: Literal["retry", "defer", "student_task"]
    reason: str


class Review(Strict):
    strategy: Strategy | None = None
    turn_plan: list[TurnDirection] | None = None
    resolved_issue_ids: list[str]
    new_issues: list[Issue]
    priorities: list[PriorityUpdate]
    dispositions: list[Disposition]
    turn_budget: int = Field(ge=1, le=4)
    new_commitments: list[Commitment]
    continue_work: bool
    reason: str = Field(min_length=1)
    student_tasks: list[StudentTask] = Field(default_factory=list)


class BaselineStage(Strict):
    feedback: list[str]
    checklist: list[str]
    revised_text: str = Field(min_length=1)


class SingleCallStages(Strict):
    stages: list[BaselineStage] = Field(min_length=2, max_length=3)
    student_tasks: list[StudentTask]


class TestResult(Strict):
    issue_id: str
    kind: str
    passed: bool
    status: Literal["pass", "fail", "error"]
    reason: str
    checklist: str
    evidence_quotes: list[str] = Field(default_factory=list)
