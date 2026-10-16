"""Response shapes, not validators of student writing or revision quality."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Issue(Strict):
    id: str
    level: Literal["macro", "micro"]
    description: str
    evidence_quotes: list[str]
    priority: int = Field(ge=1)
    needs_student_content: bool = False
    student_content_type: Literal["new_content", "new_experience", "intent"] | None = None


class StudentTask(Strict):
    issue_id: str | None = None
    instruction: str
    reason: str


class DiagnosticEvidence(Strict):
    source: Literal["kanana", "feature"]
    key: str
    value: float | None
    explanation: str


class Difficulty(Strict):
    level: Literal["easy", "moderate", "hard"]
    revision_amount: Literal["small", "moderate", "large"]
    reason: str
    diagnostic_evidence: list[DiagnosticEvidence]
    evidence_quotes: list[str]


class Strategy(Strict):
    name: Literal["front_first", "macro_first", "single_pass"]
    reason: str


class TurnDirection(Strict):
    turn: int = Field(ge=1)
    direction: str
    issue_ids: list[str]
    feedback_count: int = Field(ge=0)


class InitialPlan(Strict):
    difficulty: Difficulty
    strategy: Strategy
    turn_plan: list[TurnDirection]
    issues: list[Issue]
    turn_budget: int = Field(ge=1)
    reason: str
    student_tasks: list[StudentTask] = Field(default_factory=list)


class FeedbackItem(Strict):
    issue_id: str
    target_quote: str
    what: str
    how: str
    student_explanation: str
    feature_evidence: list[DiagnosticEvidence] = Field(default_factory=list)


class Feedback(Strict):
    items: list[FeedbackItem]
    student_tasks: list[StudentTask] = Field(default_factory=list)


class Revision(Strict):
    # Empty output, splitting, merging and changes outside feedback are observable,
    # never grounds to reject a candidate or restore its predecessor.
    revised_text: str
    student_tasks: list[StudentTask] = Field(default_factory=list)


class Assessment(Strict):
    issue_id: str
    status: Literal["applied", "partial", "not_applied"]
    reason: str


class PriorityUpdate(Strict):
    issue_id: str
    priority: int = Field(ge=1)
    reason: str


class Disposition(Strict):
    issue_id: str
    action: Literal["retry", "defer", "student_task"]
    reason: str


class Review(Strict):
    assessments: list[Assessment]
    reopened_issue_ids: list[str]
    new_issues: list[Issue]
    priorities: list[PriorityUpdate]
    dispositions: list[Disposition]
    strategy: Strategy | None = None
    turn_plan: list[TurnDirection] | None = None
    turn_budget: int = Field(ge=1)
    continue_work: bool
    reason: str
    student_tasks: list[StudentTask] = Field(default_factory=list)


class BaselineStage(Strict):
    feedback: list[str]
    revised_text: str


class SingleCallStages(Strict):
    stages: list[BaselineStage]
    student_tasks: list[StudentTask]
