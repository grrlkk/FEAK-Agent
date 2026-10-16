"""Response shapes, not validators of student writing or revision quality."""
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class Issue(Strict):
    id: str
    level: Literal["macro", "micro"]
    macro_type: Literal["restructure", "student_input"] | None = Field(default=None,
        description="거시 이슈는 restructure 또는 student_input, 미시 이슈는 null.")
    restructure_issue_ids: list[str] = Field(default_factory=list,
        description="미시 이슈의 대상 문장을 먼저 재배열해야 하는 restructure 이슈 ID. 관련 없는 거시 이슈와 student_input은 넣지 않는다.")
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
    remaining_turns: int = Field(ge=0, description="앞으로 진행할 턴 수. 수정 이슈가 없으면 0, single_pass는 1.")
    reason: str
    student_tasks: list[StudentTask] = Field(default_factory=list)
    optional_suggestions: list[str] = Field(default_factory=list,
        description="있으면 더 좋은 내용에 대한 선택 제안. 필수 이슈나 학생 과제가 아니며 결과 끝에 제시한다.")


class FeedbackItem(Strict):
    issue_id: str
    target_quote: str
    target_ids: list[str] = Field(default_factory=list,
        description="document_units의 대상 ID. 미시는 s 문장 ID, 거시는 p 문단 ID. 필요한 범위만 선택한다.")
    what: str
    how: str
    student_explanation: str
    feature_evidence: list[DiagnosticEvidence] = Field(default_factory=list)
    needs_student_input: bool = False


class Feedback(Strict):
    items: list[FeedbackItem]


class Revision(Strict):
    revised_text: str = Field(description="source_text 범위의 수정본만. 앞뒤 맥락이나 글 전체를 넣지 않는다.")
    cannot_apply: str | None = Field(default=None,
        description="학생의 내용 없이는 반영할 수 없으면 이유를 적는다. 이 경우 해당 범위는 원문을 유지한다.")


class Assessment(Strict):
    issue_id: str
    status: Literal["applied", "partial", "not_applied"]
    reason: str


class PriorityUpdate(Strict):
    issue_id: str
    priority: int = Field(ge=1)
    reason: str


class Restore(Strict):
    previous_sentence_ids: list[str] = Field(description="이전 버전에서 되살릴 연속 문장 ID. previous_document_units의 s ID를 쓴다.")
    current_sentence_ids: list[str] = Field(description="그 문장으로 되돌릴 현재 버전의 연속 문장 ID. 삭제 문장 재삽입은 빈 목록.")
    insert_before_sentence_id: str | None = Field(default=None,
        description="재삽입할 때 이 현재 문장 ID 앞에 넣는다. null이면 글 끝. 문장 교체 시 null.")
    reason: str


class Review(Strict):
    assessments: list[Assessment]
    reopened: list[str]
    new_issues: list[Issue]
    priorities: list[PriorityUpdate]
    strategy: Strategy | None = None
    turn_plan: list[TurnDirection] | None = None
    remaining_turns: int = Field(ge=0, description="이번 검토가 끝난 뒤 앞으로 진행할 턴 수. 종료는 0.")
    continue_work: bool
    reason: str
    student_tasks: list[StudentTask] = Field(default_factory=list)
    optional_suggestions: list[str] = Field(default_factory=list,
        description="있으면 더 좋은 내용에 대한 선택 제안. student_tasks와 수정 이슈에 넣지 않는다.")
    restore: list[Restore] = Field(default_factory=list)


class BaselineStage(Strict):
    feedback: list[str]
    revised_text: str


class SingleCallStages(Strict):
    stages: list[BaselineStage]
    student_tasks: list[StudentTask]
    optional_suggestions: list[str] = Field(default_factory=list)
