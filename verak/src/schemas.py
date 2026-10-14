"""Explicit source offsets and JSON contracts for the phase 1 pipeline."""

from dataclasses import asdict, dataclass, field
from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

RUBRICS = ("과제충실성", "설명명료성", "설명구체성", "설명적절성", "문장연결성", "글통일성", "어휘적절성", "어법적절성")
CONDITIONS = ("criteria_only", "surface_diff", "korean")
Rubric = Literal["과제충실성", "설명명료성", "설명구체성", "설명적절성", "문장연결성", "글통일성", "어휘적절성", "어법적절성"]
Label = Literal["pass", "fail", "unknown"]


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Evidence(Strict):
    source: Literal["question", "draft"]
    quote: str = Field(min_length=1)


class AllowedChange(Strict):
    scope_quote: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GoalContent(Strict):
    rubric: Rubric
    intent: str = Field(min_length=1)

    @field_validator("intent")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("A goal must have an intent")
        return value


class GoalDraft(GoalContent):
    target_sents: list[str] = Field(min_length=1)
    evidence: list[Evidence] = Field(min_length=1)
    allowed_changes: list[AllowedChange]


class PlannedChange(Strict):
    scope_sentence: str = Field(min_length=1)
    expected: str = Field(min_length=1)
    reason: str = Field(min_length=1)


class GoalRange(GoalContent):
    first_sentence: str = Field(min_length=1)
    last_sentence: str = Field(min_length=1)
    evidence_ids: list[str] = Field(min_length=1, max_length=4)
    allowed_changes: list[PlannedChange]


class Goal(GoalDraft):
    span_before: list[int] = Field(min_length=2, max_length=2)


class GoalResponse(Strict):
    goal: Optional[GoalRange]
    reason: str = Field(min_length=1)


class Replacement(Strict):
    replacement: str


class Issue(Strict):
    requirement: Literal["hard", "goal_valid", "goal_improved", "selective", "meaning", "global"]
    location: str = Field(min_length=1)
    before_quote: str
    after_quote: str
    reason: str = Field(min_length=1)


class UnitJudgment(Strict):
    goal_valid: Label
    goal_improved: Label
    selective: Label
    meaning: Label
    issues: list[Issue]


class GlobalIssue(Strict):
    location: str = Field(min_length=1)
    a_quote: str
    b_quote: str
    reason: str = Field(min_length=1)


class GlobalJudgment(Strict):
    preference: Literal["A", "B", "equal", "unknown"]
    issues: list[GlobalIssue]


# The active loop uses these contracts; P1 schemas remain available for replay.
Scope = Literal["morpheme", "sentence", "span", "paragraph", "document"]
Action = Literal["ADD", "DELETE", "REWRITE", "REORDER"]


class TargetSelection(Strict):
    first_id: str = Field(min_length=1)
    last_id: str = Field(min_length=1)
    position: Literal["before", "after", "replace"]


class ScopePlanSelection(Strict):
    rubric: Rubric
    goal: str = Field(min_length=1)
    scope: Scope
    target: TargetSelection
    action: Action
    preserve_ids: list[str]
    evidence_ids: list[str] = Field(min_length=1, max_length=4)
    minimal_scope_reason: str = Field(min_length=1)


class ScopePlanResponse(Strict):
    plan: Optional[ScopePlanSelection]
    reason: str = Field(min_length=1)


class SourcePiece(Strict):
    id: str
    start: int
    end: int
    text: str


class ResolvedTarget(Strict):
    start: int
    end: int
    text: str
    insertion_at: Optional[int]
    pieces: list[SourcePiece]


class ScopePlan(Strict):
    rubric: Rubric
    goal: str
    scope: Scope
    target: ResolvedTarget
    action: Action
    preserve: list[str]
    evidence: list[Evidence]
    minimal_scope_reason: str

    @property
    def span_before(self):
        if self.action == "ADD":
            return [self.target.insertion_at, self.target.insertion_at]
        return [self.target.start, self.target.end]


class ReorderResponse(Strict):
    order: list[str] = Field(min_length=2)


class ScopeIssue(Strict):
    requirement: Literal["goal", "selectivity", "preservation", "korean_consistency"]
    location: str = Field(min_length=1)
    before_quote: str
    after_quote: str
    reason: str = Field(min_length=1)


class ScopeJudgment(Strict):
    goal: Label
    selectivity: Label
    preservation: Label
    korean_consistency: Label
    issues: list[ScopeIssue]


EditRequirement = Literal["necessity", "preservation", "groundedness", "meaning", "korean_consistency"]


class EditIssue(Strict):
    requirement: EditRequirement
    before_quote: str
    after_quote: str
    reason: str = Field(min_length=1)


class EditJudgment(Strict):
    edit_id: str = Field(min_length=1)
    necessity: Label
    preservation: Label
    groundedness: Label
    meaning: Label
    korean_consistency: Label
    reason: str = Field(min_length=1)
    issues: list[EditIssue]


class EditJudgments(Strict):
    edits: list[EditJudgment] = Field(min_length=1)


@dataclass
class Token:
    form: str
    tag: str
    start: int
    end: int


@dataclass
class Sentence:
    id: str
    paragraph: int
    start: int
    end: int
    text: str
    tokens: list[Token]
    style_candidates: list[str] = field(default_factory=list)
    connectives: list[dict] = field(default_factory=list)
    subjects: list[dict] = field(default_factory=list)
    antecedent_candidates: list[dict] = field(default_factory=list)
    uncertain: list[str] = field(default_factory=list)


@dataclass
class Profile:
    sentences: list[Sentence]
    dominant_style: list[str]
    analyzer: str
    analyzer_version: str
    uncertain: list[str] = field(default_factory=list)

    def to_dict(self):
        return asdict(self)


@dataclass
class Unit:
    id: str
    before_span: list[int]
    after_span: list[int]
    before_text: str
    after_text: str
    types: list[str]
    observed: list[dict]
    interpretations: list[dict]
    uncertain: list[str]
    alignment: str
    surface_diff: list[dict]
    spelling: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)
