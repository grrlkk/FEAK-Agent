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
