"""Contracts for the September 2026 revision-verifier pilot."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from feak_tc.diagnose.constants import RUBRIC_KEYS


CRITERIA = ("goal_achievement", "necessity", "preservation", "global_benefit")


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class RevisionPlan(StrictModel):
    target_rubric: Literal["task_1", "content_1", "content_2", "content_3",
                           "organization_1", "organization_2", "expression_1", "expression_2"]
    target_span: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    goal: str = Field(min_length=1)
    must_preserve: list[str] = Field(min_length=1)

    @field_validator("target_rubric")
    @classmethod
    def known_rubric(cls, value):
        if value not in RUBRIC_KEYS:
            raise ValueError("Unknown rubric")
        return value

    @field_validator("target_span", "problem", "goal")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value

    @field_validator("must_preserve")
    @classmethod
    def nonblank_constraints(cls, values):
        if any(not value.strip() for value in values):
            raise ValueError("Preservation constraints must not be blank")
        return values


class PlanResponse(StrictModel):
    # An explicit null supplies the methodology's "no actionable issue" exit.
    plan: Optional[RevisionPlan]
    reason: str = Field(min_length=1)


class Revision(StrictModel):
    revised_text: str = Field(min_length=1)
    summary_of_change: str = Field(min_length=1)

    @field_validator("revised_text", "summary_of_change")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Text must not be blank")
        return value


class Criterion(StrictModel):
    label: Literal["PASS", "FAIL", "UNCERTAIN"]
    reason: str = Field(min_length=1)

    @field_validator("reason")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Reason must not be blank")
        return value


class RevisionVerdict(StrictModel):
    goal_achievement: Criterion
    necessity: Criterion
    preservation: Criterion
    global_benefit: Criterion
    decision: Literal["ACCEPT", "REJECT", "REVERIFY"]

    def rule_decision(self):
        labels = [getattr(self, name).label for name in CRITERIA]
        if "FAIL" in labels:
            return "REJECT"
        if "UNCERTAIN" in labels:
            return "REVERIFY"
        return "ACCEPT"

    @model_validator(mode="after")
    def consistent_decision(self):
        if self.decision != self.rule_decision():
            raise ValueError("decision must agree with the four criterion labels")
        return self


class ControllerConfig(StrictModel):
    max_iterations: int = Field(default=3, ge=1, le=3)
    max_retry_per_issue: int = Field(default=1, ge=0, le=1)
    max_reverify: int = Field(default=1, ge=0, le=1)
    schema_retries: int = Field(default=1, ge=0, le=2)
    log_score_baseline: bool = True


class OpenAIConfig(StrictModel):
    model: str = "gpt-5-mini"
    reasoning_effort: Literal["low", "medium", "high"] = "low"
    max_output_tokens: int = Field(default=4096, ge=256)
    max_input_chars: int = Field(default=30000, ge=1000)
    max_calls_per_sample: int = Field(default=24, ge=1)
    max_calls_total: int = Field(default=120, ge=1)
    timeout_s: float = Field(default=90.0, gt=0)


class KananaConfig(StrictModel):
    device_id: int = Field(default=0, ge=0)
    m: int = Field(default=3, ge=1)
    chunk_m: int = Field(default=1, ge=1)
    seed: int = Field(default=42, ge=0)
    load_in_4bit: bool = True
    generate_feedback: Literal[False] = False


class PilotConfig(StrictModel):
    openai: OpenAIConfig = Field(default_factory=OpenAIConfig)
    kanana: KananaConfig = Field(default_factory=KananaConfig)
    controller: ControllerConfig = Field(default_factory=ControllerConfig)
    diagnosis_timeout_s: float = Field(default=600.0, gt=0)
    score_source: Literal["rf_corrected_score", "soft_mean", "final"] = "rf_corrected_score"


class Sample(StrictModel):
    sample_id: str = Field(min_length=1)
    writing_prompt: str = Field(min_length=1)
    text: str = Field(min_length=1)

    @field_validator("sample_id", "writing_prompt", "text")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Sample fields must not be blank")
        return value
