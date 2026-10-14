"""Runtime contracts, separate from the historical four-axis RV dataset."""

from typing import Literal, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

from feak_tc.diagnose.constants import RUBRIC_KEYS


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class RevisionRequest(StrictModel):
    action_type: Literal["ADD_DETAIL", "DELETE_OR_FOCUS", "COMPRESS", "RESTRUCTURE", "STYLE_REFINE"]
    target_rubric: str
    target_span: str = Field(min_length=1)
    problem: str = Field(min_length=1)
    instruction: str = Field(min_length=1)
    preserve: list[str] = Field(min_length=1)

    @field_validator("target_rubric")
    @classmethod
    def known_rubric(cls, value):
        if value not in RUBRIC_KEYS:
            raise ValueError("Unknown rubric")
        return value

    @field_validator("target_span", "problem", "instruction")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Must not be blank")
        return value

    @field_validator("preserve")
    @classmethod
    def preservation_constraints(cls, value):
        if any(not item.strip() for item in value):
            raise ValueError("Preservation constraints must not be blank")
        return value


class PlanResponse(StrictModel):
    plan: Optional[RevisionRequest]
    reason: str = Field(min_length=1)


class AxisVerdict(StrictModel):
    # null means unassessable, not partial success.
    label: Optional[Literal["pass", "partial", "fail"]]
    reason: str = Field(min_length=1)


class RevisionVerdict(StrictModel):
    target_fulfillment: AxisVerdict
    preservation: AxisVerdict


class GuardVerdict(StrictModel):
    preservation: AxisVerdict
    coherence: AxisVerdict


class LocalModelConfig(StrictModel):
    model: str
    device: str
    load_in_4bit: bool
    max_input_tokens: int = Field(gt=0)
    max_new_tokens: int = Field(gt=0)
    max_calls: int = Field(gt=0)
    temperature: float = Field(ge=0)
    seed: int = Field(ge=0)


class ControllerConfig(StrictModel):
    max_steps: int = Field(gt=0)
    candidates_per_step: int = Field(gt=0)
    no_progress_patience: int = Field(gt=0)
    max_rollbacks: int = Field(gt=0)
    schema_retries: int = Field(ge=0)
    memory_steps: int = Field(gt=0)
    rv_target_labels: list[Literal["pass", "partial"]] = Field(min_length=1)
    rv_preservation_labels: list[Literal["pass", "partial"]] = Field(min_length=1)
    quality_drop_max: float = Field(ge=0)
    # Relative to the heuristic score for keeping the current state.
    noop_margin: float = Field(ge=0)
    guard_quality_drop_max: float = Field(ge=0)
    guard_require_pass: bool
