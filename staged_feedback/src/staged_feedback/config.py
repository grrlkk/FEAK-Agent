"""Only new configuration is read; legacy files are never rewritten."""
from pathlib import Path
from typing import Literal

from pydantic import Field
import yaml

from .models import Strict


class Settings(Strict):
    condition: Literal["planned", "planned_no_memory", "routed", "always_direct", "always_staged", "single_call_staged", "routed_no_memory"] = "planned"
    legacy_root: str = "/home/chanwoo/Feak_next/para_agent"
    legacy_config: str = "configs/values_agent_luna.yaml"
    prompts: str = "prompts/staged_feedback"
    model: str = "gpt-6-luna"
    coordinator_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "low"
    worker_reasoning_effort: Literal["none", "low", "medium", "high", "xhigh", "max"] = "none"
    temperature: float = Field(default=0, ge=0, le=2)
    max_output_tokens: int = Field(default=8192, ge=256)
    timeout_s: float = Field(default=180, gt=0)
    max_turns: int = Field(default=4, ge=1, le=4)
    feedback_per_turn: int = Field(default=2, ge=1, le=2)
    operation_retries: int = Field(default=1, ge=0, le=1)
    score_n: Literal[32] = 32
    budget_usd: float = Field(default=5, gt=0, le=5)
    max_api_calls: int = Field(default=150, ge=1)
    max_input_chars: int = Field(default=100000, ge=1000)
    mock: bool = False
    seed: int = 161042


def load_settings(path):
    return Settings.model_validate(yaml.safe_load(Path(path).read_text(encoding="utf-8")))
