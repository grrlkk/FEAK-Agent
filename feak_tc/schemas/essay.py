"""Canonical essay schemas for AI-Hub ingestion."""

from typing import Any, Dict, Optional, Union

import pydantic
from pydantic import BaseModel, Field

try:  # pydantic v2
    from pydantic import ConfigDict, field_validator
except ImportError:  # pydantic v1
    ConfigDict = None
    from pydantic import validator as field_validator


_PYDANTIC_V2 = int(pydantic.VERSION.split(".", 1)[0]) >= 2


class EssayRecord(BaseModel):
    """Normalized essay record stored in the `essays` collection."""

    if _PYDANTIC_V2:
        model_config = ConfigDict(extra="forbid")
    else:
        class Config:
            extra = "forbid"

    essay_id: str
    prompt: Optional[str] = None
    topic: Optional[str] = None
    grade: Optional[Union[int, str]] = None
    purpose: Optional[str] = None
    text: str
    features: Dict[str, Any] = Field(default_factory=dict)
    rubric_scores_raw: Dict[str, Any] = Field(default_factory=dict)
    rubric_scores_mean: Dict[str, float] = Field(default_factory=dict)
    expert_feedback: Any = Field(default_factory=list)
    rubric_definitions: Any = Field(default_factory=dict)
    raw_path: str

    @field_validator("essay_id", "text", "raw_path")
    def _strip_required_strings(cls, value: str) -> str:
        return value.strip()

    @field_validator("prompt", "topic", "purpose")
    def _strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    def to_mongo_document(self) -> Dict[str, Any]:
        """Return a JSON-compatible document for MongoDB insertion."""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()
