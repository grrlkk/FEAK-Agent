"""Canonical essay schemas for AI-Hub ingestion."""

from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EssayRecord(BaseModel):
    """Normalized essay record stored in the `essays` collection."""

    model_config = ConfigDict(extra="forbid")

    essay_id: str
    prompt: Optional[str] = None
    topic: Optional[str] = None
    grade: Optional[Union[str, int]] = None
    purpose: Optional[str] = None
    text: str
    features: Dict[str, Any] = Field(default_factory=dict)
    rubric_scores_raw: Dict[str, Any] = Field(default_factory=dict)
    rubric_scores_mean: Dict[str, float] = Field(default_factory=dict)
    expert_feedback: Any = Field(default_factory=list)
    rubric_definitions: Any = Field(default_factory=dict)
    raw_path: str

    @field_validator("essay_id", "text", "raw_path")
    @classmethod
    def _strip_required_strings(cls, value: str) -> str:
        return value.strip()

    @field_validator("prompt", "topic", "purpose")
    @classmethod
    def _strip_optional_strings(cls, value: Optional[str]) -> Optional[str]:
        if value is None:
            return None
        value = value.strip()
        return value or None

    def to_mongo_document(self) -> Dict[str, Any]:
        """Return a JSON-compatible document for MongoDB insertion."""

        return self.model_dump(mode="json")
