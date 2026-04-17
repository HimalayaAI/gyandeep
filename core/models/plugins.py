from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator


class PluginJobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    job_id: str = Field(min_length=1)
    plugin_id: str = Field(min_length=1)
    query: str = Field(min_length=1)
    context_text: str = ""
    mode: str = "environment"
    current_page: int = Field(ge=1)
    book_id: str | None = None
    output_dir: Path

    @field_validator("job_id", "plugin_id", "query")
    @classmethod
    def _strip_required_text(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be empty")
        return value

    @field_validator("mode")
    @classmethod
    def _validate_mode(cls, value: str) -> str:
        value = value.strip().lower()
        if value not in {"environment", "analyze"}:
            raise ValueError("mode must be environment or analyze")
        return value

    @field_validator("book_id")
    @classmethod
    def _strip_book_id(cls, value: str | None) -> str | None:
        if value is None:
            return None
        value = value.strip()
        return value or None


class PluginJobResult(BaseModel):
    model_config = ConfigDict(extra="forbid")

    plan_text: str = ""
    script_path: str | None = None
    video_path: str | None = None
