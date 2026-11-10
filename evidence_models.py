"""Pydantic contracts for search evidence and lightweight claim checks."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, Field, field_validator


class EvidenceSource(BaseModel):
    """Normalized source record passed between execution and response stages."""

    model_config = ConfigDict(extra="allow")

    source_id: str = Field(pattern=r"^S\d+$")
    title: str = ""
    url: str
    content: str = ""
    source_type: str = "other"
    published: str | None = None
    retrieved_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    extracted_title: str | None = None
    extracted_text: str | None = None
    extraction_status: str = "not_attempted"

    @field_validator("published", mode="before")
    @classmethod
    def normalize_published(cls, value):
        if value is None or isinstance(value, str):
            return value
        return str(value)

    @field_validator("url")
    @classmethod
    def validate_http_url(cls, value: str) -> str:
        parsed = urlsplit(value)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname:
            raise ValueError("evidence URL must be an absolute HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("evidence URL must not contain embedded credentials")
        return value


class EvidenceBundle(BaseModel):
    """Validated collection of evidence plus the search assessment."""

    model_config = ConfigDict(extra="forbid")

    sources: list[EvidenceSource] = Field(default_factory=list)
    assessment: dict[str, Any] = Field(default_factory=dict)


class ClaimEvidenceCheck(BaseModel):
    claim_type: str
    claim: str
    status: Literal["supported", "unsupported", "needs_review"]
    source_ids: list[str] = Field(default_factory=list)
    reason: str


class ValidationReport(BaseModel):
    valid: bool
    generation_ok: bool
    empty_answer: bool = False
    found_ids: list[str] = Field(default_factory=list)
    unknown_ids: list[str] = Field(default_factory=list)
    missing_url_ids: list[str] = Field(default_factory=list)
    evidence_warnings: list[str] = Field(default_factory=list)
    claim_checks: list[ClaimEvidenceCheck] = Field(default_factory=list)
    evidence_contract_errors: list[str] = Field(default_factory=list)
    semantic_grounding_checked: bool = False
    note: str = ""
