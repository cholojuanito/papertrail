"""Audit Readiness Score shape — produced by `core/scoring`."""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReadinessScore(BaseModel):
    score: int  # 0-100
    breakdown: dict[str, int] = Field(default_factory=dict)  # field -> points deducted
    missing: list[str] = Field(default_factory=list)  # field names that scored 0
