"""Eligibility verdict shapes — produced by `core/eligibility`, consumed by app/storage."""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class EligibilityStatus(str, Enum):
    eligible = "eligible"
    ineligible = "ineligible"
    partial = "partial"  # mix of eligible + ineligible line items
    flagged = "flagged"  # needs human review (unknown provider / item)


class LineEligibility(BaseModel):
    description: str
    status: EligibilityStatus
    citation: str  # the IRS rule reference behind this line's verdict


class EligibilityVerdict(BaseModel):
    status: EligibilityStatus  # overall verdict for the expense
    citation: str
    lines: list[LineEligibility] = Field(default_factory=list)
