"""The fully-processed result: what the pipeline returns and the UI renders.

Bundles the model's extraction with the deterministic verdicts derived from it,
plus the storage id once persisted. This is the single object that flows
inference -> core -> storage -> app.
"""

from __future__ import annotations

from pydantic import BaseModel

from .eligibility import EligibilityVerdict
from .extraction import ExtractedExpense
from .scoring import ReadinessScore


class ProcessedExpense(BaseModel):
    expense: ExtractedExpense
    eligibility: EligibilityVerdict
    readiness: ReadinessScore
    document_id: int | None = None
    expense_id: int | None = None
