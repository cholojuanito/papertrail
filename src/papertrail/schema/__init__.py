"""The integration contract. Import shapes from here: `from papertrail.schema import ...`."""

from .eligibility import EligibilityStatus, EligibilityVerdict, LineEligibility
from .extraction import (
    EXTRACTION_INSTRUCTION,
    ExtractedExpense,
    LineItem,
    ProviderType,
)
from .results import ProcessedExpense
from .scoring import ReadinessScore

__all__ = [
    "EXTRACTION_INSTRUCTION",
    "ExtractedExpense",
    "LineItem",
    "ProviderType",
    "EligibilityStatus",
    "EligibilityVerdict",
    "LineEligibility",
    "ReadinessScore",
    "ProcessedExpense",
]
