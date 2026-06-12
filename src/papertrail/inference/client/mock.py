"""Mock backend: returns a realistic ExtractedExpense with no GPU/network.

Same return type as the live client, so the UI and pipeline behave identically.
Select with `PAPERTRAIL_MOCK=true`.
"""

from __future__ import annotations

from papertrail.schema import EXTRACTION_INSTRUCTION, ExtractedExpense, LineItem, ProviderType

from .base import BasePaperTrailClient


class MockPaperTrailClient(BasePaperTrailClient):
    def extract(
        self, images: list[bytes], instruction: str = EXTRACTION_INSTRUCTION
    ) -> ExtractedExpense:
        return ExtractedExpense(
            merchant="City Pharmacy",
            date="2026-05-14",
            amount=42.50,
            provider_type=ProviderType.pharmacy,
            line_items=[
                LineItem(description="Lisinopril 10mg", amount=19.52),
                LineItem(description="Ibuprofen 200mg OTC", amount=22.98),
            ],
            patient="Jane Doe",
            confidence={"merchant": 0.95, "date": 0.98, "amount": 0.99},
        )
