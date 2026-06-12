"""Audit Readiness Score (0-100). Pure function of which fields are present.

Each field carries a point weight; missing fields deduct. The breakdown is what
the UI panel renders ("Itemised line items (-15)").
"""

from __future__ import annotations

from papertrail.schema import ExtractedExpense, ProviderType, ReadinessScore

# field -> points it is worth (sum = 100)
WEIGHTS: dict[str, int] = {
    "merchant": 20,
    "date": 20,
    "amount": 25,
    "provider_type": 10,
    "line_items": 15,
    "patient": 10,
}


def _present(expense: ExtractedExpense) -> dict[str, bool]:
    return {
        "merchant": bool(expense.merchant),
        "date": bool(expense.date),
        "amount": expense.amount is not None,
        "provider_type": expense.provider_type != ProviderType.other,
        "line_items": len(expense.line_items) > 0,
        "patient": bool(expense.patient),
    }


def score(expense: ExtractedExpense) -> ReadinessScore:
    present = _present(expense)
    breakdown: dict[str, int] = {}
    missing: list[str] = []
    total = 100
    for field, weight in WEIGHTS.items():
        if not present[field]:
            breakdown[field] = weight
            missing.append(field)
            total -= weight
    return ReadinessScore(score=max(0, total), breakdown=breakdown, missing=missing)
