"""Pure eligibility classification. No I/O, no model, no DB — just rules.

`classify(expense) -> EligibilityVerdict`: per-line verdicts plus an overall
status (all-eligible / all-ineligible / mixed=partial / unknown=flagged).
"""

from __future__ import annotations

from papertrail.schema import (
    EligibilityStatus,
    EligibilityVerdict,
    ExtractedExpense,
    LineEligibility,
    ProviderType,
)

from .table import KEYWORD_RULES, PROVIDER_RULES

E = EligibilityStatus


def classify_line(description: str, provider_type: ProviderType) -> LineEligibility:
    d = description.lower()
    for keywords, status, citation in KEYWORD_RULES:
        if any(k in d for k in keywords):
            return LineEligibility(description=description, status=status, citation=citation)
    status, citation = PROVIDER_RULES.get(provider_type, PROVIDER_RULES[ProviderType.other])
    return LineEligibility(description=description, status=status, citation=citation)


def _overall(statuses: set[EligibilityStatus]) -> EligibilityStatus:
    if statuses == {E.eligible}:
        return E.eligible
    if statuses == {E.ineligible}:
        return E.ineligible
    if E.eligible in statuses and E.ineligible in statuses:
        return E.partial
    return E.flagged


def classify(expense: ExtractedExpense) -> EligibilityVerdict:
    if not expense.line_items:
        status, citation = PROVIDER_RULES.get(
            expense.provider_type, PROVIDER_RULES[ProviderType.other]
        )
        return EligibilityVerdict(status=status, citation=citation, lines=[])

    lines = [classify_line(li.description, expense.provider_type) for li in expense.line_items]
    overall = _overall({line.status for line in lines})
    citation = "Mixed eligibility — see line items" if overall == E.partial else lines[0].citation
    return EligibilityVerdict(status=overall, citation=citation, lines=lines)
