"""Static HSA eligibility rules (IRC §213(d) / Pub 969 / Notice 2004-2, Pub 502 base).

Pure data. `classify.py` applies these. Keyword rules win over the provider-type
default so a non-medical item on a pharmacy receipt is still caught.
"""

from __future__ import annotations

from papertrail.schema import EligibilityStatus, ProviderType

E = EligibilityStatus

# provider_type -> (default status, citation)
PROVIDER_RULES: dict[ProviderType, tuple[EligibilityStatus, str]] = {
    ProviderType.pharmacy: (E.eligible, "IRS Pub 502 — Medicines"),
    ProviderType.dental: (E.eligible, "IRS Pub 502 — Dental Treatment"),
    ProviderType.vision: (E.eligible, "IRS Pub 502 — Eye Exams & Eyeglasses"),
    ProviderType.medical: (E.eligible, "IRS Pub 502 — Medical Services"),
    ProviderType.other: (E.flagged, "Manual review — provider type not recognized"),
}

# (keywords, status, citation) — first match on the lowercased line description wins.
KEYWORD_RULES: list[tuple[tuple[str, ...], EligibilityStatus, str]] = [
    (
        ("gym", "fitness", "membership", "personal training"),
        E.ineligible,
        "IRS Pub 502 — Health-club dues (not eligible)",
    ),
    (
        ("cosmetic", "whitening", "anti-aging", "botox", "face cream"),
        E.ineligible,
        "IRS Pub 502 — Cosmetic procedures (not eligible)",
    ),
    (
        ("vitamin", "supplement", "collagen", "protein", "multivitamin"),
        E.ineligible,
        "IRS Pub 502 — General-health supplements (not eligible)",
    ),
    (
        ("menstrual", "tampon", "pad ", "pads"),
        E.eligible,
        "IRC §223(d)(2)(A) — Menstrual products eligible (CARES Act)",
    ),
    (
        ("otc", "ibuprofen", "acetaminophen", "aspirin", "loratadine", "cetirizine", "allergy"),
        E.eligible,
        "IRC §223(d)(2)(A) — OTC drugs eligible without Rx (CARES Act)",
    ),
]
