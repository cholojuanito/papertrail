"""The integration contract: the shape the model extracts and the app consumes.

This module is intentionally dependency-free (only Pydantic + stdlib). Everything
else in the project — the synthetic data generator, the Modal training job, the
inference client, and the Gradio UI — agrees on these shapes.

Design note: `confidence` is deliberately NOT part of the *training target*. We
cannot fabricate calibrated confidence from synthetic labels (the model would
just learn to always emit 0.97). Confidence is populated at inference time from
the server's token logprobs, so it lives as an optional field here but is dropped
by `target_json()`.
"""

from __future__ import annotations

import json
from enum import Enum

from pydantic import BaseModel, Field


class ProviderType(str, Enum):
    pharmacy = "pharmacy"
    dental = "dental"
    vision = "vision"
    medical = "medical"
    other = "other"


class LineItem(BaseModel):
    description: str
    amount: float | None = None


class ExtractedExpense(BaseModel):
    """One expense document's extracted facts.

    Used as both the fine-tuning target (via `target_json`) and the inference
    output the UI renders. Fields are nullable where a real document can legibly
    omit them (obscured totals, missing dates) — teaching the model to report
    what it can read rather than hallucinate.
    """

    merchant: str | None = None
    date: str | None = None  # ISO 8601 YYYY-MM-DD
    amount: float | None = None  # document total
    currency: str = "USD"
    provider_type: ProviderType = ProviderType.other
    line_items: list[LineItem] = Field(default_factory=list)
    patient: str | None = None
    # Inference-only; never set on a training target.
    confidence: dict[str, float] | None = None

    def target_json(self) -> str:
        """Canonical JSON string the model is trained to emit.

        Stable key order, drops `confidence` and `None` scalars so the label is
        deterministic across runs (important for reproducible eval).
        """
        payload: dict = {}
        if self.merchant is not None:
            payload["merchant"] = self.merchant
        if self.date is not None:
            payload["date"] = self.date
        if self.amount is not None:
            payload["amount"] = round(self.amount, 2)
        payload["currency"] = self.currency
        payload["provider_type"] = self.provider_type.value
        payload["line_items"] = [
            {
                "description": li.description,
                **({"amount": round(li.amount, 2)} if li.amount is not None else {}),
            }
            for li in self.line_items
        ]
        if self.patient is not None:
            payload["patient"] = self.patient
        return json.dumps(payload, ensure_ascii=False, indent=2)


# The fixed instruction paired with every document. Embeds the target schema so
# the model knows the exact output contract. Keep this byte-stable: it is part of
# the training data AND the inference prompt, and changing it invalidates a tune.
EXTRACTION_INSTRUCTION = (
    "You are an expense-document extractor. Read the attached medical expense "
    "document (receipt, invoice, or insurance EOB) and return ONLY a JSON object "
    "with these fields:\n"
    '  "merchant": string or null — the business/provider name\n'
    '  "date": string or null — service or purchase date as YYYY-MM-DD\n'
    '  "amount": number or null — the document total the patient paid\n'
    '  "currency": string — ISO currency code, default "USD"\n'
    '  "provider_type": one of "pharmacy" | "dental" | "vision" | "medical" | "other"\n'
    '  "line_items": array of {"description": string, "amount": number or null}\n'
    '  "patient": string or null — patient name if present\n'
    "Use null for any field you cannot read. Do not invent values. Output JSON only."
)
