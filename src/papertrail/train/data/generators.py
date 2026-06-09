"""Per-document-type synthetic generators.

Each `gen_*` takes a seeded `random.Random` and returns a `Doc`: a rendered
PIL image plus the ground-truth `ExtractedExpense` the model should emit. The
target reflects only what is legibly present in the image (hard negatives set
obscured fields to None) so the model learns to report, not hallucinate.
"""

from __future__ import annotations

import datetime as dt
import random
from dataclasses import dataclass

from PIL import Image, ImageDraw

from papertrail.schema.extraction import ExtractedExpense, LineItem, ProviderType

from . import catalog as C
from . import render as R


@dataclass
class Doc:
    image: Image.Image
    target: ExtractedExpense
    doc_type: str


def _usd(x: float) -> str:
    return f"${x:,.2f}"


def _recent_date(rng: random.Random) -> dt.date:
    start = dt.date(2025, 6, 1)
    return start + dt.timedelta(days=rng.randint(0, 365))


def _address(rng: random.Random) -> tuple[str, str]:
    street = rng.choice(C.STREETS)
    city, st, zp = rng.choice(C.CITIES)
    return street, f"{city}, {st} {zp}"


# --- shared receipt compositor -----------------------------------------------


def _compose_receipt(
    rng: random.Random,
    *,
    title: str,
    addr: tuple[str, str] | None,
    meta: list[tuple[str, str]],
    items: list[tuple[str, float | None]],
    totals: list[tuple[str, float]],
    footer: str | None,
    width: int = 480,
    font_kind: str = "mono",
    size: int = 15,
    handwritten: bool = False,
) -> tuple[Image.Image, dict[str, tuple[int, int, int, int]]]:
    """Render a receipt-style document on a tall canvas, then crop to content.

    Returns the image and a dict of named pixel boxes (e.g. {"total": box}) so
    callers can later obscure regions for hard negatives.
    """
    body = R.font(font_kind, size)
    head = R.font("sans_bold", size + 5)
    pad = 22
    lh = size + 8
    x0, x1 = pad, width - pad
    img, d = R.blank(width, 1500)
    boxes: dict[str, tuple[int, int, int, int]] = {}
    y = pad

    # Title (centered)
    tw = R.text_width(d, title, head)
    if handwritten:
        R.jitter_text(d, (width - tw) // 2, y, title, head, rng)
    else:
        d.text(((width - tw) // 2, y), title, font=head, fill=(20, 20, 20))
    y += lh + 8

    if addr is not None:
        for ln in addr:
            cw = R.text_width(d, ln, body)
            d.text(((width - cw) // 2, y), ln, font=body, fill=(70, 70, 70))
            y += lh
        y += 4

    R.rule(d, x0, x1, y, dashed=True)
    y += 10

    for label, val in meta:
        s = f"{label}: {val}"
        if handwritten:
            R.jitter_text(d, x0, y, s, body, rng)
        else:
            R.line(d, x0, y, s, body)
        y += lh
    if meta:
        y += 4
        R.rule(d, x0, x1, y, dashed=True)
        y += 10

    for desc, amt in items:
        if handwritten:
            R.jitter_text(d, x0, y, desc, body, rng)
        else:
            R.line(d, x0, y, desc[:40], body)
        if amt is not None:
            R.right(d, x1, y, _usd(amt), body)
        y += lh
    y += 6
    R.rule(d, x0, x1, y)
    y += 10

    for label, amt in totals:
        is_total = label.lower().startswith("total")
        f = R.font("sans_bold", size + 1) if is_total else body
        R.line(d, x0, y, label, f)
        R.right(d, x1, y, _usd(amt), f)
        if is_total:
            boxes["total"] = (x1 - 120, y - 2, x1 + 2, y + lh)
        y += lh

    if footer:
        y += 10
        cw = R.text_width(d, footer, body)
        d.text(((width - cw) // 2, y), footer, font=body, fill=(90, 90, 90))
        y += lh

    y += pad
    return img.crop((0, 0, width, min(y, 1500))), boxes


# --- generators ---------------------------------------------------------------


def gen_pharmacy(rng: random.Random) -> Doc:
    merchant = rng.choice(C.PHARMACIES)
    street, citystate = _address(rng)
    date = _recent_date(rng)
    patient = rng.choice(C.PATIENTS)

    items: list[tuple[str, float | None]] = []
    line_items: list[LineItem] = []
    n_rx = rng.randint(1, 2)
    n_otc = rng.randint(0, 2)
    for name, qty in rng.sample(C.RX_DRUGS, n_rx):
        amt = round(rng.uniform(8, 60), 2)
        items.append((f"{name} ({qty})", amt))
        line_items.append(LineItem(description=name, amount=amt))
    for name, qty in rng.sample(C.OTC_DRUGS, min(n_otc, len(C.OTC_DRUGS))):
        amt = round(rng.uniform(5, 22), 2)
        items.append((f"{name} ({qty})", amt))
        line_items.append(LineItem(description=name, amount=amt))
    if rng.random() < 0.35:  # non-medical distractor
        name, amt = rng.choice(C.NON_MEDICAL_ITEMS)
        items.append((name, amt))
        line_items.append(LineItem(description=name, amount=amt))

    total = round(sum(li.amount for li in line_items), 2)
    img, _ = _compose_receipt(
        rng,
        title=merchant,
        addr=(street, citystate, "Tel: (555) 010-2837"),
        meta=[
            ("Patient", patient),
            ("Rx Date", date.isoformat()),
            ("Rx #", str(rng.randint(1000000, 9999999))),
        ],
        items=items,
        totals=[("Subtotal", total), ("TOTAL", total)],
        footer="Thank you — Insurance may reimburse eligible items",
    )
    target = ExtractedExpense(
        merchant=merchant,
        date=date.isoformat(),
        amount=total,
        provider_type=ProviderType.pharmacy,
        line_items=line_items,
        patient=patient,
    )
    return Doc(img, target, "pharmacy")


def gen_clinic(rng: random.Random) -> Doc:
    """Doctor visit / dental / vision receipt (provider_type varies)."""
    domain = rng.choice(["medical", "dental", "vision"])
    if domain == "dental":
        merchant = rng.choice(C.DENTAL_OFFICES)
        pool, ptype, title = C.DENTAL_PROCEDURES, ProviderType.dental, "DENTAL STATEMENT"
    elif domain == "vision":
        merchant = rng.choice(C.VISION_OFFICES)
        pool, ptype, title = C.VISION_ITEMS, ProviderType.vision, "VISION CARE RECEIPT"
    else:
        merchant = rng.choice(C.CLINICS)
        pool, ptype, title = C.MEDICAL_SERVICES, ProviderType.medical, "PATIENT RECEIPT"

    street, citystate = _address(rng)
    date = _recent_date(rng)
    patient = rng.choice(C.PATIENTS)
    chosen = rng.sample(pool, rng.randint(1, 2))
    line_items = [LineItem(description=desc, amount=amt) for desc, amt in chosen]
    items: list[tuple[str, float | None]] = [(d, a) for d, a in chosen]
    total = round(sum(a for _, a in chosen), 2)

    img, _ = _compose_receipt(
        rng,
        title=merchant,
        addr=(street, citystate),
        meta=[
            ("Patient", patient),
            ("Date of Service", date.isoformat()),
            ("Account", str(rng.randint(10000, 99999))),
        ],
        items=items,
        totals=[("Amount Due", total), ("TOTAL PAID", total)],
        footer=title,
    )
    target = ExtractedExpense(
        merchant=merchant,
        date=date.isoformat(),
        amount=total,
        provider_type=ptype,
        line_items=line_items,
        patient=patient,
    )
    return Doc(img, target, "clinic")


def gen_eob(rng: random.Random) -> Doc:
    insurer = rng.choice(C.INSURERS)
    is_dental = rng.random() < 0.3
    provider = rng.choice(C.DENTAL_OFFICES if is_dental else C.CLINICS)
    pool = C.DENTAL_PROCEDURES if is_dental else C.MEDICAL_SERVICES
    ptype = ProviderType.dental if is_dental else ProviderType.medical
    date = _recent_date(rng)
    patient = rng.choice(C.PATIENTS)
    width = 760
    size = 16

    services = rng.sample(pool, rng.randint(1, 3))
    rows = []
    line_items: list[LineItem] = []
    patient_total = 0.0
    for desc, billed in services:
        plan_paid = round(billed * rng.uniform(0.5, 0.85), 2)
        responsibility = round(billed - plan_paid, 2)
        patient_total += responsibility
        rows.append((desc, billed, plan_paid, responsibility))
        line_items.append(LineItem(description=desc, amount=responsibility))
    patient_total = round(patient_total, 2)

    body = R.font("sans", size)
    bold = R.font("sans_bold", size)
    head = R.font("sans_bold", size + 8)
    img, d = R.blank(width, 1100)
    pad = 30
    x0, x1 = pad, width - pad
    y = pad
    d.text((x0, y), insurer, font=head, fill=(20, 20, 60))
    y += size + 16
    d.text((x0, y), "EXPLANATION OF BENEFITS", font=bold, fill=(20, 20, 20))
    R.right(d, x1, y, "THIS IS NOT A BILL", body, fill=(150, 30, 30))
    y += size + 6
    R.rule(d, x0, x1, y)
    y += 14
    for label, val in [
        ("Member", patient),
        ("Provider", provider),
        ("Claim #", str(rng.randint(10**9, 10**10 - 1))),
        ("Service Date", date.isoformat()),
    ]:
        d.text((x0, y), f"{label}:", font=bold, fill=(40, 40, 40))
        d.text((x0 + 150, y), val, font=body, fill=(20, 20, 20))
        y += size + 8
    y += 6

    # table header
    cols = [x0, x0 + 320, x0 + 460, x1]
    d.text((cols[0], y), "Service", font=bold, fill=(20, 20, 20))
    R.right(d, cols[1] + 80, y, "Billed", bold)
    R.right(d, cols[2] + 80, y, "Plan Paid", bold)
    R.right(d, cols[3], y, "Your Resp.", bold)
    y += size + 6
    R.rule(d, x0, x1, y)
    y += 10
    for desc, billed, plan_paid, resp in rows:
        d.text((cols[0], y), desc[:40], font=body, fill=(20, 20, 20))
        R.right(d, cols[1] + 80, y, _usd(billed), body)
        R.right(d, cols[2] + 80, y, _usd(plan_paid), body)
        R.right(d, cols[3], y, _usd(resp), body)
        y += size + 10
    R.rule(d, x0, x1, y)
    y += 12
    d.text((cols[0], y), "PATIENT RESPONSIBILITY", font=bold, fill=(20, 20, 20))
    R.right(d, x1, y, _usd(patient_total), bold)
    y += size + pad

    img = img.crop((0, 0, width, min(y, 1100)))
    target = ExtractedExpense(
        merchant=provider,
        date=date.isoformat(),
        amount=patient_total,
        provider_type=ptype,
        line_items=line_items,
        patient=patient,
    )
    return Doc(img, target, "eob")


def gen_handwritten(rng: random.Random) -> Doc:
    merchant = rng.choice(C.CLINICS + C.DENTAL_OFFICES)
    date = _recent_date(rng)
    patient = rng.choice(C.PATIENTS)
    service, amt = rng.choice(C.MEDICAL_SERVICES + C.DENTAL_PROCEDURES)
    amt = round(amt + rng.uniform(-10, 10), 2)
    line_items = [LineItem(description=service.split("(")[0].strip(), amount=amt)]
    img, _ = _compose_receipt(
        rng,
        title="Receipt",
        addr=(merchant,),
        meta=[("Patient", patient), ("Date", date.strftime("%m/%d/%Y"))],
        items=[(service.split("(")[0].strip(), amt)],
        totals=[("Total", amt)],
        footer="Paid - Thank you",
        width=560,
        font_kind="hand",
        size=22,
        handwritten=True,
    )
    target = ExtractedExpense(
        merchant=merchant,
        date=date.isoformat(),
        amount=amt,
        provider_type=ProviderType.other,
        line_items=line_items,
        patient=patient,
    )
    return Doc(img, target, "handwritten")


def gen_ineligible(rng: random.Random) -> Doc:
    merchant = rng.choice(C.INELIGIBLE_MERCHANTS)
    street, citystate = _address(rng)
    date = _recent_date(rng)
    chosen = rng.sample(C.INELIGIBLE_ITEMS, rng.randint(1, 2))
    line_items = [LineItem(description=d, amount=a) for d, a in chosen]
    total = round(sum(a for _, a in chosen), 2)
    img, _ = _compose_receipt(
        rng,
        title=merchant,
        addr=(street, citystate),
        meta=[("Date", date.isoformat()), ("Order #", str(rng.randint(100000, 999999)))],
        items=[(d, a) for d, a in chosen],
        totals=[("Subtotal", total), ("TOTAL", total)],
        footer="Thank you for your purchase",
    )
    target = ExtractedExpense(
        merchant=merchant,
        date=date.isoformat(),
        amount=total,
        provider_type=ProviderType.other,
        line_items=line_items,
    )
    return Doc(img, target, "ineligible")


def gen_hard_negative(rng: random.Random) -> Doc:
    """A degraded/incomplete document; obscured fields become null in the target."""
    base = rng.choice([gen_pharmacy, gen_clinic])(rng)
    mode = rng.choice(["obscure_total", "no_date", "partial_crop"])
    t = base.target

    if mode == "obscure_total":
        img, boxes = _compose_receipt(
            rng,
            title=t.merchant or "Receipt",
            addr=None,
            meta=[("Patient", t.patient or ""), ("Date", t.date or "")],
            items=[(li.description, li.amount) for li in t.line_items],
            totals=[("TOTAL", t.amount or 0.0)],
            footer=None,
        )
        if "total" in boxes:
            R.smudge(ImageDraw.Draw(img), *boxes["total"], rng)
        target = ExtractedExpense(
            merchant=t.merchant,
            date=t.date,
            amount=None,
            provider_type=t.provider_type,
            line_items=t.line_items,
            patient=t.patient,
        )
        return Doc(img, target, "hard_negative")

    if mode == "no_date":
        img, _ = _compose_receipt(
            rng,
            title=t.merchant or "Receipt",
            addr=None,
            meta=[("Patient", t.patient or "")],
            items=[(li.description, li.amount) for li in t.line_items],
            totals=[("TOTAL", t.amount or 0.0)],
            footer=None,
        )
        target = ExtractedExpense(
            merchant=t.merchant,
            date=None,
            amount=t.amount,
            provider_type=t.provider_type,
            line_items=t.line_items,
            patient=t.patient,
        )
        return Doc(img, target, "hard_negative")

    # partial_crop: only the header survives
    img = R.crop_partial(base.image, rng)
    target = ExtractedExpense(
        merchant=t.merchant,
        date=t.date,
        amount=None,
        provider_type=t.provider_type,
        line_items=[],
        patient=t.patient,
    )
    return Doc(img, target, "hard_negative")


# Maps the PLAN.md category names to generators and default counts (sum = 100).
GENERATORS = {
    "pharmacy": (gen_pharmacy, 25),
    "eob": (gen_eob, 25),
    "clinic": (gen_clinic, 15),
    "handwritten": (gen_handwritten, 10),
    "ineligible": (gen_ineligible, 10),
    "hard_negative": (gen_hard_negative, 15),
}
