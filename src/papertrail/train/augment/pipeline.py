"""Albumentations degradation pipelines, profiled by document type (PLAN.md 4.4).

Real-world reality drives two profiles:

* **document** (EOBs, clinic statements, general PDFs): people upload clean
  scans/PDFs, so only mild artifacts apply — light compression, small skew,
  gentle lighting shifts.
* **receipt** (pharmacy, handwritten, retail/ineligible, partial): a genuine
  toss-up. Some are crisp phone shots; many are crumpled, angled, blurry thermal
  paper. We model that bimodally — a fraction pass through pristine, the rest get
  light or heavy degradation.

The same artifacts are what `preprocess/` must survive at inference — train and
inference share the distribution.

Each transform is built defensively: Albumentations renames parameters across
minor versions, so we probe constructors and skip any that reject our kwargs.
"""

from __future__ import annotations

import random

import numpy as np
from PIL import Image

import albumentations as A

# doc_type -> profile. Anything unlisted defaults to "receipt" (the cautious,
# heavier path).
DOC_PROFILE: dict[str, str] = {
    "eob": "document",
    "clinic": "document",
    "pharmacy": "receipt",
    "handwritten": "receipt",
    "ineligible": "receipt",
    "hard_negative": "receipt",
}


def _try(ctor, **kwargs):
    """Build a transform, dropping kwargs the installed version rejects."""
    try:
        return ctor(**kwargs)
    except TypeError:
        try:
            return ctor(p=kwargs.get("p", 0.5))
        except Exception:
            return None


def _compose(transforms) -> A.Compose:
    return A.Compose([t for t in transforms if t is not None])


def _light() -> A.Compose:
    """Clean-document degradation: scanner/PDF realism, nothing destructive."""
    return _compose(
        [
            _try(A.Rotate, limit=3, border_mode=0, fill=255, p=0.5),
            _try(A.RandomBrightnessContrast, brightness_limit=0.12, contrast_limit=0.12, p=0.5),
            _try(A.GaussNoise, std_range=(0.01, 0.04), p=0.2),
            _try(A.ImageCompression, quality_range=(72, 95), p=0.5),
        ]
    )


def _heavy() -> A.Compose:
    """Phone-photo-of-a-crumpled-receipt degradation."""
    return _compose(
        [
            _try(A.Rotate, limit=12, border_mode=0, fill=255, p=0.8),
            _try(A.Perspective, scale=(0.02, 0.06), fill=255, p=0.5),
            _try(A.RandomBrightnessContrast, brightness_limit=0.25, contrast_limit=0.25, p=0.7),
            _try(A.GaussianBlur, blur_limit=(3, 5), p=0.4),
            _try(A.GaussNoise, std_range=(0.04, 0.13), p=0.5),
            _try(A.ImageCompression, quality_range=(30, 70), p=0.7),
            _try(A.SaltAndPepper, amount=(0.005, 0.03), p=0.3),
        ]
    )


def make_pipelines() -> dict[str, A.Compose]:
    """Build the two reusable Composes once; reuse across the whole dataset."""
    return {"light": _light(), "heavy": _heavy()}


def _apply(pipeline: A.Compose, img: Image.Image) -> Image.Image:
    arr = np.array(img.convert("RGB"))
    return Image.fromarray(pipeline(image=arr)["image"])


def degrade(
    img: Image.Image,
    doc_type: str,
    pipelines: dict[str, A.Compose],
    rng: random.Random,
) -> Image.Image:
    """Degrade `img` according to its document-type profile.

    - document  -> always light.
    - receipt   -> toss-up: ~25% pristine, ~25% light, ~50% heavy.
    """
    profile = DOC_PROFILE.get(doc_type, "receipt")
    if profile == "document":
        return _apply(pipelines["light"], img)

    roll = rng.random()
    if roll < 0.25:
        return img  # pristine — a good phone shot
    pipeline = pipelines["light"] if roll < 0.5 else pipelines["heavy"]
    return _apply(pipeline, img)
