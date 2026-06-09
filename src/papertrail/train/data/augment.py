"""Albumentations degradation pipeline (PLAN.md 4.4).

Clean rendered documents do not generalise to real phone photographs. This
pipeline simulates the artifacts a real upload carries: angle, compression,
lighting, focus, perspective, and sensor noise. The same artifacts are what
`preprocess/` must survive at inference time — train/inference symmetry.

Each transform is wrapped defensively: Albumentations renames parameters across
minor versions, so we probe constructors and skip any that don't accept our
arguments rather than crash the whole run.
"""

from __future__ import annotations

import numpy as np
from PIL import Image

import albumentations as A


def _try(ctor, **kwargs):
    """Build a transform, dropping kwargs the installed version rejects."""
    try:
        return ctor(**kwargs)
    except TypeError:
        # Retry with only p= (every transform accepts it).
        try:
            return ctor(p=kwargs.get("p", 0.5))
        except Exception:
            return None


def build_pipeline(seed: int = 0) -> A.Compose:
    """Construct the degradation Compose. `seed` is applied per-call in `apply`."""
    transforms = [
        _try(A.Rotate, limit=12, border_mode=0, fill=255, p=0.7),
        _try(A.Perspective, scale=(0.02, 0.05), fill=255, p=0.4),
        _try(A.RandomBrightnessContrast, brightness_limit=0.2, contrast_limit=0.2, p=0.6),
        _try(A.GaussianBlur, blur_limit=(3, 5), p=0.35),
        _try(A.GaussNoise, std_range=(0.04, 0.12), p=0.4),
        _try(A.ImageCompression, quality_range=(35, 75), p=0.6),
    ]
    sp = _try(A.SaltAndPepper, amount=(0.005, 0.03), p=0.25)
    if sp is not None:
        transforms.append(sp)
    return A.Compose([t for t in transforms if t is not None])


def apply(img: Image.Image, pipeline: A.Compose) -> Image.Image:
    """Run the pipeline on a PIL image and return a PIL image."""
    arr = np.array(img.convert("RGB"))
    out = pipeline(image=arr)["image"]
    return Image.fromarray(out)
