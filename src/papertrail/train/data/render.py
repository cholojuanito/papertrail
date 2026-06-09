"""Pillow rendering helpers for synthetic documents.

We render training images directly with Pillow rather than going through a
PDF -> raster step: the model trains on images, and direct rendering removes a
system dependency (poppler) and gives pixel control over layout. Fonts are
resolved from common system paths with a graceful fallback.
"""

from __future__ import annotations

import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

# Candidate font files, in preference order. Resolved once and cached.
_MONO_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationMono-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
    "/usr/share/fonts/noto/NotoSansMono-Regular.ttf",
]
_SANS_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Regular.ttf",
    "/usr/share/fonts/TTF/DejaVuSans.ttf",
]
_SANS_BOLD_CANDIDATES = [
    "/usr/share/fonts/liberation/LiberationSans-Bold.ttf",
    "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
]
_HAND_CANDIDATES = [
    # No handwriting font is guaranteed; oblique sans + per-glyph jitter
    # approximates handwriting well enough once augmentation is applied.
    "/usr/share/fonts/TTF/DejaVuSans-Oblique.ttf",
    "/usr/share/fonts/liberation/LiberationSerif-Italic.ttf",
]

_INK = (20, 20, 20)
_PAPER = (252, 251, 248)


def _first_existing(paths: list[str]) -> str | None:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def font(kind: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """Load a font by kind ('mono'|'sans'|'sans_bold'|'hand') at `size`."""
    candidates = {
        "mono": _MONO_CANDIDATES,
        "sans": _SANS_CANDIDATES,
        "sans_bold": _SANS_BOLD_CANDIDATES,
        "hand": _HAND_CANDIDATES,
    }[kind]
    path = _first_existing(candidates)
    if path is not None:
        return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def blank(width: int, height: int) -> tuple[Image.Image, ImageDraw.ImageDraw]:
    img = Image.new("RGB", (width, height), _PAPER)
    return img, ImageDraw.Draw(img)


def text_width(draw: ImageDraw.ImageDraw, s: str, fnt) -> int:
    return int(draw.textlength(s, font=fnt))


def line(draw, x: int, y: int, s: str, fnt, fill=_INK) -> None:
    draw.text((x, y), s, font=fnt, fill=fill)


def right(draw, x_right: int, y: int, s: str, fnt, fill=_INK) -> None:
    """Right-align text so its right edge sits at `x_right`."""
    draw.text((x_right - text_width(draw, s, fnt), y), s, font=fnt, fill=fill)


def rule(draw, x0: int, x1: int, y: int, fill=(120, 120, 120), dashed: bool = False) -> None:
    if not dashed:
        draw.line([(x0, y), (x1, y)], fill=fill, width=1)
        return
    x = x0
    while x < x1:
        draw.line([(x, y), (min(x + 6, x1), y)], fill=fill, width=1)
        x += 12


def jitter_text(draw, x: int, y: int, s: str, fnt, rng: random.Random, fill=_INK) -> int:
    """Draw text glyph-by-glyph with vertical/spacing jitter (handwriting feel).

    Returns the x cursor after the string.
    """
    cx = x
    for ch in s:
        dy = rng.randint(-3, 3)
        draw.text((cx, y + dy), ch, font=fnt, fill=fill)
        cx += text_width(draw, ch, fnt) + rng.randint(-1, 2)
    return cx


def crop_partial(img: Image.Image, rng: random.Random) -> Image.Image:
    """Cut off the bottom portion of a document (a 'partial receipt')."""
    w, h = img.size
    keep = int(h * rng.uniform(0.45, 0.7))
    return img.crop((0, 0, w, keep))


def smudge(draw, x0: int, y0: int, x1: int, y1: int, rng: random.Random) -> None:
    """Obscure a region (an unreadable/torn amount) with a dark blob."""
    shade = rng.randint(30, 90)
    draw.rectangle([x0, y0, x1, y1], fill=(shade, shade, shade))
