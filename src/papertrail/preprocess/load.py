"""Turn an uploaded file into model-ready page images (PNG bytes).

The inference client wants raw image bytes; this normalizes the two input kinds:
- PDF  -> one PNG per page (via pdf2image / poppler)
- image -> a single PNG (re-encoded so the client always sees PNG)

Images are downscaled to a max long edge before encoding. This is the cheapest,
restart-free way to control vision token usage: llama.cpp tiles the image by
resolution (~256 tokens per 512px tile), so a 4000px phone photo costs thousands
of tokens while a ~1536px page is still legible at a fraction of that. Tune via
`PAPERTRAIL_MAX_IMAGE_EDGE` (set 0 to disable).

Deskew/contrast normalization can layer in here later without changing the
signature — `load_document(path) -> list[bytes]` is the stable contract.
"""

from __future__ import annotations

import os
from io import BytesIO
from pathlib import Path

from PIL import Image

# 0 disables downscaling. ~1536 keeps receipt/EOB text legible (~3 tiles/edge).
DEFAULT_MAX_EDGE = int(os.environ.get("PAPERTRAIL_MAX_IMAGE_EDGE", "1536"))


def _png_bytes(img: Image.Image, max_edge: int) -> bytes:
    img = img.convert("RGB")
    longest = max(img.size)
    if max_edge and longest > max_edge:
        scale = max_edge / longest
        img = img.resize((round(img.width * scale), round(img.height * scale)), Image.Resampling.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def load_document(
    path: str | Path, *, pdf_dpi: int = 200, max_edge: int = DEFAULT_MAX_EDGE
) -> list[bytes]:
    path = Path(path)
    if path.suffix.lower() == ".pdf":
        from pdf2image import convert_from_path

        return [_png_bytes(page, max_edge) for page in convert_from_path(str(path), dpi=pdf_dpi)]
    return [_png_bytes(Image.open(path), max_edge)]
