"""The inference seam: one interface, two backends (live llama-server / mock).

Everything upstream (pipeline, app) depends only on this ABC and the
`ExtractedExpense` schema — never on which backend is running. Swap backends via
`PAPERTRAIL_MOCK` without touching a line of UI or pipeline code.
"""

from __future__ import annotations
import abc
import base64
from collections.abc import Generator
from papertrail.schema import EXTRACTION_INSTRUCTION, ExtractedExpense


def to_data_url(image: bytes, mime: str = "image/png") -> str:
    """Encode raw image bytes as a data URL for an OpenAI `image_url` block."""
    return f"data:{mime};base64,{base64.b64encode(image).decode('ascii')}"


class BasePaperTrailClient(abc.ABC):
    @abc.abstractmethod
    def extract(
        self, images: list[bytes], instruction: str = EXTRACTION_INSTRUCTION
    ) -> ExtractedExpense:
        """Extract structured fields from one document's page image(s)."""
        raise NotImplementedError

    def stream_extract(
        self, images: list[bytes], instruction: str = EXTRACTION_INSTRUCTION
    ) -> Generator[tuple[str, str], None, ExtractedExpense]:
        """Yield (kind, text) deltas; return the parsed expense (via StopIteration.value).
        Default = non-streaming: run extract() and emit the JSON as one content
        delta. Backends that support token streaming override this.
        """
        expense = self.extract(images, instruction)
        yield ("content", expense.model_dump_json(indent=2))
        return expense
