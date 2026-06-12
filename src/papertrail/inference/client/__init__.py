"""Client selection. `get_client()` picks mock vs live from `PAPERTRAIL_MOCK`.

from papertrail.inference.client import get_client
client = get_client()              # honors PAPERTRAIL_MOCK
expense = client.extract(images)
"""

from __future__ import annotations

import os

from .base import BasePaperTrailClient
from .sampling import SamplingParams

_TRUE = {"1", "true", "yes", "on"}


def get_client(*, think: bool = False) -> BasePaperTrailClient:
    if os.environ.get("PAPERTRAIL_MOCK", "").lower() in _TRUE:
        from .mock import MockPaperTrailClient

        return MockPaperTrailClient()
    from .llama import LlamaPaperTrailClient

    return LlamaPaperTrailClient(think=think)


__all__ = ["BasePaperTrailClient", "SamplingParams", "get_client"]
