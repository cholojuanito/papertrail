"""Live backend: talks to llama-server over its OpenAI-compatible API.

Uses the standard `openai` client (lazy-imported so the mock path needs no deps).
Sends `image_url` data-URL blocks, strips the reasoning block, and validates the
JSON into `ExtractedExpense`.
"""

from __future__ import annotations

import os
import re

from collections.abc import Generator
from papertrail.schema import EXTRACTION_INSTRUCTION, ExtractedExpense
from .base import BasePaperTrailClient, to_data_url
from .sampling import SamplingParams

_THINK_REGEX = re.compile(r"<think>.*?</think>", re.DOTALL)


def _parse(
    raw: str, *, finish_reason: str | None = None, reasoning: str | None = None
) -> ExtractedExpense:
    """Strip reasoning, slice out the JSON object, validate to the schema."""
    for candidate in (raw, reasoning):  # answer first, then salvage from a think block
        if not candidate:
            continue
        text = _THINK_REGEX.sub("", candidate)
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            return ExtractedExpense.model_validate_json(text[start : end + 1])
    hint = (
        " — output hit the context limit before finishing (model ran away thinking; "
        "keep reasoning off, or raise --ctx-size)"
        if finish_reason == "length"
        else ""
    )
    raise ValueError(
        f"no JSON in model output (finish_reason={finish_reason}){hint}: {raw[:200]!r}"
    )


def _reasoning_text(obj) -> str | None:
    """Read `reasoning_content` whether openai exposed it as a field or model_extra."""
    value = getattr(obj, "reasoning_content", None)
    if value is None:
        extra = getattr(obj, "model_extra", None)
        if extra:
            value = extra.get("reasoning_content")
    return value


class LlamaPaperTrailClient(BasePaperTrailClient):
    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        sampling: SamplingParams | None = None,
        think: bool = False,  # extraction = instruct mode; reasoning runs away + breaks JSON
    ) -> None:
        url = (base_url or os.environ.get("LLAMA_BASE_URL", "http://localhost:8080")).rstrip("/")
        self.base_url = url if url.endswith("/v1") else url + "/v1"
        self.model = model or os.environ.get("PAPERTRAIL_MODEL_ALIAS", "nemotron")
        self.api_key = api_key or os.environ.get("LLAMA_API_KEY", "no-key-required")
        self.sampling = sampling or SamplingParams.from_env()
        self.think = think
        self._client = None

    @property
    def client(self):
        if self._client is None:
            from openai import OpenAI

            self._client = OpenAI(base_url=self.base_url, api_key=self.api_key)
        return self._client

    def _payload(
        self, images: list[bytes], instruction: str, sampling: SamplingParams | None
    ) -> tuple[list[dict], dict, dict]:
        content: list[dict] = [{"type": "text", "text": instruction}]
        for img in images:
            content.append({"type": "image_url", "image_url": {"url": to_data_url(img)}})
        native, extra = (sampling or self.sampling).to_request()
        if not self.think:
            # Disable the reasoning block regardless of how the server was launched.
            extra["chat_template_kwargs"] = {"enable_thinking": False}
        return [{"role": "user", "content": content}], native, extra

    def extract(
        self,
        images: list[bytes],
        instruction: str = EXTRACTION_INSTRUCTION,
        sampling: SamplingParams | None = None,
    ) -> ExtractedExpense:
        messages, native, extra = self._payload(images, instruction, sampling)
        resp = self.client.chat.completions.create(
            model=self.model, messages=messages, extra_body=extra or None, **native
        )
        choice = resp.choices[0]
        return _parse(
            choice.message.content or "",
            finish_reason=choice.finish_reason,
            reasoning=_reasoning_text(choice.message),
        )

    def stream_extract(
        self,
        images: list[bytes],
        instruction: str = EXTRACTION_INSTRUCTION,
        sampling: SamplingParams | None = None,
    ) -> Generator[tuple[str, str], None, ExtractedExpense]:
        messages, native, extra = self._payload(images, instruction, sampling)
        stream = self.client.chat.completions.create(
            model=self.model, messages=messages, stream=True, extra_body=extra or None, **native
        )
        answer: list[str] = []
        reasoning: list[str] = []
        finish: str | None = None
        for chunk in stream:
            if not chunk.choices:
                continue
            choice = chunk.choices[0]
            delta = choice.delta
            think = _reasoning_text(delta)
            if think:
                reasoning.append(think)
                yield ("reasoning", think)
            if getattr(delta, "content", None):
                answer.append(delta.content)
                yield ("content", delta.content)
            if choice.finish_reason:
                finish = choice.finish_reason
        return _parse("".join(answer), finish_reason=finish, reasoning="".join(reasoning))
