"""Per-request sampling knobs for the OpenAI-compatible chat call.

This is the *request-time* surface (distinct from `server/args.py`, which sets the
server's launch-time defaults). The OpenAI client accepts some of these as named
kwargs; the llama.cpp-specific ones (top_k, min_p, repeat_penalty, ...) must ride
in `extra_body`. `to_request()` does that split so the client stays dumb.

Defaults are Nemotron instruct-mode (temp 0.2, top_k 1). Override per-instance,
per-call, or from env (`PAPERTRAIL_TEMPERATURE`, `PAPERTRAIL_TOP_K`, ...).
"""

from __future__ import annotations

import dataclasses
import os
from dataclasses import dataclass
from typing import ClassVar


@dataclass
class SamplingParams:
    temperature: float | None = 0.2
    top_p: float | None = None
    top_k: int | None = 1
    min_p: float | None = None
    max_tokens: int | None = 2048  # safety cap; extraction JSON is tiny, never needs more
    repeat_penalty: float | None = None
    presence_penalty: float | None = None
    frequency_penalty: float | None = None
    seed: int | None = None
    stop: list[str] | None = None

    # Names the OpenAI client accepts directly; everything else -> extra_body.
    _NATIVE: ClassVar[frozenset[str]] = frozenset(
        {
            "temperature",
            "top_p",
            "max_tokens",
            "presence_penalty",
            "frequency_penalty",
            "seed",
            "stop",
        }
    )
    # env var -> (field, caster)
    _ENV: ClassVar[dict[str, tuple[str, type]]] = {
        "PAPERTRAIL_TEMPERATURE": ("temperature", float),
        "PAPERTRAIL_TOP_P": ("top_p", float),
        "PAPERTRAIL_TOP_K": ("top_k", int),
        "PAPERTRAIL_MIN_P": ("min_p", float),
        "PAPERTRAIL_MAX_TOKENS": ("max_tokens", int),
        "PAPERTRAIL_REPEAT_PENALTY": ("repeat_penalty", float),
        "PAPERTRAIL_SEED": ("seed", int),
    }

    @classmethod
    def from_env(cls) -> SamplingParams:
        """Build from PAPERTRAIL_* env, falling back to the dataclass defaults."""
        overrides: dict[str, object] = {}
        for env, (field, cast) in cls._ENV.items():
            raw = os.environ.get(env)
            if raw is not None and raw != "":
                overrides[field] = cast(raw)
        return cls(**overrides)

    def merged(self, **overrides: object) -> SamplingParams:
        """Return a copy with non-None overrides applied (for per-call tweaks)."""
        clean = {k: v for k, v in overrides.items() if v is not None}
        return dataclasses.replace(self, **clean)

    def to_request(self) -> tuple[dict, dict]:
        """Split into (native openai kwargs, extra_body) — dropping unset (None)."""
        native: dict = {}
        extra: dict = {}
        for f in dataclasses.fields(self):
            value = getattr(self, f.name)
            if value is None:
                continue
            (native if f.name in self._NATIVE else extra)[f.name] = value
        return native, extra
