"""Event vocabulary for the streaming pipeline.

`process_document_stream` yields these so the UI can show progress and live model
output instead of blocking on one long call:

    Stage  — pipeline progress ("preprocess: 2 pages", "inference: calling model")
    Token  — an incremental model-output delta (kind="reasoning" | "content")
    Done   — terminal success, carries the ProcessedExpense
    Failed — terminal error, carries a message
"""

from __future__ import annotations

from dataclasses import dataclass

from papertrail.schema import ProcessedExpense


@dataclass
class Stage:
    name: str  # preprocess | inference | eligibility | scoring | storage
    message: str


@dataclass
class Token:
    kind: str  # "reasoning" | "content"
    text: str


@dataclass
class Done:
    result: ProcessedExpense


@dataclass
class Failed:
    error: str


Event = Stage | Token | Done | Failed
