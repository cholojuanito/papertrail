"""The orchestrator — the one place the layers meet.

    file ──> preprocess.load_document ──> images (PNG bytes)
         ──> inference.client.extract  ──> ExtractedExpense        (model, probabilistic)
         ──> core.eligibility.classify ──> EligibilityVerdict      (rules, deterministic)
         ──> core.scoring.score        ──> ReadinessScore          (rules, deterministic)
         ──> storage.Repository        ──> persisted rows          (optional)
         ──> ProcessedExpense                                       (returned to the UI)

Dependency direction stays legal: this module imports inward (app/inference/
storage/preprocess -> core -> schema). `core` and `schema` never import this.
The UI calls `process_document` and renders the result — it owns no rules.
"""

from __future__ import annotations

from pathlib import Path

from collections.abc import Iterator
from papertrail.core.eligibility import classify
from papertrail.core.scoring import score
from papertrail.events import Done, Event, Failed, Stage, Token
from papertrail.inference.client import BasePaperTrailClient, get_client
from papertrail.preprocess import load_document
from papertrail.schema import EXTRACTION_INSTRUCTION, ProcessedExpense
from papertrail.storage import Repository


def process_document(
    path: str | Path,
    *,
    client: BasePaperTrailClient | None = None,
    repo: Repository | None = None,
    instruction: str = EXTRACTION_INSTRUCTION,
    doc_type: str = "upload",
) -> ProcessedExpense:
    """Run one document end-to-end. Persists if `repo` is given."""
    client = client or get_client()

    images = load_document(path)
    expense = client.extract(images, instruction)
    verdict = classify(expense)
    readiness = score(expense)

    result = ProcessedExpense(expense=expense, eligibility=verdict, readiness=readiness)

    if repo is not None:
        doc_id = repo.add_document(
            Path(path).name, path, doc_type, raw_model_output=expense.model_dump_json()
        )
        result.document_id = doc_id
        result.expense_id = repo.add_expense(doc_id, expense, verdict, readiness)

    return result


def process_document_stream(
    path: str | Path,
    *,
    client: BasePaperTrailClient | None = None,
    repo: Repository | None = None,
    instruction: str = EXTRACTION_INSTRUCTION,
    doc_type: str = "upload",
) -> Iterator[Event]:
    """Same flow as `process_document`, but yields Stage/Token/Done/Failed events.
    Lets the UI show live progress + streamed model reasoning. Errors are emitted
    as a terminal `Failed` event rather than raised, so the stream always closes
    cleanly for the consumer.
    """
    client = client or get_client()
    try:
        yield Stage("preprocess", "loading document…")
        images = load_document(path)
        yield Stage("preprocess", f"{len(images)} page image(s) ready")
        yield Stage("inference", "calling model…")
        # Forward token deltas as events; capture the parsed expense (StopIteration.value).
        gen = client.stream_extract(images, instruction)
        expense: ProcessedExpense | None = None
        while True:
            try:
                kind, text = next(gen)
            except StopIteration as stop:
                expense = stop.value
                break
            yield Token(kind, text)
        yield Stage("inference", "extraction complete")
        yield Stage("eligibility", "classifying line items…")
        verdict = classify(expense)
        yield Stage("scoring", "scoring audit readiness…")
        readiness = score(expense)
        result = ProcessedExpense(expense=expense, eligibility=verdict, readiness=readiness)
        if repo is not None:
            yield Stage("storage", "saving record…")
            doc_id = repo.add_document(
                Path(path).name, path, doc_type, raw_model_output=expense.model_dump_json()
            )
            result.document_id = doc_id
            result.expense_id = repo.add_expense(doc_id, expense, verdict, readiness)
        yield Done(result)
    except Exception as exc:  # surface as a terminal event, don't crash the stream
        yield Failed(f"{type(exc).__name__}: {exc}")
