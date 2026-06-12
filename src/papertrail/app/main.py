"""Gradio UI — wiring only. Streams pipeline events for live progress + reasoning.

No rules, no SQL, no prompt parsing live here (see AGENTS.md "UI is dumb"). The
handler is a generator: it consumes `process_document_stream` events and yields
incremental UI updates. Run against the mock with `PAPERTRAIL_MOCK=true`.
"""

from __future__ import annotations

import gradio as gr

from papertrail.events import Done, Failed, Stage, Token
from papertrail.inference.client import get_client
from papertrail.pipeline import process_document_stream


def _run_stream(file_path: str | None, show_thoughts: bool):
    if not file_path:
        yield "Upload a receipt or EOB first.", "", "", "", "", {}
        return

    client = get_client(think=bool(show_thoughts))
    status, thoughts, answer = "⏳ starting…", "", ""
    summary = eligibility = readiness = ""
    raw: dict = {}
    yield status, thoughts, summary, eligibility, readiness, raw

    for ev in process_document_stream(file_path, client=client):
        if isinstance(ev, Stage):
            status = f"⏳ **{ev.name}** — {ev.message}"
        elif isinstance(ev, Token):
            if ev.kind == "reasoning":
                thoughts += ev.text
            else:
                answer += ev.text
        elif isinstance(ev, Failed):
            yield f"❌ {ev.error}", thoughts, summary, eligibility, readiness, raw
            return
        elif isinstance(ev, Done):
            e, v, r = ev.result.expense, ev.result.eligibility, ev.result.readiness
            amount = f"${e.amount:,.2f}" if e.amount is not None else "—"
            summary = (
                f"{e.merchant or '—'}  •  {e.date or '—'}  •  {amount}  •  {e.provider_type.value}"
            )
            eligibility = f"{v.status.value.upper()} — {v.citation}"
            readiness = f"{r.score}/100" + (
                f"   (missing: {', '.join(r.missing)})" if r.missing else ""
            )
            raw = e.model_dump()
            status = "✅ done"
        shown_thoughts = (
            thoughts or answer or "(enable 'stream reasoning' to watch the model think)"
        )
        yield status, shown_thoughts, summary, eligibility, readiness, raw


def build_demo() -> gr.Blocks:
    with gr.Blocks(title="PaperTrail") as demo:
        gr.Markdown("# PaperTrail\nMedical-expense extraction · local-first")
        with gr.Row():
            file_in = gr.File(label="Receipt / EOB (image or PDF)", type="filepath")
            with gr.Column():
                show_thoughts = gr.Checkbox(
                    label="Stream model reasoning (slower; for debugging)", value=False
                )
                run_btn = gr.Button("Extract", variant="primary")
        status = gr.Markdown("Idle.")
        with gr.Accordion("Model thoughts / live output", open=True):
            thoughts = gr.Textbox(label="", lines=10, interactive=False, autoscroll=True)
        summary = gr.Textbox(label="Extracted", interactive=False)
        eligibility = gr.Textbox(label="HSA eligibility", interactive=False)
        readiness = gr.Textbox(label="Audit readiness", interactive=False)
        raw = gr.JSON(label="Structured JSON")
        run_btn.click(
            _run_stream,
            inputs=[file_in, show_thoughts],
            outputs=[status, thoughts, summary, eligibility, readiness, raw],
        )
    return demo


def main():
    demo = build_demo()
    demo.launch()

if __name__ == "__main__":
    main()
