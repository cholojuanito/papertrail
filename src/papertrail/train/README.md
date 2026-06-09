# train — fine-tuning pipeline

**Responsibility:** generate data, train the LoRA on Modal, measure improvement. Runs offline, not at app runtime.

## `data/` — synthetic document generation
- ReportLab/Pillow generators per PLAN.md §4.3 (pharmacy, EOB, doctor visit, handwritten, ineligible edge cases, hard negatives). Each emits `(image, instruction, JSON)` triples.

## `modal/` — the training job
- `job.py` — Modal A100-80GB entrypoint. Unsloth vision LoRA (primary) with the text-LoRA fallback path. Merge adapter → convert to GGUF + mmproj → push to HF Hub.

## `eval/` — before/after metrics
- Per-field exact-match accuracy on `amount/date/merchant` over the held-out test split. Baseline (un-tuned) recorded first so the improvement is citable.

**Why it exists / talking point:** this is the "Well-Tuned" badge. Two decisions worth narrating: (1) the **Day-1 spike gates** the whole approach — train a tiny LoRA end-to-end through GGUF+mmproj before committing, with a text-LoRA fallback if the round-trip fails; (2) **baseline-first eval** — you can't claim "fine-tuning improved accuracy" without the un-tuned number, and the test set is held out from generation so it's honest.
