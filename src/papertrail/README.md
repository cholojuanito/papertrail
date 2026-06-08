# PaperTrail — Package Layout

This is the architecture map. Each subpackage owns one responsibility and has a documented boundary so two people can build in parallel without colliding.

```
src/papertrail/
├── schema/       Integration contract — the JSON shape everything agrees on. Pure data, no logic.
├── core/         Domain logic. Pure functions, no I/O, no GPU. The part you unit-test hardest.
│   ├── eligibility/   IRC §213(d) / Pub 502 lookup → eligible/ineligible/partial/flagged
│   └── scoring/       Audit Readiness Score (0–100) from field completeness + confidence
├── preprocess/   Image prep: PDF→image, deskew, contrast. Turns "a file" into "model-ready images".
├── inference/    Talking to the model.
│   ├── client/        OpenAI-compatible client + mock stub (same interface, dev with no GPU)
│   └── server/        llama-server launch config / args (the deployable lives in /deploy)
├── storage/      SQLite persistence + the corrections→JSONL training loop.
│   └── migrations/    schema DDL (documents, expenses, corrections)
├── app/          Gradio UI ONLY. Wiring, components, theme. No business logic.
│   ├── components/
│   └── theme/
├── train/        Everything fine-tuning.
│   ├── data/          synthetic doc generation (ReportLab + Pillow)
│   ├── augment/       Albumentations degradation pipeline
│   ├── modal/         the Modal A100-80GB job
│   └── eval/          before/after field-accuracy metrics
└── experiments/  Reproducible runs that produce the numbers for the blog post.
```

Repo-root, outside the Python package:
```
deploy/           Docker + compose for the llama-server (ops artifacts, not importable code)
scripts/          One-command setup + run helpers
tests/            Unit/integration tests
```

## The dependency rule (the one thing to remember)

Arrows point inward. `schema` and `core` depend on **nothing** in this project. `app`, `inference`, `storage`, `train` may depend on `core`/`schema`, but never the reverse.

```
app ─┐
inference ─┤
storage ─┼──> core ──> schema
train ─┤
preprocess ─┘
```

Why it matters (interview answer): the parts most likely to be wrong (eligibility rules, scoring) are the parts with zero dependencies — so they run in milliseconds in a unit test with no model, no GPU, no DB. The parts that need a GPU or network (`inference`, `app`) hold no rules worth testing in isolation.
