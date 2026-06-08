# app — Gradio UI (wiring only)

**Responsibility:** the interface. Upload flow, results display, inline correction, dashboard, export. Nothing else.

**Planned files**
- `main.py` — Gradio app entrypoint (the repo-root `main.py` calls into this).
- `components/` — upload panel, extraction result + confidence highlighting, readiness-score widget, correction form, dashboard/search, audit-report export.
- `theme/` — the custom Gradio theme (the "Off-Brand" badge requirement).

**Must NOT contain:** eligibility rules, scoring math, SQL, prompt strings, reasoning-strip logic. It *calls* `inference.client`, `core`, and `storage`; it doesn't reimplement them.

**Why it exists / talking point:** keeping the UI dumb is what makes the rest testable — when a verdict looks wrong you know it's `core`, not a display bug. The UI's only "intelligence" is highlighting low-confidence fields (a `schema` field) and writing corrections back through `storage`. Custom theme + readiness widget + dashboard is the polish the Backyard AI track is judged on.
