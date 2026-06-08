# inference — talking to the model

**Responsibility:** send images to the model and return validated `schema` objects. Hide *where* the model runs.

## `client/`
- `base.py` — the interface every backend implements: `extract(images, instruction) -> ExtractedExpense`.
- `llama_client.py` — real backend. Calls llama-server `POST /v1/chat/completions` with `image_url` base64 blocks. **Strips `reasoning_content` / `<think>…</think>` before JSON parse** (reasoning-model requirement).
- `mock_client.py` — returns the same response shape with no GPU. Toggle via `PAPERTRAIL_MOCK=true`.

## `server/`
- `args.py` — the confirmed llama-server flag set (PLAN.md §3.2): `--mmproj`, `--flash-attn on`, `--no-context-shift`, `--fit on`, `--jinja`, `--spec-default`, `--chat-template-kwargs`, `-c`. Two profiles: thinking-on (accuracy) / thinking-off (demo speed).

**Must NOT contain:** eligibility/scoring rules, Gradio, DB.

**Why it exists / talking point:** `mock_client` and `llama_client` behind one `base` interface = **dependency inversion**. Your brother built the entire UI on the mock for 5 days with an 8 GB card that can't load the model, then flipped one env var to hit your 3090 over Tailscale. Same parser, same Pydantic validation, zero rework. The reasoning-strip lives here so the UI never sees `<think>` blocks.

> The Docker/compose that actually *runs* llama-server lives in `/deploy`, not here — that's an ops artifact, not importable Python.
