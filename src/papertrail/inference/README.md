# inference — talking to the model

**Responsibility:** send images to the model and return validated `schema` objects. Hide *where* the model runs.

## `client/`
- `base.py` — the interface every backend implements: `extract(images, instruction) -> ExtractedExpense`.
- `llama_client.py` — real backend. Calls llama-server `POST /v1/chat/completions` with `image_url` base64 blocks. **Strips `reasoning_content` / `<think>…</think>` before JSON parse** (reasoning-model requirement).
- `mock_client.py` — returns the same response shape with no GPU. Toggle via `PAPERTRAIL_MOCK=true`.

## `server/`
- `args.py` — flag logic; `server.py` — entrypoint (`scripts/serve.sh` wraps it). Parses `llama-server --help` (real source of truth, captured to `_llama_server_help.txt`) and propagates **every** flag into an argparse CLI, then applies Nemotron defaults (`--mmproj`, `--ctx-size 16384`, `--flash-attn on`, `--fit on`, `--no-context-shift`, `--jinja`, `--alias`, `--parallel 1`). **No pinned `--n-gpu-layers`** — `--fit on` auto-offloads as many layers as fit in free VRAM (pinning ngl disables fit's back-off and OOMs a 22 GiB model on a busy 24 GiB card). Emits a `docker run`/local command or `LLAMA_ARG_*` env.
  - Run: `python -m papertrail.inference.server.server --print` (dry-run) / `--run --detach`; refresh flags for your build with `--refresh-help`.
  - Note: this wraps the **C++ `llama-server`**, not `llama_cpp.Llama` — the verified vision flags (`--mmproj`/`--jinja`/`--chat-template-kwargs`) are not on the Python class.

**Must NOT contain:** eligibility/scoring rules, Gradio, DB.

**Why it exists / talking point:** `mock_client` and `llama_client` behind one `base` interface = **dependency inversion**. Your brother built the entire UI on the mock for 5 days with an 8 GB card that can't load the model, then flipped one env var to hit your 3090 over Tailscale. Same parser, same Pydantic validation, zero rework. The reasoning-strip lives here so the UI never sees `<think>` blocks.

> The Docker/compose that actually *runs* llama-server lives in `/deploy`, not here — that's an ops artifact, not importable Python.
