---
title: PaperTrail
emoji: 🧾
colorFrom: blue
colorTo: green
sdk: gradio
app_file: main.py
pinned: false
---
# PaperTrail
### *Every expense. Proven.*

A fully local, privacy-first medical expense intelligence app. Drop in a receipt or insurance EOB and get structured extraction, HSA eligibility classification, and an audit-readiness score — entirely on your own hardware. Nothing leaves the machine.

Built for the [HuggingFace Build Small Hackathon 2026](https://huggingface.co/build-small-hackathon).

---

## Quickstart

### Prerequisites

- NVIDIA GPU with CUDA 12.x (**not** 13.2 — known gibberish output for this model)
- Docker + NVIDIA Container Toolkit **or** a local llama.cpp build
- [uv](https://docs.astral.sh/uv/) for Python deps
- ~25 GB free disk for the model + projector

---

### Step 1 — Get the model (skip if you already have it)

Both the model GGUF and the multimodal projector are required — vision silently fails without the projector. Download into `<models-root>/<repo>` (the launcher auto-discovers the filenames):

```bash
hf download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  --include "*UD-Q4_K_XL*" --include "*mmproj-BF16*" \
  --local-dir ./models/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF
```

**Already have it elsewhere?** Just point the launcher at your HF local-dir — no re-download:

```bash
export PAPERTRAIL_MODELS_ROOT=/mnt/chuno-models/llms   # dir that contains the repo folder
```

### Step 2 — Start the inference server

One command. It parses `llama-server --help`, applies the verified Nemotron flags (`--mmproj`, `--jinja`, `--fit on`, `--ctx-size 32768`, `--n-gpu-layers 99`, …), auto-discovers your model + projector, and launches it.

```bash
uv sync

./scripts/serve.sh                  # Docker, thinking on, detached (default)
BACKEND=local ./scripts/serve.sh    # use a local llama-server binary instead
THINKING=off ./scripts/serve.sh     # demo speed (no reasoning)
DRYRUN=1 ./scripts/serve.sh         # print the exact command without launching
```

Override any server flag inline, e.g. `./scripts/serve.sh --ctx-size 16384 --port 8001`.
Stop a detached Docker server with `docker rm -f papertrail-llama`.

**Configuration** (all optional, via env):

| Env var | Default | Meaning |
|---|---|---|
| `PAPERTRAIL_MODELS_ROOT` | `./models` | Your HF local-dir (mounted to `/models` in Docker) |
| `PAPERTRAIL_MODEL_REPO` | `NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF` | Repo subdir to load |
| `PAPERTRAIL_QUANT` | `UD-Q4_K_XL` | Preferred quant when multiple GGUFs exist |
| `PAPERTRAIL_MODEL` / `PAPERTRAIL_MMPROJ` | (auto) | Full paths, override discovery entirely |

Loading a different model is just `PAPERTRAIL_MODEL_REPO=<repo> ./scripts/serve.sh`.

#### Verify the server is up

```bash
curl http://localhost:8080/health
# → {"status":"ok"}
```

#### Under the hood / manual control

The launcher entrypoint is `python -m papertrail.inference.server.server` (logic lives in `args.py`). See the exact `docker run`/`llama-server` command it builds with `--print`, or the equivalent compose env with `--print-env`:
```bash
uv run python -m papertrail.inference.server.server --print
uv run python -m papertrail.inference.server.server --print-env > llama-server.env
```

> Defaults rely on `--fit on` with **no** pinned `--n-gpu-layers`, so llama.cpp auto-offloads as many layers as fit in free VRAM (the rest run on CPU). The 22.3 GiB Q4 model needs ~23 GiB free for *full* GPU offload — close GPU apps / run headless, then add `--n-gpu-layers 99` for max speed.

After switching to a new llama.cpp build, refresh the known flags with `--refresh-help`.

---

### Step 3 — Run PaperTrail

```bash
uv sync
uv run python main.py
```

The Gradio UI opens at `http://localhost:7860`. Set the inference endpoint in `.env`:

```env
LLAMA_BASE_URL=http://localhost:8080
```

---

### Remote inference (brother's machine / Tailscale)

If the model is running on a different machine over Tailscale:

```env
LLAMA_BASE_URL=http://<tailscale-ip>:8080
```

The Gradio UI calls the OpenAI-compatible `POST /v1/chat/completions` endpoint using `image_url` base64 content blocks. The server response includes a `reasoning_content` / `<think>…</think>` block — the app strips this before JSON parsing.

During local UI development without a GPU, use the mock stub (`PAPERTRAIL_MOCK=true`) which returns the same response shape as the live server.

---

### docker-compose

For NFS-hosted models or persistent setups, use a compose file. Adapt from the pattern below (mirrors the home-lab setup):

```yaml
services:
  llama-server:
    image: ghcr.io/ggml-org/llama.cpp:server-cuda
    container_name: papertrail-llamacpp
    ports:
      - "8080:8080"
    volumes:
      - ./models:/models          # or an NFS volume — see below
    env_file:
      - llama-server.env
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    restart: unless-stopped

# NFS volume example (if models live on a NAS):
# volumes:
#   nfs-models:
#     driver: local
#     driver_opts:
#       type: nfs
#       o: "addr=<nas-ip>,rw,nfsvers=4.2,rsize=1048576,wsize=1048576,hard,timeo=600,retrans=3"
#       device: ":/<path-on-nas>"
```

```bash
docker compose up -d
```

---

## Architecture

Local-first: Gradio UI → llama-server (OpenAI-compatible) → SQLite. No cloud, no API keys.

```
Receipt / EOB PDF  →  Pre-processor (OpenCV + pdf2image)
                   →  Nemotron 3 Nano Omni 31B (llama-server --mmproj)
                   →  strip reasoning block  →  parse JSON
                   →  IRS Pub 502 eligibility lookup
                   →  Audit Readiness Score (0–100)
                   →  SQLite  →  Gradio UI
```

See [PLAN.md](PLAN.md) for full spec, database schema, fine-tuning plan, and sprint timeline.

---

## Inference notes

- **CUDA 12.x only** — CUDA 13.2 produces gibberish output for this model.
- **Ollama does not work** — requires separate mmproj; use llama.cpp directly.
- **Reasoning model** — responses contain a `<think>…</think>` / `reasoning_content` block before the JSON payload. Always strip before parsing.
- For extraction tasks: `temperature=0.2`, `top_k=1`.
