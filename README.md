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

### Step 1 — Download the model

Both the model GGUF and the multimodal projector are required. Vision will silently fail without the projector.

```bash
huggingface-cli download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  --include "*UD-Q4_K_XL*" \
  --include "*mmproj-BF16*" \
  --local-dir ./models/nemotron
```

Result:
```
./models/nemotron/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
./models/nemotron/mmproj-BF16.gguf
```

---

### Step 2 — Start the inference server

Pick the method that matches your setup.

#### Option A — Docker (recommended)

Copy the example env file and fill in the model paths:

```bash
cp llama-server.env.example llama-server.env
```

Edit `llama-server.env`:
```env
LLAMA_ARG_MODEL=/models/nemotron/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
LLAMA_ARG_MMPROJ=/models/nemotron/mmproj-BF16.gguf
LLAMA_ARG_ALIAS=nemotron
LLAMA_ARG_MODELS_DIR=/models
LLAMA_ARG_MODELS_MAX=1
LLAMA_ARG_THREADS=1
LLAMA_ARG_FLASH_ATTN=1
LLAMA_ARG_N_BATCH=4096
LLAMA_ARG_N_UBATCH=2048
LLAMA_ARG_N_GPU_LAYERS=99
LLAMA_ARG_CTX_SIZE=32768
LLAMA_ARG_HOST=0.0.0.0
LLAMA_ARG_PORT=8080
LLAMA_ARG_TEMP=1.0
LLAMA_ARG_TOP_P=1.0
LLAMA_ARG_TOP_K=0
LLAMA_ARG_JINJA=true
LLAMA_ARG_LOAD_TIMEOUT=300
```

Then start the container, mounting your local models directory:

```bash
docker run --rm --gpus all \
  --env-file llama-server.env \
  -p 8080:8080 \
  -v "$(pwd)/models:/models" \
  ghcr.io/ggml-org/llama.cpp:server-cuda
```

> If you have models on an NFS share, replace the `-v` mount with a Docker volume pointing at your NFS path. See the [docker-compose example](#docker-compose) below.

#### Option B — Local llama.cpp binary

```bash
MODEL_DIR=./models/nemotron
MODEL=${MODEL_DIR}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
MMPROJ=${MODEL_DIR}/mmproj-BF16.gguf

llama-server \
  -m ${MODEL} \
  --mmproj ${MMPROJ} \
  --port 8080 \
  --host 0.0.0.0 \
  --n-gpu-layers 99 \
  --jinja \
  --ctx-size 32768
```

> `--n-gpu-layers 99` = full offload to VRAM. Reduce if OOM; the model needs ~25 GB across VRAM + system RAM for Q4_K_XL.

#### Verify the server is up

```bash
curl http://localhost:8080/health
# → {"status":"ok"}
```

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
