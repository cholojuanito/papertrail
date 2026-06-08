# deploy — llama-server ops artifacts

Not importable Python — this is how the inference server actually runs. Lives at repo root because Docker build contexts and the existing `llama-server.env.example` / compose pattern belong next to the repo, not inside the package.

**Planned files**
- `docker/Dockerfile` — pins `ghcr.io/ggml-org/llama.cpp:server-cuda` (CUDA 12.x — **not** 13.2) + model/mmproj mount points.
- `docker-compose.yml` — the service from the README quickstart; NFS volume option for NAS-hosted models.
- `funnel.md` — Tailscale Funnel steps to expose `:8080` for the Gradio Space during judging.

**Why split from `src/papertrail/inference/`:** `inference/server/args.py` is the Python that *knows the flags*; `deploy/` is the container that *runs them*. Keeping ops artifacts out of the importable package means `pip install`/tests never drag in Docker concerns, and the deploy story is one obvious folder a teammate or judge can read.
