# scripts — one-command setup & run helpers

Thin wrappers so a fresh clone (or your brother) gets running without memorising flags.

**Planned scripts**
- `download_model.sh` — the `huggingface-cli download` for model + mmproj (README Step 1).
- `serve.sh` — launch llama-server with the confirmed flags; `THINKING=on|off` switch.
- `dev.sh` — run the Gradio app against the mock client (`PAPERTRAIL_MOCK=true`), no GPU.
- `gen_data.sh` — run `train/data` + `train/augment` to build the synthetic set.

**Why it exists / talking point:** every script is a decision you'd otherwise re-explain in Discord. `serve.sh THINKING=off` is the demo-day button; `dev.sh` is what let UI work happen on an 8 GB card. "I made the two-person workflow a two-command workflow."
