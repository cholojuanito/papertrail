# experiments — reproducible numbers for the blog post

**Responsibility:** scripted runs that produce the data behind every claim in the Field Notes post. Each writes results to a committed file, so numbers are reproducible — not remembered.

**Planned experiments**
- `thinking_latency.py` — same docs through `enable_thinking: true` vs `false`. Captures latency (you measured 71s with thinking on) **and** the accuracy delta. The headline blog tradeoff: reasoning cost vs. extraction quality.
- `context_vram.py` — sweep `-c` (4096 → 32768) on the 3090, record VRAM + whether multi-page EOBs fit. Decides per-page-split vs single-pass.
- `finetune_before_after.py` — baseline (un-tuned) vs fine-tuned field accuracy on the held-out test set. The "Well-Tuned" evidence.
- `throughput.py` — tokens/sec under each config (you saw 21.8 tok/s) and end-to-end seconds per document.

**Must NOT contain:** product logic. Experiments *import* `inference`, `core`, `train.eval` and orchestrate them; they don't reimplement.

**Why it exists / talking point:** a blog post that says "fine-tuning improved accuracy" is hand-waving; one that links a script + a results file is engineering. Separating experiments from the app means your benchmark claims are rerunnable on demand — if a judge asks "how'd you get 21.8 tok/s," you point at the script, not your memory.
