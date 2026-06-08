# tests

Mirrors the package. The bulk of value is in `core/` — pure functions, no GPU, no network.

**Planned coverage**
- `test_eligibility.py` — every override that's easy to get wrong: OTC drug = **eligible** (post-CARES), menstrual = eligible, gym/cosmetic = ineligible. Behavior, not the table's current contents.
- `test_scoring.py` — missing field → expected deduction; full record → 100; invariant: score ∈ [0,100].
- `test_schema.py` — a real llama-server response (with `reasoning_content`) and the mock both validate to the same model; reasoning-strip yields parseable JSON.
- `test_repository.py` — correction writes a row **and** appends to `corrections.jsonl`.

**Why it exists / talking point:** the model is probabilistic and tested by the eval split in `train/eval`; these tests cover the **deterministic** half — the rules a human is accountable for in an audit. Fast, offline, run on every change. Don't test defaults (a renamed citation string shouldn't break a test); test logic (OTC classifies eligible).
