# core — domain logic (pure, testable, no I/O)

**Responsibility:** the actual rules of the product. Given extracted fields, decide eligibility and score audit readiness.

## `eligibility/`
- `table.py` — the static lookup dict (IRC §213(d) / Pub 969 / Notice 2004-2, Pub 502 base table). The OTC-eligible-post-CARES override lives here.
- `classify.py` — `classify(expense) -> EligibilityVerdict`. Pure function.

## `scoring/`
- `readiness.py` — `score(expense) -> (int, breakdown)`. The 0–100 completeness+confidence logic and the per-field deductions shown in the UI panel.

**Must NOT contain:** model calls, DB writes, Gradio, file reads. Inputs are `schema` objects, outputs are `schema` objects.

**Why it exists / talking point:** eligibility and scoring are where a bug is *invisible* — a wrong verdict still looks plausible. By keeping them pure, every rule is covered by a fast unit test (`classify(otc_drug) == eligible`, `score(missing_amount) drops N points`). No GPU, no network. This is the "test behavior, not plumbing" core of the project and the most defensible thing to demo: deterministic, auditable rules separate from the probabilistic model.
