# schema — the integration contract

**Responsibility:** define the exact JSON shape the model emits and the app consumes. One source of truth.

**Planned files**
- `extraction.py` — Pydantic models: `ExtractedExpense`, `Confidence`, `LineItem`. The `(merchant, date, amount, provider_type, line_items, confidence{})` structure from PLAN.md §4.1.
- `eligibility.py` — `EligibilityVerdict` (`eligible|ineligible|partial|flagged` + IRS citation).
- `record.py` — the full persisted shape that maps to the SQLite tables.

**Must NOT contain:** any logic, I/O, model calls, or DB access. Just shapes + validation.

**Why it exists / talking point:** this was the **Day-1 deliverable** in the sprint. Defining the contract first is what let one person build the Gradio UI against a mock stub while the other built the real inference path — both conform to the same Pydantic models, so when they meet on Day 6 nothing has to be reshaped. The mock stub and the live llama-server response validate against the *same* models.
