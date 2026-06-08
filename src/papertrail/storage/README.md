# storage — local persistence + training loop

**Responsibility:** persist records to SQLite and append every human correction to a training-ready JSONL.

**Planned files**
- `migrations/001_init.sql` — the three tables from PLAN.md §5: `documents`, `expenses`, `corrections` + indexes.
- `db.py` — connection / schema bootstrap.
- `repository.py` — typed read/write over `schema` objects (no raw SQL leaking into the app).
- `corrections.py` — on every UI edit: write the `corrections` row **and** append to `corrections.jsonl` in `(image, instruction, corrected JSON)` format.

**Must NOT contain:** model calls or eligibility logic.

**Why it exists / talking point:** the corrections loop is the "production ML story" — every manual fix becomes gold-standard labelled data for the next fine-tune. It's deliberately a thin repository, not an ORM: the queries in PLAN.md §5.4 (YTD eligible total, flagged-for-review, duplicates) are all structured, so SQLite + indexes beats a vector DB. Talking point: "I chose SQLite because every query I have is a `WHERE` on a column I index — adding a vector store would be complexity with no question it answers."
