# PaperTrail
### *Every expense. Proven.*
**Project Spec & Design Doc · HuggingFace Build Small Hackathon 2026**

---

| Field | Detail |
|---|---|
| **Hackathon Track** | Chapter One — Backyard AI |
| **Submission Deadline** | June 15, 2026 |
| **Team** | 2 people |
| **Target Badges** | All 6 — Off the Grid, Llama Champion, Off-Brand, Well-Tuned, Sharing is Caring, Field Notes |
| **Model** | Nemotron 3 Nano Omni 31B + LoRA (single model — vision + text) |
| **Inference Runtime** | llama.cpp — RTX 3090 (primary) / Modal A100-80GB (fine-tune) |
| **Collaboration** | Tailscale Funnel — brother's Gradio app hits inference server on 3090 |
| **Fine-Tune Budget** | $250 Modal hackathon credits |
| **UI Framework** | Gradio — custom theme |
| **Storage** | SQLite — fully local, no cloud dependency |

---

## Table of Contents

1. [Problem](#1-problem)
2. [Solution](#2-solution)
3. [Architecture](#3-architecture)
4. [Data & Fine-Tuning](#4-data--fine-tuning)
5. [Database Schema](#5-database-schema)
6. [Tiered MVP](#6-tiered-mvp)
7. [7-Day Sprint (June 8–15)](#7-7-day-sprint-june-815)
8. [Contest Strategy](#8-contest-strategy)
9. [Constraints & Risks](#9-constraints--risks)
10. [Team Work Split](#10-team-work-split)
11. [Appendix](#appendix--quick-reference)

---

## 1. Problem

Anyone who claims medical expenses against a tax-advantaged account — HSA in the US, METC in Canada, Krankheitskosten in Germany, HMRC self-employment claims in the UK — is legally required to retain documentation for every expense. In practice, most people manage this with a folder of receipts and good intentions. When an audit notice arrives, the process is entirely manual.

**Core pain points:**
- No quick way to verify whether a past expense is eligible under applicable tax rules
- Document formats vary wildly — paper receipts, insurance EOBs, emailed invoices, scanned superbills
- Existing tools require cloud upload, creating a privacy risk with sensitive medical documents
- Documentation gaps only surface during an audit, not during the year when they can still be fixed

> **Real User:** Primary demo user is a team member — an actual HSA account holder with real receipts across pharmacy, dental, and medical categories. On-screen demo uses synthetic documents only. Real receipts used for offline personal testing only.

---

## 2. Solution

PaperTrail is a fully local, privacy-first Gradio application. Drop in any medical expense document and a single fine-tuned Nemotron 3 Nano Omni 31B model extracts structured fields, classifies eligibility against a static IRS Pub 502 lookup table, scores audit readiness out of 100, and stores a correctable record to SQLite. Nothing leaves the machine.

### 2.1 Why a Single Omni Model

Nemotron 3 Nano Omni 31B handles vision and text in a single model — no pipeline handoffs between a VLM and a separate language model. Despite 31B total parameters, only ~3B are active per inference step (MoE architecture), making it fast on a single 3090. It delivers strong document-intelligence and OCR performance directly relevant to receipt and EOB extraction. 256K context window handles multi-page documents trivially.

> **Parameter Budget Check**
> - Nemotron 3 Nano Omni total params: **31B ✓** (hackathon limit: 32B; model card: 3.1×10¹⁰, branded "Nano Omni 30B")
> - Active params per inference step: **~3B** (MoE — fast on single GPU)
> - Single model replaces the prior two-model architecture entirely

> **⚠️ Reasoning Model Note**
> This is a **Reasoning** model. For extraction tasks use **Instruct mode** (temp 0.2, top_k 1). The model response includes a `reasoning_content` / `<think>…</think>` block before the JSON — **strip it before JSON parsing**. The mock stub must mirror this response shape so brother's UI parser works identically against both stub and live server.

### 2.2 Core Features (V1 — must ship)

**Structured Extraction**
The fine-tuned model extracts merchant, date, amount, line items, provider type, and per-field confidence scores as structured JSON. Low-confidence fields are highlighted for manual correction. Extraction quality is the product — everything else depends on it.

**Audit Readiness Score**
Every record receives a 0–100 score based on field completeness and confidence. The panel shows exactly what is present and what is missing — actionable, not opaque.

```
Audit Readiness   87 / 100

✓  Provider name
✓  Service date
✓  Amount paid
✗  Itemised line items    (−8)
✗  Provider NPI / tax ID  (−5)
```

**Eligibility Classification**
Each expense is classified as `eligible / ineligible / partial / flagged` using a static lookup table governed by **IRC §213(d) / IRS Pub 969 / Notice 2004-2**, with IRS Pub 502 as the medical-expense base table. HSA-specific overrides apply where HSA rules differ from the general medical deduction (OTC drugs and menstrual products are **eligible** for HSA post-CARES Act 2020, even without a prescription). Fast, reliable, explainable — no LLM reasoning chain required for standard cases. Every verdict carries the IRS rule reference.

**Manual Correction UI + Training Loop**
Low-confidence fields are editable inline. Every correction is written to SQLite and simultaneously appended to `corrections.jsonl` in training-ready `(image, instruction, corrected JSON)` format — building a verified dataset for future fine-tuning runs automatically.

**Local-First Privacy**
All inference runs via llama.cpp on the user's own GPU. No API keys, no cloud upload, no telemetry. Medical documents never leave the machine.

### 2.3 Features Deliberately Cut

> ❌ **Intentionally out of scope**
> - OCR reconciliation layer — single fine-tuned model handles this; add back only if extraction quality is poor post fine-tune
> - Separate orchestrator model — Nemotron Omni replaces the prior two-model architecture entirely
> - Audio ingestion — llama.cpp audio support for this model is broken; not a demo feature
> - Gap detection — needs sufficient history to be meaningful; V3 stretch only
> - Document classes beyond receipts and EOBs — scope risk; invoices and statements are V3
> - CPU HF Space optimisation — Space hosts UI only; inference runs on live 3090 via tunnel
> - Vector database — all queries are structured (date, merchant, eligibility); SQLite with indexes is sufficient

---

## 3. Architecture

| Component | Tool | Role | Notes |
|---|---|---|---|
| Pre-processor | Python / OpenCV + pdf2image | Image prep and doc-type routing | Deskew, contrast enhance, PDF→image |
| Inference | Nemotron 3 Nano Omni 31B + LoRA | Extraction (vision + text) | Single model — vision + language; strip reasoning block before parse |
| Eligibility DB | Static JSON / Python dict | IRC §213(d) / IRS Pub 502 lookup table | No LLM call for standard cases |
| Storage | SQLite + JSONL | Local persistence + training loop | Corrections auto-append to JSONL |
| Export | Python / csv | Audit report generation | Sorted by Audit Readiness Score |
| UI | Gradio | User interface | Custom theme, hosted on HF Space (UI only — inference via tunnel) |

### 3.1 Pipeline Flow

```
User drops receipt photo or EOB PDF into Gradio UI
        ↓
[Pre-processor]
  Detect doc type, deskew, normalise contrast, render PDF pages to images
        ↓
[Nemotron 3 Nano Omni 31B — fine-tuned LoRA]
  via llama-server OpenAI-compatible endpoint (image_url base64 content block)
  → strip reasoning_content / <think> block
  → structured JSON with per-field confidence scores
        ↓
[Static IRS Pub 502 / IRC §213(d) Eligibility Lookup]
  provider_type + line items → eligible / ineligible / partial / flagged
  + IRS rule citation
        ↓
[Audit Readiness Score]
  0–100 based on field completeness and confidence
        ↓
[SQLite write]
  Full record persisted locally
        ↓
[Gradio UI]
  Low-confidence fields highlighted for correction
  Every correction → SQLite + corrections.jsonl (training-ready)
```

### 3.2 Inference Commands

> **⚠️ CUDA Version:** Do **not** use CUDA 13.2 — known gibberish output for this model. Use CUDA 12.x.
> **⚠️ Ollama:** Ollama multimodal does not work for this model (separate mmproj architecture). Use llama.cpp only.

The command below is derived from the confirmed-working Unsloth Studio invocation (tested June 8 — 21.8 tok/s on RTX 3090, 100% extraction accuracy on real docs).

```bash
MODEL_DIR=/path/to/nemotron-gguf
MODEL=${MODEL_DIR}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
MMPROJ=${MODEL_DIR}/mmproj-BF16.gguf

# Server mode — production (thinking on: ~71s, highest accuracy)
llama-server \
  -m ${MODEL} \
  --mmproj ${MMPROJ} \
  --port 8080 \
  --host 0.0.0.0 \
  --flash-attn on \
  --no-context-shift \
  --fit on \
  --threads -1 \
  --jinja \
  --spec-default \
  --chat-template-kwargs '{"enable_thinking": true}' \
  -c 32768 \
  --parallel 1

# Demo / speed mode (thinking off: much faster, slightly lower accuracy)
llama-server \
  -m ${MODEL} \
  --mmproj ${MMPROJ} \
  --port 8080 \
  --host 0.0.0.0 \
  --flash-attn on \
  --no-context-shift \
  --fit on \
  --threads -1 \
  --jinja \
  --spec-default \
  --chat-template-kwargs '{"enable_thinking": false}' \
  -c 32768 \
  --parallel 1
```

> **Context budget:** Unsloth Studio loaded with `-c 4096`; a small invoice + receipt used ~2700 tokens of that. For multi-page EOBs, test `-c 32768` (may need VRAM headroom check on 3090). If 32768 OOMs, process one page at a time and concatenate results.
>
> **Thinking mode tradeoff:** `enable_thinking: true` → 71s latency, higher accuracy (tested 100% on real docs). `enable_thinking: false` → much faster, suitable for demo; quality difference worth measuring and noting in the blog post.
>
> The server exposes an **OpenAI-compatible `chat/completions` endpoint**. Send images as `image_url` with base64-encoded data. Strip `reasoning_content` / `<think>…</think>` from the response before JSON parsing.

### 3.3 Hardware & Hosting

**Local Dev & Demo — RTX 3090 (24 GB VRAM + 32 GB RAM)**
```
Q4_K_XL GGUF requires ~25 GB — fits across VRAM + system RAM with layer offload
llama-server bound to 0.0.0.0 on port 8080; exposed via Tailscale Funnel for Gradio Space
Target latency: < 20 seconds end-to-end per document
```

**Brother's Machine — RTX 3060 Ti (8 GB VRAM)**
```
Cannot run 31B model locally — 8 GB VRAM insufficient even with full offload

Days 1–5:  Mock stub locally (returns realistic fake JSON matching live server shape) for UI development
Days 6–7:  Gradio app points to 3090 inference server via Tailscale Funnel

Tailscale setup: ~10 minutes, zero router config, works across any network

Server command (your machine):
  llama-server \
    -m ${MODEL_DIR}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf \
    --mmproj ${MODEL_DIR}/mmproj-BF16.gguf \
    --port 8080 --host 0.0.0.0 \
    --n-gpu-layers 99 --jinja --ctx-size 32768

Brother's .env:
  LLAMA_BASE_URL=http://<your-tailscale-ip>:8080

Integration contract (Day 1 deliverable):
  POST /v1/chat/completions — image_url base64 in, JSON (after stripping reasoning) out
  Mock stub returns identical shape so brother's parser works against both stub and live server.
```

**Fine-Tuning — Modal (A100-80GB, $250 hackathon credits)**
```
Nemotron 3 Nano Omni vision LoRA via Unsloth (NVIDIA-confirmed day-zero support)
A100-80GB required — 30B+ LoRA needs ~60-80GB VRAM; A100-40GB is insufficient
Modal A100-80GB rate: ~$2.50/hr → ~$5-10 for a 2-4h run (well within $250 budget)
Estimated training time: 2–4 hours for 80 examples / 3 epochs

Day 1 gate: confirm credits are granted AND A100-80GB is available on the credit tier.
  If 80GB unavailable: attempt QLoRA-4bit on A100-40GB as documented fallback (same Day 1 spike gates it).

Publish adapter to HF Hub on completion.
```

**HF Space — Submission**
```
Gradio Space hosts the UI only — no inference in-Space.
Inference calls the live RTX 3090 over Tailscale Funnel during the judging window.
This is on-brand for "Off the Grid": own hardware, no third-party cloud API.

README documents:
  - Local-first design intent and hardware requirements
  - That inference runs on author's 3090 via tunnel during judging
  - Pre-recorded demo video link (backstop if tunnel is unreachable at judging time)
  - Scheduled live window for judges who want to test interactively
```

---

## 4. Data & Fine-Tuning

### 4.1 Fine-Tuning Objective

**Primary path (vision LoRA):** Instruction-following supervised fine-tune: given an image of a medical expense document plus a fixed extraction instruction, the model outputs a structured JSON object. Training examples are `(image, instruction, JSON response)` triples. Gated by the Day-1 full de-risk spike.

**Fallback path (text LoRA):** If the vision-LoRA→GGUF+mmproj round-trip fails the Day-1 spike, switch to a text LoRA: the base Omni model already does OCR/extraction via prompting; the LoRA maps extracted text → normalized eligibility JSON. Text LoRA merges to GGUF cleanly and still earns the "Well-Tuned" badge.

```json
// Training format — each example (vision LoRA)
{
  "instruction": "Extract all expense fields from this document as JSON.",
  "response": {
    "merchant": "City Dental Group",
    "date": "2026-03-14",
    "amount": 85.00,
    "provider_type": "dental",
    "line_items": ["cleaning", "x-rays"],
    "confidence": {
      "merchant": 0.97,
      "date": 0.95,
      "amount": 0.99
    }
  }
}
```

### 4.2 Dataset Split

| Split | Size | Purpose |
|---|---|---|
| Train | ~68 docs (80%) | LoRA fine-tuning on Modal |
| Validation | ~8 docs (10%) | Monitor overfitting during training |
| Test | ~8 docs (10%) | Held-out eval — report before/after field accuracy in blog post and demo |

**Primary metric:** per-field exact-match accuracy on `amount`, `date`, and `merchant` across the test set. Baseline (un-fine-tuned model) recorded first so improvement is quantifiable and citable.

### 4.3 Synthetic Document Plan

| Document Type | Tool | Count | Key Scenarios |
|---|---|---|---|
| Pharmacy receipt | ReportLab | 25 | Prescriptions (eligible) vs OTC (also eligible post-CARES Act 2020); cosmetic/gym ineligible |
| Insurance EOB | ReportLab | 25 | Single and multi-page; service amount vs. patient responsibility |
| Doctor visit receipt | ReportLab | 15 | Co-pays, superbills with procedure codes |
| Handwritten receipt | Pillow + fonts | 10 | Caveat / Homemade Apple font overlaid on receipt template |
| Ineligible edge cases | ReportLab | 10 | Gym membership, vitamins, cosmetic — tests eligibility refusal |
| Hard negatives | ReportLab | 15 | Partial receipts, obscured amounts, missing dates, multi-page totals |
| **Total** | | **100** | |

### 4.4 Image Degradation Pipeline (Albumentations)

All synthetic docs pass through an augmentation pipeline before training. Clean PDFs will not generalise to real-world phone photographs without this step.

| Augmentation | Simulates |
|---|---|
| Random rotation ±15° | Document photographed at a slight angle |
| JPEG compression artifacts | Low-quality phone camera or MMS-forwarded receipt |
| Brightness / contrast shift | Faded thermal receipt paper or poor lighting |
| Gaussian blur | Out-of-focus phone camera shot |
| Perspective warp | Receipt photographed from the side |
| Salt-and-pepper noise | Low-DPI scanner output |

### 4.5 Corrections-to-Training Loop

Every manual field correction in the UI is written to SQLite and appended to `corrections.jsonl` in training-ready format. This gives PaperTrail a production ML story: every correction is gold-standard training data for the next fine-tuning run.

```python
# Appended on every user correction
{
  "image_path": "receipts/doc_00042.jpg",
  "instruction": "Extract all expense fields from this document as JSON.",
  "response": { ...corrected_fields... },
  "source": "human_correction",
  "corrected_at": "2026-06-10T14:23:00"
}
```

---

## 5. Database Schema

### 5.1 `documents` table
Stores one record per uploaded file.

```sql
CREATE TABLE documents (
  id                 INTEGER PRIMARY KEY AUTOINCREMENT,
  filename           TEXT NOT NULL,
  file_path          TEXT NOT NULL,         -- local path to stored image/PDF
  doc_type           TEXT NOT NULL,         -- 'receipt_photo' | 'eob_pdf' | 'invoice' | 'statement'
  upload_timestamp   TEXT NOT NULL,         -- ISO 8601
  raw_model_output   TEXT,                  -- full JSON from Nemotron (audit log)
  status             TEXT DEFAULT 'pending' -- 'pending' | 'validated' | 'flagged' | 'manual_review'
);

CREATE INDEX idx_documents_status ON documents(status);
CREATE INDEX idx_documents_upload ON documents(upload_timestamp);
```

### 5.2 `expenses` table
Stores one record per extracted line item. One document can produce multiple expense records (e.g. an EOB with several services).

```sql
CREATE TABLE expenses (
  id                       INTEGER PRIMARY KEY AUTOINCREMENT,
  doc_id                   INTEGER NOT NULL REFERENCES documents(id),

  -- Extracted fields
  merchant                 TEXT,
  provider_type            TEXT,     -- 'pharmacy' | 'dental' | 'vision' | 'medical' | 'other'
  service_date             TEXT,     -- ISO 8601 date string
  amount                   REAL,
  line_item_description    TEXT,
  patient                  TEXT,

  -- Eligibility
  hsa_eligible             TEXT,     -- 'eligible' | 'ineligible' | 'partial' | 'flagged'
  irs_citation             TEXT,     -- e.g. 'IRS Pub 502 p.8 — Dental Treatment'
  eligibility_confidence   REAL,     -- 0.0 – 1.0

  -- Extraction quality
  extraction_confidence    REAL,     -- overall confidence across all fields
  low_confidence_fields    TEXT,     -- JSON array of field names below threshold
  manually_corrected       INTEGER DEFAULT 0,  -- 1 if user edited any field

  -- Audit readiness
  audit_readiness_score    INTEGER,  -- 0–100
  readiness_breakdown      TEXT,     -- JSON object: field → points deducted

  -- Deduplication
  duplicate_of             INTEGER REFERENCES expenses(id),
  duplicate_confidence     REAL
);

CREATE INDEX idx_expenses_doc_id       ON expenses(doc_id);
CREATE INDEX idx_expenses_service_date ON expenses(service_date);
CREATE INDEX idx_expenses_eligible     ON expenses(hsa_eligible);
CREATE INDEX idx_expenses_merchant     ON expenses(merchant);
```

### 5.3 `corrections` table
Full audit trail of every manual field edit. Used both for compliance ("AI-extracted, human-verified") and as a source of training data.

```sql
CREATE TABLE corrections (
  id               INTEGER PRIMARY KEY AUTOINCREMENT,
  expense_id       INTEGER NOT NULL REFERENCES expenses(id),
  field_name       TEXT NOT NULL,     -- e.g. 'amount', 'service_date', 'merchant'
  original_value   TEXT,
  corrected_value  TEXT,
  corrected_at     TEXT NOT NULL      -- ISO 8601 timestamp
);

CREATE INDEX idx_corrections_expense ON corrections(expense_id);
```

### 5.4 Common Queries

```sql
-- YTD eligible expenses total
SELECT SUM(amount) FROM expenses
WHERE hsa_eligible = 'eligible'
  AND service_date >= '2026-01-01';

-- All flagged / low-confidence records needing review
SELECT e.*, d.filename FROM expenses e
JOIN documents d ON e.doc_id = d.id
WHERE e.hsa_eligible = 'flagged'
   OR e.audit_readiness_score < 70
ORDER BY e.audit_readiness_score ASC;

-- Potential duplicates
SELECT * FROM expenses
WHERE duplicate_of IS NOT NULL;

-- Audit report export — sorted by risk
SELECT
  d.filename,
  e.service_date,
  e.merchant,
  e.amount,
  e.hsa_eligible,
  e.irs_citation,
  e.audit_readiness_score,
  e.manually_corrected
FROM expenses e
JOIN documents d ON e.doc_id = d.id
ORDER BY e.audit_readiness_score ASC;
```

---

## 6. Tiered MVP

### ✅ V1 — Guaranteed Finish (Days 1–5)
- Synthetic data generation with Albumentations degradation pipeline
- Nemotron 3 Nano Omni 31B LoRA fine-tune on Modal A100-80GB — eval split with before/after accuracy metrics
  - **Vision LoRA** (image→JSON): default path, gated by Day-1 full de-risk spike
  - **Text LoRA** (extracted-text→JSON): committed fallback if vision-LoRA→GGUF round-trip fails Day-1 spike
- Receipt + EOB extraction via fine-tuned model (structured JSON output; reasoning block stripped)
- Eligibility classification via static IRC §213(d) / IRS Pub 502 lookup table (OTC drugs eligible post-CARES Act)
- SQLite storage with all three tables + corrections-to-JSONL training loop
- Manual correction UI with low-confidence field highlighting
- Audit Readiness Score (0–100) with missing field breakdown

### 🔶 V2 — Strong Submission (Days 6–7)
- Audit report CSV export sortable by readiness score
- Evaluation dashboard showing fine-tune before/after metrics
- Custom Gradio theme and full UI polish
- Duplicate detection (same merchant + amount + date window)
- Demo video + Field Notes blog post

### 🔴 V3 — Stretch (only if clearly ahead of schedule)
- Multi-page EOB document linking
- International document format variants (UK, Canada, Germany)
- Gap detection across full dataset

---
## 7. 7-Day Sprint (June 8–15)

| Day(s) | Phase | Goal | Owner |
|---|---|---|---|
| 1 (Jun 8) | Setup & Fine-Tune Spike | **Vision model confirmed working** (Jun 8 test: 21.8 tok/s, 100% accuracy on real docs, Unsloth Studio). Remaining Day 1: (1) Confirm Modal credits + A100-80GB access. (2) Train tiny vision LoRA on ~5-10 docs → merge → convert to GGUF+mmproj → verify extraction still works. **If loop fails → text-LoRA fallback.** (3) Set up Tailscale Funnel (expose port 8080). (4) Brother sets up mock stub matching real OpenAI-compatible response shape (with reasoning block). (5) Test `-c 32768` context on 3090 — needed for multi-page EOBs. | Both |
| 2–3 (Jun 9–10) | Data | Generate 100 synthetic docs (ReportLab + Pillow). Run Albumentations degradation pipeline. Create 80/10/10 train/val/test split. Record baseline extraction accuracy on test set. | Brother (gen) / You (eval) |
| 3–4 (Jun 10–11) | Fine-Tune | Run full Nemotron 3 Nano Omni LoRA on Modal A100-80GB. Record before/after field accuracy on test set. Publish adapter + dataset to HF Hub. | You |
| 4–5 (Jun 11–12) | Pipeline | Build pre-processor (OpenCV + pdf2image). Wire fine-tuned model extraction end-to-end (OpenAI-compatible client, strip reasoning, parse JSON). Build static eligibility lookup. Audit Readiness Score logic. | You |
| 5–6 (Jun 12–13) | Storage & UI Core | SQLite schema (all 3 tables) + corrections JSONL loop. Manual correction UI. Readiness score widget. Full upload → result flow working end-to-end against real model via Tailscale. | Both |
| 6–7 (Jun 13–14) | V2 Features + Demo | Audit report CSV export. Duplicate detection. Eval dashboard. Gradio custom theme and polish. Demo video with synthetic docs. Field Notes blog post. Social media post copy. | Both |
| 7 (Jun 15) | Ship | HF Space deployment (UI + tunnel README). Agent trace publish. Full checklist review. Submit by deadline. V3 only if everything else is done. | Both |
---

## 8. Contest Strategy

### 8.1 Track
**Chapter One — Backyard AI.** Judged on: real problem, real user, honest fit with small-model constraints, Gradio polish. PaperTrail qualifies on all four axes. The international reframe (HSA / METC / HMRC / Krankheitskosten) broadens appeal without changing the core scope.

### 8.2 Target Badges

| Badge | Requirement | How We Qualify |
|---|---|---|
| 🔌 Off the Grid | No cloud APIs at runtime | All inference via local llama.cpp on 3090; Gradio Space UI calls live 3090 over Tailscale Funnel — own hardware, no third-party API |
| 🦙 Llama Champion | Runs via llama.cpp | `llama-server` with `--mmproj` for vision + text inference on 3090 |
| 🎨 Off-Brand | Custom Gradio UI | Custom theme, document viewer, readiness score widget, dashboard |
| 🎯 Well-Tuned | Fine-tuned model published to HF Hub | Nemotron 3 Nano Omni LoRA adapter (vision or text) trained on Modal, published |
| 📡 Sharing is Caring | Publish agent trace to HF Hub | Full pipeline traces logged and pushed as HF dataset |
| 📓 Field Notes | Blog post on HF | Before/after fine-tune metrics, VLM limits on real receipts, IRS lookup design decisions |

### 8.3 Submission Checklist

- [ ] Confirm team is registered under hackathon HF org — verify before Day 1
- [ ] Confirm Nemotron sponsor prize criteria in hackathon Discord — check Day 1
- [ ] Confirm Modal credits granted and A100-80GB tier available — check Day 1
- [ ] App submitted under hackathon HF org namespace
- [ ] HF Space running — UI only, inference via Tailscale Funnel to live 3090, documented in README
- [ ] Pre-recorded demo video uploaded and linked in README (backstop if tunnel unreachable at judging)
- [ ] Scheduled live judging window noted in README (so judges can request a live session)
- [ ] Demo video: synthetic receipt → extraction → readiness score → audit report
- [ ] Social media post live before deadline
- [ ] LoRA adapter published to HF Hub
- [ ] Synthetic dataset published to HF Hub
- [ ] Agent trace dataset published to HF Hub
- [ ] Field Notes blog post published on HF

---

## 9. Constraints & Risks

### 9.1 Hard Constraints
- Total model parameters ≤ 32B — Nemotron 3 Nano Omni: **31B ✓** (model card: 3.1×10¹⁰; branded "30B"; well within 32B limit)
- UI must be Gradio
- Space submitted under hackathon HF org
- Submission deadline: June 15, 2026

### 9.2 Risk Register

| Risk | Likelihood | Mitigation |
|---|---|---|
| 3090 runs OOM on Q4_K_XL GGUF | Low | **Jun 8 test: model runs on 3090 at 21.8 tok/s.** Test `-c 32768` for multi-page EOBs; drop to `-c 4096` per-page if needed. Q3_K_M fallback (~20GB) if OOM. |
| Context limit too small for multi-page EOBs | Medium | 2 small docs used ~2700 of 4096 tokens. Test 32768 context; if VRAM-constrained, split at page boundaries and merge results in pipeline. |
| Thinking mode too slow for demo (71s observed) | Low | Use `enable_thinking: false` for demo mode. Measure accuracy delta; document tradeoff in blog post. |
| Vision-LoRA adapter cannot be converted to GGUF+mmproj for llama.cpp | Medium | Day-1 fine-tune spike gates the decision. Text-LoRA fallback committed — still earns Well-Tuned badge. |
| Modal A100-80GB unavailable on credit tier | Medium | Confirm on Day 1. QLoRA-4bit on A100-40GB is documented fallback (same spike gates it). |
| Modal fine-tune cost overruns | Low | $250 budget. Est. spend $5-10 training + $8-10 dev inference. Well within budget. |
| Tailscale tunnel / host unavailable during judging | Medium | Pre-recorded demo video as backstop; scheduled live window in README. |
| Nemotron sponsor prize criteria unclear | Medium | Check Discord on Day 1. Single-model Nemotron is the strongest possible qualifying position. |
| Fine-tune overruns Days 3–4 | Medium | Time-box strictly. Submit base model if needed — write honestly about attempt in blog post. |
| Reasoning block in model output breaks JSON parse | Low | Strip `reasoning_content` / `<think>…</think>` before parsing. Mock stub mirrors this shape from Day 1. |

---

## 10. Team Work Split

| Area | Owner |
|---|---|
| llama.cpp + Nemotron GGUF + mmproj setup, layer offload tuning | You |
| Tailscale Funnel setup (expose 3090 inference server) | You |
| Pre-processor — OpenCV + pdf2image | You |
| Fine-tuning pipeline on Modal (Unsloth) | You |
| Extraction prompt engineering + JSON schema (reasoning block stripping) | You |
| Static IRC §213(d) / IRS Pub 502 eligibility lookup table | You |
| Audit Readiness Score logic | You |
| SQLite schema + corrections JSONL loop | Both |
| Synthetic data generation scripts (ReportLab + Pillow) | Brother |
| Albumentations degradation pipeline | Brother |
| Mock inference stub (matching real server response shape, with reasoning block) | Brother |
| Gradio UI — upload flow, extraction results, corrections | Brother |
| Gradio UI — dashboard, search, audit report export | Brother |
| HF Space deployment + README (tunnel docs + demo video link) | Brother |
| Demo video recording and editing | Both |
| Field Notes blog post | You |
| Social media post | Both |
| Agent trace logging + HF dataset publish | Both |

---

## Appendix — Quick Reference

### Model Download
```bash
# Correct repo and filenames — download both model and multimodal projector
huggingface-cli download unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF \
  --include "*UD-Q4_K_XL*" \
  --include "*mmproj-BF16*"
# Model file: NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
# Projector:  mmproj-BF16.gguf
```

### llama.cpp Inference Commands

> Commands derived from confirmed-working Unsloth Studio invocation (Jun 8 — 21.8 tok/s, 100% accuracy on real docs).

```bash
MODEL_DIR=/path/to/nemotron-gguf
MODEL=${MODEL_DIR}/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-UD-Q4_K_XL.gguf
MMPROJ=${MODEL_DIR}/mmproj-BF16.gguf

# Server mode — production (thinking on: ~71s latency, highest accuracy)
llama-server \
  -m ${MODEL} \
  --mmproj ${MMPROJ} \
  --port 8080 \
  --host 0.0.0.0 \
  --flash-attn on \
  --no-context-shift \
  --fit on \
  --threads -1 \
  --jinja \
  --spec-default \
  --chat-template-kwargs '{"enable_thinking": true}' \
  -c 32768 \
  --parallel 1

# Server mode — demo / speed (thinking off: much faster, slightly lower accuracy)
llama-server \
  -m ${MODEL} \
  --mmproj ${MMPROJ} \
  --port 8080 \
  --host 0.0.0.0 \
  --flash-attn on \
  --no-context-shift \
  --fit on \
  --threads -1 \
  --jinja \
  --spec-default \
  --chat-template-kwargs '{"enable_thinking": false}' \
  -c 32768 \
  --parallel 1
```

### Brother's Mock Stub (local UI dev — no GPU needed)
```python
# Returns shape identical to the live llama-server OpenAI-compatible response
# (including reasoning block) so brother's UI parser works against both stub and server.
def mock_extract(image):
    # Simulates full OpenAI-compatible chat/completions response shape
    return {
        "choices": [{
            "message": {
                "role": "assistant",
                "reasoning_content": "<think>This is a pharmacy receipt...</think>",
                "content": """{
  "merchant": "City Pharmacy",
  "date": "2026-05-14",
  "amount": 42.50,
  "provider_type": "pharmacy",
  "line_items": ["lisinopril 10mg — 30 day supply"],
  "confidence": {
    "merchant": 0.95,
    "date": 0.98,
    "amount": 0.99,
    "line_items": 0.87
  }
}"""
            }
        }]
    }

# Caller strips reasoning_content before JSON parsing:
def extract_json(response):
    content = response["choices"][0]["message"]["content"]
    return json.loads(content)
```

### IRS Pub 502 Eligibility Lookup (structure)
```python
# Governed by IRC §213(d) / IRS Pub 969 / Notice 2004-2.
# IRS Pub 502 is the medical-expense base table; HSA-specific overrides apply.
# Key override: OTC drugs and menstrual products are HSA-ELIGIBLE post-CARES Act (2020),
# even without a prescription (IRC §223(d)(2)(A) as amended).
ELIGIBILITY = {
    "dental":         {"status": "eligible",   "citation": "IRS Pub 502 — Dental Treatment"},
    "vision":         {"status": "eligible",   "citation": "IRS Pub 502 — Eye Exams"},
    "prescription":   {"status": "eligible",   "citation": "IRS Pub 502 — Medicines"},
    "otc_drug":       {"status": "eligible",   "citation": "IRC §223(d)(2)(A) — OTC drugs eligible post-CARES Act 2020"},
    "menstrual":      {"status": "eligible",   "citation": "IRC §223(d)(2)(A) — Menstrual products eligible post-CARES Act 2020"},
    "gym":            {"status": "ineligible", "citation": "IRS Pub 502 — Weight-Loss Programs"},
    "cosmetic":       {"status": "ineligible", "citation": "IRS Pub 502 — Cosmetic Surgery"},
    "chiropractic":   {"status": "eligible",   "citation": "IRS Pub 502 — Chiropractic Care"},
    "therapy":        {"status": "eligible",   "citation": "IRS Pub 502 — Psychiatric Care"},
    # ... extend for all Pub 502 categories
}
```

### Key Links

| Resource | URL |
|---|---|
| Hackathon | huggingface.co/build-small-hackathon |
| Nemotron 3 Nano Omni Reasoning GGUF | huggingface.co/unsloth/NVIDIA-Nemotron-3-Nano-Omni-30B-A3B-Reasoning-GGUF |
| Unsloth fine-tuning | github.com/unslothai/unsloth |
| Tailscale Funnel | tailscale.com/kb/1223/funnel |
| Albumentations | albumentations.ai |
| IRS Publication 502 | irs.gov/pub/irs-pdf/p502.pdf |
| IRS Publication 969 (HSA rules) | irs.gov/pub/irs-pdf/p969.pdf |
| Modal | modal.com |
