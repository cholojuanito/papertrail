# preprocess — file → model-ready images

**Responsibility:** turn an uploaded receipt photo or EOB PDF into clean image(s) the model can read, and route by doc type.

**Planned files**
- `load.py` — accept image or PDF; `pdf2image` renders PDF pages to images.
- `clean.py` — OpenCV deskew, contrast normalise.
- `route.py` — detect doc type (`receipt_photo | eob_pdf | …`) for downstream handling.
- `paginate.py` — split multi-page documents (the 4096-context constraint from the test means multi-page EOBs may need per-page inference, then merge).

**Must NOT contain:** model calls or eligibility logic.

**Why it exists / talking point:** the Jun-8 test showed the Omni model nails clean docs with no preprocessing — but real phone photos are skewed, dim, low-DPI. This is also where the **training/inference symmetry** matters: the Albumentations degradations in `train/augment` simulate exactly the artifacts this module must survive. Note in the blog: minimal preprocessing on clean inputs, escalate only if real-receipt accuracy drops.
