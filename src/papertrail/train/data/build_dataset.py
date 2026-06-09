"""Build the synthetic training dataset.

Generates documents per the PLAN.md 4.3 distribution, applies the Albumentations
degradation pipeline, assigns an 80/10/10 split, and writes:

    <out>/images/<id>.png            rendered (+ degraded) document images
    <out>/manifest.jsonl             one row per doc (see schema below)
    <out>/summary.json               counts per split and doc_type

Manifest row:
    {"id", "doc_type", "split", "image", "instruction", "response", "fields"}
  - "image": path relative to <out>
  - "response": the exact JSON string the model is trained to emit
  - "fields": the same data as a parsed object (handy for eval / inspection)

Usage:
    uv run python -m papertrail.train.data.build_dataset --out data/synthetic --n 100 --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from papertrail.schema.extraction import EXTRACTION_INSTRUCTION

from ..augment.pipeline import degrade, make_pipelines
from .generators import GENERATORS


def _scaled_counts(n: int) -> dict[str, int]:
    """Scale the default per-type counts (sum 100) to a total of `n`."""
    base = {name: cnt for name, (_, cnt) in GENERATORS.items()}
    total = sum(base.values())
    scaled = {name: max(1, round(cnt * n / total)) for name, cnt in base.items()}
    # Fix rounding drift so the counts sum to exactly n.
    drift = n - sum(scaled.values())
    order = sorted(scaled, key=lambda k: scaled[k], reverse=True)
    i = 0
    while drift != 0 and order:
        k = order[i % len(order)]
        if drift > 0:
            scaled[k] += 1
            drift -= 1
        elif scaled[k] > 1:
            scaled[k] -= 1
            drift += 1
        i += 1
    return scaled


def _assign_split(rng: random.Random, val_frac: float, test_frac: float) -> str:
    r = rng.random()
    if r < test_frac:
        return "test"
    if r < test_frac + val_frac:
        return "val"
    return "train"


def build(out: Path, n: int, seed: int, augment: bool, val_frac: float, test_frac: float) -> None:
    rng = random.Random(seed)
    (out / "images").mkdir(parents=True, exist_ok=True)
    pipelines = make_pipelines() if augment else None

    # Build the work list, then shuffle so splits are not clustered by type.
    counts = _scaled_counts(n)
    work: list[str] = []
    for name, cnt in counts.items():
        work.extend([name] * cnt)
    rng.shuffle(work)

    manifest_path = out / "manifest.jsonl"
    split_counter: Counter = Counter()
    type_counter: Counter = Counter()

    with manifest_path.open("w", encoding="utf-8") as mf:
        for i, doc_type in enumerate(work):
            gen = GENERATORS[doc_type][0]
            doc = gen(rng)
            img = doc.image
            if pipelines is not None:
                img = degrade(img, doc_type, pipelines, rng)

            doc_id = f"{i:05d}_{doc_type}"
            rel = f"images/{doc_id}.png"
            img.save(out / rel)

            split = _assign_split(rng, val_frac, test_frac)
            response = doc.target.target_json()
            row = {
                "id": doc_id,
                "doc_type": doc.doc_type,
                "split": split,
                "image": rel,
                "instruction": EXTRACTION_INSTRUCTION,
                "response": response,
                "fields": json.loads(response),
            }
            mf.write(json.dumps(row, ensure_ascii=False) + "\n")
            split_counter[split] += 1
            type_counter[doc.doc_type] += 1

    summary = {
        "total": len(work),
        "seed": seed,
        "augmented": augment,
        "by_split": dict(split_counter),
        "by_type": dict(type_counter),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print(f"\nwrote {len(work)} docs -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate the PaperTrail synthetic dataset.")
    ap.add_argument("--out", type=Path, default=Path("data/synthetic"))
    ap.add_argument("--n", type=int, default=100, help="total documents to generate")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-augment", dest="augment", action="store_false")
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--test-frac", type=float, default=0.1)
    args = ap.parse_args()
    build(args.out, args.n, args.seed, args.augment, args.val_frac, args.test_frac)


if __name__ == "__main__":
    main()
