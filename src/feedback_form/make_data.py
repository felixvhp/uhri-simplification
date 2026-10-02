#!/usr/bin/env python3
"""
Pkg 3: Extract review pairs (original + AI drafts) from a simplification
results CSV into review_data.json, which build.py then bakes into the page.
[last updated 2026-10-02]

The results CSV is the long layout written by the simplification step
(`python -m src.simplification ... --layout long`): one row per
(record, prompt) with an `original_text`, `prompt` and `transcribed_text`
column; any other columns are ignored here. Rows are grouped by
`original_text`, so each record becomes one review item with one draft per
prompt.

Usage (from the repo root):
    python -m src.feedback_form.make_data                       # all records
    python -m src.feedback_form.make_data --sample 80           # random 80 records
    python -m src.feedback_form.make_data --sample 80 --seed 7  # reproducible draw
    python -m src.feedback_form.make_data --csv data/cleaned.csv

Then rebuild the page:
    python -m src.feedback_form.build
"""
import argparse
import csv
import json
import random
from pathlib import Path

from ..settings import load_settings

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = Path(load_settings()["feedback_form"]["data_path"])
OUT = HERE / "review_data.json"

REQUIRED_COLS = {"original_text", "prompt", "transcribed_text"}


def main() -> None:
    ap = argparse.ArgumentParser(description="simplification results CSV -> review_data.json")
    ap.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="simplification results CSV")
    ap.add_argument("--sample", type=int, default=None, help="draw this many records at random (default: all)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for --sample (reproducible)")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with open(args.csv, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        fields = set(reader.fieldnames or [])
        rows = list(reader)

    missing = REQUIRED_COLS - fields
    if missing:
        raise SystemExit(
            f"Missing column(s) {sorted(missing)} in {args.csv}.\n"
            "Expected the long layout (one row per record+prompt) written by "
            "`python -m src.simplification ... --layout long`."
        )

    # Group rows by original text: one review item per record, one draft per prompt.
    grouped: dict[str, dict[str, str]] = {}
    for row in rows:
        orig = (row.get("original_text") or "").strip()
        prompt = (row.get("prompt") or "").strip()
        draft = (row.get("transcribed_text") or "").strip()
        if not orig or not prompt or not draft:
            continue
        grouped.setdefault(orig, {})[prompt] = draft

    items = [
        {"id": i, "original": orig, "variants": variants}   # id = record order in CSV, stays stable
        for i, (orig, variants) in enumerate(grouped.items())
    ]

    total = len(items)
    if args.sample and args.sample < total:
        items = sorted(random.Random(args.seed).sample(items, args.sample), key=lambda x: x["id"])

    prompts = sorted({p for it in items for p in it["variants"]})
    args.out.write_text(json.dumps(items, ensure_ascii=False))
    note = f" (sampled from {total})" if args.sample and args.sample < total else ""
    print(f"Wrote {args.out.name}: {len(items)} records{note}. Prompts seen: {prompts}")


if __name__ == "__main__":
    main()
