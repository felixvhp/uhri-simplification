#!/usr/bin/env python3
"""
Extract review pairs (original + AI drafts) from a simplification results CSV
into review_data.json, which build_review_tool.py then bakes into the page.

The results CSV is produced by src/simplification.py (long/tidy format):
one row per (record, prompt) with an `original_text`, `prompt` and
`transcribed_text` column, plus metric columns which are ignored here.
Rows are grouped by `original_text`, so each record becomes one review item
with one draft per prompt.

Wide-format CSVs from the pre-June-2026 script (one column per prompt name)
are not supported; rerun src/simplification.py to produce a current CSV.

Usage:
    python review/make_review_data.py                       # all records
    python review/make_review_data.py --sample 80           # random 80 records
    python review/make_review_data.py --sample 80 --seed 7  # reproducible draw
    python review/make_review_data.py --csv ../data/cleaned.csv

Then rebuild the page:
    python review/build_review_tool.py
"""
import argparse
import csv
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent          # code/review
ROOT = HERE.parent                               # code
DEFAULT_CSV = ROOT.parent / "data" / "cleaned.csv"   # simplification.py's OUTPUT_PATH
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
            "Expected the long format written by src/simplification.py "
            "(one row per record+prompt). Wide-format CSVs from the old "
            "llm-text-simplification.py script are no longer supported."
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
