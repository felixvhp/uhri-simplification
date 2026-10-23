#!/usr/bin/env python3
"""
Pkg 3: Extract review pairs (original + AI drafts) from a simplification
results CSV into review_data.jsonl, which serve.py samples from per reviewer
and build.py bakes into the standalone page. [last updated 2026-10-23]

The results CSV is the long layout written by the simplification step
(`python -m src.simplification ... --layout long`): one row per
(record, prompt) with an `original_text`, `prompt` and `transcribed_text`
column; any other columns are ignored here. Rows are grouped by
`original_text`, so each record becomes one review item with one draft per
prompt. Output is JSON Lines (one record per line) so the server can index
and sample it without reading the whole file; the full UHRI export fits.

Usage (from the repo root):
    python -m src.feedback_form.make_data                       # all records (default)
    python -m src.feedback_form.make_data --csv data/cleaned.csv
    python -m src.feedback_form.make_data --sample 80 --seed 7  # pre-trim the pool

Keep every record here; reviewers get their random subset from the server
(settings: feedback_form.sample_size). --sample only trims the pool itself.
The standalone page needs a rebuild afterwards: python -m src.feedback_form.build
"""
import argparse
import csv
import json
import random
import re
from pathlib import Path

from ..settings import load_settings

HERE = Path(__file__).resolve().parent
DEFAULT_CSV = Path(load_settings()["feedback_form"]["data_path"])
OUT = HERE / "review_data.jsonl"

REQUIRED_COLS = {"original_text", "prompt", "transcribed_text"}

# Same cleanup as simplification.core.strip_markdown, copied rather than
# imported (the tools never import each other): older results CSVs still
# carry the model's **bold** headings.
_MD_BOLD = re.compile(r"(\*\*|__)(.+?)\1", re.S)
_MD_HEADING = re.compile(r"^[ \t]{0,3}#{1,6}[ \t]+", re.M)


def strip_markdown(text: str) -> str:
    """Remove ``**``, ``__`` and leading ``#`` markers from a draft."""
    text = _MD_BOLD.sub(r"\2", text)
    text = _MD_HEADING.sub("", text)
    return text.replace("**", "").strip()


def main() -> None:
    ap = argparse.ArgumentParser(description="simplification results CSV -> review_data.jsonl")
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
        draft = strip_markdown(row.get("transcribed_text") or "")
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
    with open(args.out, "w", encoding="utf-8") as f:
        for item in items:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    note = f" (sampled from {total})" if args.sample and args.sample < total else ""
    print(f"Wrote {args.out.name}: {len(items)} records{note}. Prompts seen: {prompts}")


if __name__ == "__main__":
    main()
