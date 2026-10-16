#!/usr/bin/env python3
"""
Pkg 3: Human-vs-judge calibration of the LLM evaluation [last updated 2026-10-16]

Joins reviewer feedback (the feedback form's JSON exports, whether submitted
via POST /api/reviews or saved manually from the page) against the judge's
scores in the results CSV, and reports per evaluation dimension: how often
human and judge agree (raw agreement and Cohen's kappa), plus every
disagreement with the judge's stated reason — the cases worth reading to
decide which side is wrong and how to fix the judge.

Usage (from the repo root):
    python -m src.feedback_form.calibration                 # exports/ + settings data_path
    python -m src.feedback_form.calibration --reviews my_export.json
    python -m src.feedback_form.calibration --csv data/cleaned.csv --disagreements out.csv

Open Issues:
    - Exports are anonymous by design, so reviews cannot be grouped per
      reviewer and human-vs-human (inter-rater) agreement is not computable.
      If wanted later, export the page's device seed as a non-personal
      session id.

Resumption:
    The report adapts to metric changes without edits: it reads whatever
    human_<dim> columns an export contains and matches each to the judge's
    <dim>__score column (or the legacy <dim>_score spelling).
"""
import argparse
import csv
import json
import math
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from ..settings import load_settings

HERE = Path(__file__).resolve().parent
EXPORTS_DIR = HERE / "exports"
SNIPPET_LEN = 90


def load_review_records(paths: Iterable[Path]) -> list[dict]:
    """Load reviewer records from export files and/or directories of them.

    Args:
        paths: JSON export files, or directories scanned for *.json.

    Returns:
        The concatenated ``records`` lists of every export found.
    """
    files: list[Path] = []
    for p in paths:
        p = Path(p)
        files.extend(sorted(p.glob("*.json")) if p.is_dir() else [p])
    records = []
    for f in files:
        payload = json.loads(Path(f).read_text())
        records.extend(payload.get("records", []))
    return records


def judge_column(df: pd.DataFrame, dim: str, field: str = "score") -> Optional[str]:
    """Judge CSV column for a dimension: <dim>__score, or legacy <dim>_score."""
    for col in (f"{dim}__{field}", f"{dim}_{field}"):
        if col in df.columns:
            return col
    return None


def join_pairs(records: list[dict], df: pd.DataFrame) -> tuple[list[dict], int]:
    """Pair each human answer with the judge's score for the same text/prompt/dimension.

    Joins on (original text, reviewed variant) == (original_text, prompt),
    whitespace-stripped. Blank human answers and judge parse errors (-1) are
    skipped.

    Args:
        records: Reviewer records from load_review_records().
        df: Long-format results CSV with judge score/reason columns.

    Returns:
        (pairs, unmatched) — one pair dict per (record, dimension) holding
        dim/human/judge/reason/original/variant, and the count of reviewer
        records that matched no CSV row.
    """
    index = {
        (str(row["original_text"]).strip(), str(row["prompt"]).strip()): row
        for _, row in df.iterrows()
    }
    pairs, unmatched = [], 0
    for rec in records:
        key = (str(rec.get("original", "")).strip(), str(rec.get("reviewed_variant", "")).strip())
        row = index.get(key)
        if row is None:
            unmatched += 1
            continue
        for col, human in rec.items():
            if not col.startswith("human_") or human not in (0, 1):
                continue
            dim = col[len("human_"):]
            score_col = judge_column(df, dim)
            if score_col is None or row[score_col] not in (0, 1, "0", "1"):
                continue
            reason_col = judge_column(df, dim, "reason")
            pairs.append({
                "dim": dim,
                "human": int(human),
                "judge": int(row[score_col]),
                "judge_reason": str(row[reason_col]) if reason_col else "",
                "original": str(rec.get("original", "")),
                "variant": rec.get("reviewed_variant", ""),
            })
    return pairs, unmatched


def cohen_kappa(pairs: list[tuple[int, int]]) -> float:
    """Cohen's kappa for binary (human, judge) pairs; NaN when undefined.

    Kappa is undefined when expected agreement is 1 (both raters constant);
    that reads as "no evidence either way", not as 0.
    """
    n = len(pairs)
    if n == 0:
        return math.nan
    po = sum(1 for h, j in pairs if h == j) / n
    ph1 = sum(h for h, _ in pairs) / n
    pj1 = sum(j for _, j in pairs) / n
    pe = ph1 * pj1 + (1 - ph1) * (1 - pj1)
    if pe == 1.0:
        return math.nan
    return (po - pe) / (1 - pe)


def calibrate(records: list[dict], df: pd.DataFrame) -> dict:
    """Compute the per-dimension calibration report.

    Args:
        records: Reviewer records from load_review_records().
        df: Long-format results CSV.

    Returns:
        Dict with ``per_dim`` (dimension -> {n, human_pass, judge_pass,
        agreement, kappa}), ``disagreements`` (list of pair dicts where
        human != judge), and ``unmatched`` (reviewer records with no CSV row).
    """
    pairs, unmatched = join_pairs(records, df)
    per_dim: dict[str, dict] = {}
    for dim in sorted({p["dim"] for p in pairs}):
        dp = [p for p in pairs if p["dim"] == dim]
        hj = [(p["human"], p["judge"]) for p in dp]
        n = len(dp)
        per_dim[dim] = {
            "n": n,
            "human_pass": sum(h for h, _ in hj) / n,
            "judge_pass": sum(j for _, j in hj) / n,
            "agreement": sum(1 for h, j in hj if h == j) / n,
            "kappa": cohen_kappa(hj),
        }
    return {
        "per_dim": per_dim,
        "disagreements": [p for p in pairs if p["human"] != p["judge"]],
        "unmatched": unmatched,
    }


def _fmt(x: float) -> str:
    return " n/a" if isinstance(x, float) and math.isnan(x) else f"{x:4.2f}"


def print_report(result: dict) -> None:
    """Print the calibration tables and the disagreement cases."""
    per_dim, disagreements = result["per_dim"], result["disagreements"]
    if not per_dim:
        print("No joinable (human, judge) pairs found — nothing to calibrate.")
        return

    print(f"{'Dimension':<30} {'n':>4} {'human%':>7} {'judge%':>7} {'agree':>6} {'kappa':>6}")
    print("-" * 64)
    for dim, s in per_dim.items():
        print(f"{dim:<30} {s['n']:>4} {s['human_pass']:>7.2f} {s['judge_pass']:>7.2f} "
              f"{s['agreement']:>6.2f} {_fmt(s['kappa']):>6}")
    if result["unmatched"]:
        print(f"\nNote: {result['unmatched']} reviewed record(s) matched no row in the results CSV.")

    print(f"\nDisagreements ({len(disagreements)}):")
    for p in disagreements:
        snippet = p["original"].replace("\n", " ")[:SNIPPET_LEN]
        print(f"  [{p['dim']}] human={p['human']} judge={p['judge']} ({p['variant']})")
        print(f"    judge reason: {p['judge_reason']}")
        print(f"    original: {snippet}…")


def write_disagreements_csv(disagreements: list[dict], path: Path) -> None:
    """Write the disagreement cases to a CSV for spreadsheet triage."""
    cols = ["dim", "human", "judge", "judge_reason", "variant", "original"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows({c: p[c] for c in cols} for p in disagreements)


def main() -> None:
    default_csv = load_settings().get("feedback_form", {}).get("data_path", "data/cleaned.csv")
    ap = argparse.ArgumentParser(description="Human-vs-judge calibration report.")
    ap.add_argument("--reviews", nargs="+", type=Path, default=[EXPORTS_DIR],
                    help="export JSON files and/or directories (default: feedback_form/exports/)")
    ap.add_argument("--csv", type=Path, default=Path(default_csv),
                    help="long-format results CSV with the judge's scores")
    ap.add_argument("--disagreements", type=Path, default=None,
                    help="also write disagreement cases to this CSV")
    args = ap.parse_args()

    records = load_review_records(args.reviews)
    print(f"Loaded {len(records)} reviewed record(s) from {len(args.reviews)} source(s).")
    df = pd.read_csv(args.csv)

    result = calibrate(records, df)
    print_report(result)
    if args.disagreements and result["disagreements"]:
        write_disagreements_csv(result["disagreements"], args.disagreements)
        print(f"\nWrote {len(result['disagreements'])} disagreement(s) to {args.disagreements}")


if __name__ == "__main__":
    main()
