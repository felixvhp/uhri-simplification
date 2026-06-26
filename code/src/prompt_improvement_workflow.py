"""
Pkg 3: Prompt improvement workflow [last updated 2026-06-19]

Usage:
    cd code
    python src/prompt_improvement_workflow.py

    The script reads data/cleaned.csv (the long-format CSV produced by
    unpivoting simplification_results.csv — see analyse_simplification.ipynb)
    and prints a four-step evaluation report to stdout.

Reads a long-format CSV and runs a four-step evaluation workflow:
    1. Load and score: normalise FK-grade delta, compute weighted average.
    2. Metric variance check: flag low-variance metrics (poor discriminators).
    3. Prompt comparison: rank prompts by mean score, flag high-variance ones.
    4. Best-prompt verification: show sample outputs for manual inspection.

Open Issues:
    - FK normalisation uses the dataset's own min/max, so the scale shifts
      when new data is added.
    - Weights are hand-tuned; no principled calibration yet.

Resumption:
    - Tweak WEIGHTS in __main__, re-run, read the verify-best-prompt output.
    - Add new metrics to the CSV and they will be picked up automatically.
"""

import sys
from pathlib import Path

# When run as `python src/prompt_improvement_workflow.py` from code/,
# Python adds code/src/ to sys.path. We also need code/ so that src.*
# imports in this file and its dependencies (prompts.py, evaluation.py) resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional, Tuple

import pandas as pd
from src.prompts import PROMPTS
from src.evaluation import EVAL_DIMENSIONS

# Tuneable thresholds ─────────────────────────────────────────────────────────
# Variance below this flags a metric as nearly constant → poor discriminator.
LOW_VARIANCE_THRESHOLD = 0.04
# Variance above this flags a prompt as inconsistent across texts.
HIGH_VARIANCE_THRESHOLD = 0.03
# Default number of sample rows shown in each inspection block.
N_SAMPLES = 2


# ── Data loading ──────────────────────────────────────────────────────────────

# TODO: This function is quite hacky, and this is preferably handled somewhere with data cleaning, or something more central.
def get_score_columns(df: pd.DataFrame) -> List[str]:
    """Return all column names that end with ``_score``.

    Args:
        df: Long-format DataFrame from :func:`load_and_unpivot`.

    Returns:
        List of score column names (e.g. ``['content_preserved_score', ...]``).
    """
    return [col for col in df.columns if col.endswith("_score")]


# ── Score computation ─────────────────────────────────────────────────────────


def normalise_fk_grade(df: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalise ``fk_grade_delta`` into [0, 1] as ``fk_normalized_score``.

    A larger FK-grade delta means a greater reduction in reading difficulty,
    so higher normalised values indicate better readability improvement.

    Note: the scale is dataset-relative; adding new rows shifts the range.

    Args:
        df: Long-format DataFrame containing a ``fk_grade_delta`` column.

    Returns:
        df with ``fk_normalized_score`` added (modifies in place and returns).
    """
    lo, hi = df["fk_grade_delta"].min(), df["fk_grade_delta"].max()
    df["fk_normalized_score"] = (df["fk_grade_delta"] - lo) / (hi - lo)
    return df


def compute_weighted_average(
    df: pd.DataFrame,
    score_columns: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute a weighted mean over score columns and store it as ``average_score``.

    Args:
        df: Long-format DataFrame with all ``score_columns`` present.
        score_columns: Column names to include in the average.
        weights: Optional mapping of column name to weight. Columns not in the
                 dict default to 1.0. Pass ``None`` for equal weights.

    Returns:
        df with ``average_score`` added or updated (modifies in place and returns).
    """
    if weights is None:
        weights = {}
    w = pd.Series({col: weights.get(col, 1.0) for col in score_columns})
    df["average_score"] = df[score_columns].mul(w).sum(axis=1) / w.sum()
    return df


# ── Metric inspection ─────────────────────────────────────────────────────────


def check_metric_variance(
    df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = LOW_VARIANCE_THRESHOLD,
    n_samples: int = N_SAMPLES,
) -> None:
    """Print a variance report for each metric and inspect low-variance ones.

    A metric with low variance assigns nearly the same score to every output.
    It contributes little signal to the ranking and is a candidate for removal.
    For each flagged metric, sample rows are printed so the user can read the
    LLM's reasoning and decide whether the metric is worth keeping.

    Args:
        df: Long-format DataFrame with score and optional ``*_reason`` columns.
        score_columns: Metric score column names to evaluate.
        threshold: Variance below this triggers a LOW warning.
        n_samples: Number of sample rows to show per flagged metric.
    """
    variances = df[score_columns].var().sort_values()

    print("Metric variances (ascending):")
    for metric, var in variances.items():
        flag = "  <- LOW" if var < threshold else ""
        print(f"  {metric:<40} {var:.4f}{flag}")
    print()

    low_var = variances[variances < threshold].index.tolist()
    if not low_var:
        print("All metrics have sufficient variance — no candidates for removal.")
        return

    for metric in low_var:
        var = variances[metric]
        print("=" * 70)
        print(f"WARNING: '{metric}' has low variance ({var:.4f})")
        print("  -> Nearly constant score across all outputs.")
        print("  -> If the samples below look reasonable, consider removing this metric.")
        print()

        reason_col = metric.replace("_score", "_reason")
        sample_cols = ["original_text", "transcribed_text", "prompt", metric]
        if reason_col in df.columns:
            sample_cols.append(reason_col)

        sample = df[sample_cols].dropna(subset=[metric]).sample(min(n_samples, len(df)))
        _print_samples(sample, [metric])


def show_sanity_samples(
    df: pd.DataFrame,
    score_columns: List[str],
    n_samples: int = N_SAMPLES,
) -> None:
    """Print a full metric breakdown for randomly selected rows.

    A quick sanity-check that scores and LLM reasons look sensible across all
    metrics at once before comparing prompts.

    Args:
        df: Long-format DataFrame with score and optional ``*_reason`` columns.
        score_columns: Metric score columns to display per sample.
        n_samples: Number of rows to sample.
    """
    reason_cols = [c for c in df.columns if c.endswith("_reason")]
    display_cols = ["original_text", "transcribed_text", "prompt"] + score_columns + reason_cols
    sample = df[display_cols].sample(n_samples)

    print(f"=== Sanity check: {n_samples} random samples ===\n")
    _print_samples(sample, score_columns)


# ── Prompt comparison ─────────────────────────────────────────────────────────


def compare_prompts(
    df: pd.DataFrame,
    high_var_threshold: float = HIGH_VARIANCE_THRESHOLD,
) -> pd.DataFrame:
    """Rank prompts by mean ``average_score`` and flag high-variance ones.

    High variance on a prompt means it performs well on some texts and poorly
    on others — worth investigating before trusting its mean ranking.

    Args:
        df: Long-format DataFrame with ``prompt`` and ``average_score`` columns.
        high_var_threshold: Variance above this triggers a HIGH VARIANCE flag.

    Returns:
        DataFrame indexed by prompt with columns ``['mean_score', 'variance']``,
        sorted by ``mean_score`` descending.
    """
    stats = (
        df.groupby("prompt")["average_score"]
        .agg(["mean", "var"])
        .rename(columns={"mean": "mean_score", "var": "variance"})
        .sort_values("mean_score", ascending=False)
    )

    print("Prompt comparison (ranked by mean score):")
    print(f"  {'Prompt':<30} {'Mean':>6}  {'Var':>7}")
    print(f"  {'-'*30} {'-'*6}  {'-'*7}")
    for prompt, row in stats.iterrows():
        flag = "  <- HIGH VARIANCE" if row["variance"] > high_var_threshold else ""
        print(f"  {prompt:<30} {row['mean_score']:>6.3f}  {row['variance']:>7.4f}{flag}")
    print()

    high_var = stats[stats["variance"] > high_var_threshold]
    for prompt in high_var.index:
        print(f"WARNING: '{prompt}' has high variance ({high_var.loc[prompt, 'variance']:.4f})")
        print("  -> Score swings a lot across texts — inspect outputs before trusting its rank.")
    if not high_var.empty:
        print()

    return stats


def verify_best_prompt(
    df: pd.DataFrame,
    prompt_stats: pd.DataFrame,
    score_columns: List[str],
    n_samples: int = N_SAMPLES,
) -> None:
    """Show sample outputs from the top-ranked prompt for manual review.

    Args:
        df: Long-format DataFrame with ``prompt``, score, and ``*_reason`` columns.
        prompt_stats: DataFrame returned by :func:`compare_prompts`.
        score_columns: Metric score columns to display per sample.
        n_samples: Number of sample rows to show.
    """
    best = prompt_stats.index[0]
    mean = prompt_stats.loc[best, "mean_score"]
    print(f"Best prompt: '{best}'  (mean={mean:.3f})")
    print("Read the outputs below — does this look like good simplification?\n")

    reason_cols = [c for c in df.columns if c.endswith("_reason")]
    subset = df[df["prompt"] == best]
    display_cols = ["original_text", "transcribed_text"] + score_columns + reason_cols
    sample = subset[display_cols].sample(min(n_samples, len(subset)), random_state=7)

    _print_samples(sample, score_columns)


# ── Shared display helper ─────────────────────────────────────────────────────


def _print_samples(sample: pd.DataFrame, score_columns: List[str]) -> None:
    """Render sample rows as readable text blocks with scores and reasons.

    Args:
        sample: Rows to print. Expected columns: original_text, transcribed_text
                (optional), prompt (optional), plus score and reason columns.
        score_columns: Metric score columns to display for each sample.
    """
    for i, (_, row) in enumerate(sample.iterrows(), 1):
        print("-" * 70)
        prompt_label = f"  (prompt: {row['prompt']})" if "prompt" in row.index else ""
        print(f"Sample {i}{prompt_label}")

        print(f"\n  Original:\n    {row['original_text']}\n")
        if "transcribed_text" in row.index:
            print(f"  Simplified:\n    {row['transcribed_text']}\n")

        print("  Metrics:")
        for metric in score_columns:
            if metric not in row.index:
                continue
            reason_col = metric.replace("_score", "_reason")
            reason = str(row[reason_col]) if reason_col in row.index else "—"
            print(f"    {metric:<40} {row[metric]}")
            print(f"      reason: {reason}")
        print()


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    DATA_PATH = "../data/cleaned.csv"

    # TODO: get the data for a random subset of items for each loop

    # Step 1: Load and clean data ────────────────────────────────────────────────
    prompt_names = PROMPTS.keys()
    metrics = EVAL_DIMENSIONS.keys()
    df = pd.read_csv(DATA_PATH)

    df = normalise_fk_grade(df)
    score_columns = get_score_columns(df)

    # TODO: have the weights be changable in the metrics loop somehow.

    # Increase weight on readability metrics — they matter most for the end user.
    weights = {
        "fk_normalized_score": 5,
        "accessible_and_simple_score": 5,
    }
    df = compute_weighted_average(df, score_columns, weights)

    # Step 3: Inspect metrics ─────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 1: METRIC VARIANCE CHECK")
    print("=" * 70 + "\n")
    check_metric_variance(df, score_columns)

    # TODO: check if the user wants to delete the identified metrics, or change some others.
    input("continue >")

    print("\n" + "=" * 70)
    print("STEP 2: SANITY CHECK — RANDOM SAMPLES")
    print("=" * 70 + "\n")
    show_sanity_samples(df, score_columns)

    # TODO: Check if the user wants to change or remove some metrics
    input("continue >")

    # TODO: Only continue if the user didn't change metrics

    # Step 4: Compare prompts ─────────────────────────────────────────────────

    # TODO: Before evaluation maybe have the user input different prompts?
    print("\n" + "=" * 70)
    print("STEP 3: PROMPT COMPARISON")
    print("=" * 70 + "\n")
    prompt_stats = compare_prompts(df)

    # TODO: Let the user change prompts or metrics
    input("continue >")

    # Step 5: Verify best prompt ──────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("STEP 4: VERIFY BEST PROMPT")
    print("=" * 70 + "\n")
    verify_best_prompt(df, prompt_stats, score_columns)

    # TODO: let the user change prompts or metrics.
    input("continue >")

    # TODO: if prompts have been changed, let the user select the prompts to use for comparison.
