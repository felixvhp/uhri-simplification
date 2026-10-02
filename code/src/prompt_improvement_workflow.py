"""
Pkg 3: Prompt improvement workflow [last updated 2026-09-13]

Usage:
    cd code
    python src/prompt_improvement_workflow.py

    The script reads data/cleaned.csv (the long-format CSV produced by
    unpivoting simplification_results.csv — see analyse_simplification.ipynb)
    and prints a four-step evaluation report to stdout.

    A Streamlit GUI over the same functions lives at src/prompt_improvement_app.py:
        streamlit run src/prompt_improvement_app.py

Reads a long-format CSV and runs a four-step evaluation workflow:
    1. Load and score: normalise FK-grade delta, compute weighted average.
    2. Metric variance check: flag low-variance metrics (poor discriminators).
    3. Prompt comparison: rank prompts by mean score, flag high-variance ones.
    4. Best-prompt verification: show sample outputs for manual inspection.

Design note — compute/display split:
    Every step function (check_metric_variance, show_sanity_samples,
    compare_prompts, verify_best_prompt) takes a ``verbose`` flag. Default
    True prints exactly like the original CLI. A caller that wants to render
    results itself (the Streamlit app, a notebook, a future dashboard) passes
    verbose=False and gets the same computation back as plain DataFrames/dicts
    instead of stdout text — so new front ends can be added without touching
    this module's logic.

Open Issues:
    - FK normalisation uses the dataset's own min/max, so the scale shifts
      when new data is added.
    - Weights are hand-tuned; no principled calibration yet.

Resumption:
    - Tweak WEIGHTS in __main__ (or the GUI sidebar), re-run, read the
      verify-best-prompt output.
    - Add new metrics to the CSV and they will be picked up automatically.
"""

import sys
from pathlib import Path

# When run as `python src/prompt_improvement_workflow.py` from code/,
# Python adds code/src/ to sys.path. We also need code/ so that src.*
# imports in this file and its dependencies (prompts.py, evaluation.py) resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict, List, Optional

import pandas as pd

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
        Every row gets 0.0 if every delta is equal (no spread to normalise
        against — the naive formula would otherwise divide by zero).
    """
    lo, hi = df["fk_grade_delta"].min(), df["fk_grade_delta"].max()
    span = hi - lo
    df["fk_normalized_score"] = (df["fk_grade_delta"] - lo) / span if span else 0.0
    return df


def compute_weighted_average(
    df: pd.DataFrame,
    score_columns: List[str],
    weights: Optional[Dict[str, float]] = None,
) -> pd.DataFrame:
    """Compute a weighted mean over score columns and store it as ``average_score``.

    NaN-safe per row: a row missing one metric (e.g. a custom metric that was
    never scored for a prompt added later — see generate_new_prompt() in
    simplification.py) has that metric's weight excluded from *its own* row's
    denominator too, not just its numerator. Averaging over the remaining
    metrics is what a reader expects from "missing data"; silently treating a
    missing score as 0 while still charging its weight would deflate that
    row's average for no real reason.

    Args:
        df: Long-format DataFrame with all ``score_columns`` present (any
            individual cell may be NaN).
        score_columns: Column names to include in the average.
        weights: Optional mapping of column name to weight. Columns not in the
                 dict default to 1.0. Pass ``None`` for equal weights.

    Returns:
        df with ``average_score`` added or updated (modifies in place and
        returns). NaN for a row where every metric is missing.
    """
    if weights is None:
        weights = {}
    w = pd.Series({col: weights.get(col, 1.0) for col in score_columns})
    weighted_sum = df[score_columns].mul(w).sum(axis=1)
    applicable_weight = df[score_columns].notna().mul(w).sum(axis=1)
    df["average_score"] = weighted_sum / applicable_weight
    return df


# ── Metric inspection ─────────────────────────────────────────────────────────


def check_metric_variance(
    df: pd.DataFrame,
    score_columns: List[str],
    threshold: float = LOW_VARIANCE_THRESHOLD,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
) -> Dict[str, object]:
    """Compute a variance report for each metric and sample the low-variance ones.

    A metric with low variance assigns nearly the same score to every output.
    It contributes little signal to the ranking and is a candidate for removal.
    For each flagged metric, sample rows are pulled so a caller can read the
    LLM's reasoning and decide whether the metric is worth keeping.

    Args:
        df: Long-format DataFrame with score and optional ``*_reason`` columns.
        score_columns: Metric score column names to evaluate.
        threshold: Variance below this triggers a LOW warning.
        n_samples: Number of sample rows to gather per flagged metric.
        verbose: If True (default), print the report as before. Set to False
            to compute silently, e.g. for a GUI that renders the result itself.

    Returns:
        Dict with keys ``variances`` (Series, ascending), ``low_variance_metrics``
        (list of flagged metric names), and ``samples`` (dict of metric name to
        sample DataFrame, one entry per flagged metric).
    """
    variances = df[score_columns].var().sort_values()
    low_var = variances[variances < threshold].index.tolist()

    samples = {}
    for metric in low_var:
        reason_col = metric.replace("_score", "_reason")
        sample_cols = ["original_text", "transcribed_text", "prompt", metric]
        if reason_col in df.columns:
            sample_cols.append(reason_col)
        samples[metric] = df[sample_cols].dropna(subset=[metric]).sample(min(n_samples, len(df)))

    if verbose:
        print("Metric variances (ascending):")
        for metric, var in variances.items():
            flag = "  <- LOW" if var < threshold else ""
            print(f"  {metric:<40} {var:.4f}{flag}")
        print()

        if not low_var:
            print("All metrics have sufficient variance — no candidates for removal.")

        for metric in low_var:
            var = variances[metric]
            print("=" * 70)
            print(f"WARNING: '{metric}' has low variance ({var:.4f})")
            print("  -> Nearly constant score across all outputs.")
            print("  -> If the samples below look reasonable, consider removing this metric.")
            print()
            _print_samples(samples[metric], [metric])

    return {"variances": variances, "low_variance_metrics": low_var, "samples": samples}


def show_sanity_samples(
    df: pd.DataFrame,
    score_columns: List[str],
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Sample rows for a full metric breakdown across all metrics at once.

    A quick sanity-check that scores and LLM reasons look sensible before
    comparing prompts.

    Args:
        df: Long-format DataFrame with score and optional ``*_reason`` columns.
        score_columns: Metric score columns to display per sample.
        n_samples: Number of rows to sample.
        verbose: If True (default), print the samples as before. Set to False
            to compute silently, e.g. for a GUI that renders the result itself.

    Returns:
        The sampled DataFrame (``n_samples`` rows).
    """
    reason_cols = [c for c in df.columns if c.endswith("_reason")]
    display_cols = ["original_text", "transcribed_text", "prompt"] + score_columns + reason_cols
    sample = df[display_cols].sample(n_samples)

    if verbose:
        print(f"=== Sanity check: {n_samples} random samples ===\n")
        _print_samples(sample, score_columns)

    return sample


# ── Prompt comparison ─────────────────────────────────────────────────────────


def get_high_variance_samples(
    df: pd.DataFrame,
    stats: pd.DataFrame,
    score_columns: Optional[List[str]] = None,
    n_samples: int = N_SAMPLES,
) -> Dict[str, pd.DataFrame]:
    """Pull sample rows for each prompt flagged ``high_variance`` in ``stats``.

    Mirrors the per-metric sample lookup in :func:`check_metric_variance`, but
    keyed by prompt instead of metric — lets a caller (CLI or GUI) show the
    LLM's reasoning for a prompt's inconsistent scores without recomputing
    anything already done by :func:`compare_prompts`.

    Args:
        df: Long-format DataFrame with ``prompt``, score, and optional
            ``*_reason`` columns.
        stats: DataFrame returned by :func:`compare_prompts` (must have a
            boolean ``high_variance`` column).
        score_columns: Metric score columns to include per sample. Defaults
            to :func:`get_score_columns` if not given.
        n_samples: Number of sample rows per flagged prompt.

    Returns:
        Mapping of prompt name to a sample DataFrame, one entry per prompt
        where ``stats.loc[prompt, "high_variance"]`` is True.
    """
    if score_columns is None:
        score_columns = get_score_columns(df)

    reason_cols = [c for c in df.columns if c.endswith("_reason")]
    display_cols = ["original_text", "transcribed_text", "prompt"] + score_columns + reason_cols

    flagged = stats[stats["high_variance"]]
    samples = {}
    for prompt in flagged.index:
        subset = df[df["prompt"] == prompt]
        samples[prompt] = subset[display_cols].sample(min(n_samples, len(subset)))
    return samples


def compare_prompts(
    df: pd.DataFrame,
    high_var_threshold: float = HIGH_VARIANCE_THRESHOLD,
    score_columns: Optional[List[str]] = None,
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
) -> pd.DataFrame:
    """Rank prompts by mean ``average_score`` and flag high-variance ones.

    High variance on a prompt means it performs well on some texts and poorly
    on others — worth investigating before trusting its mean ranking. A
    prompt evaluated on only a single text has an undefined sample variance
    (pandas' ``.var()`` returns NaN for n=1); that's treated as 0.0 here —
    "no evidence of inconsistency yet" — rather than left as NaN, so it never
    trips the high-variance flag or breaks numeric comparisons downstream.

    Args:
        df: Long-format DataFrame with ``prompt`` and ``average_score`` columns.
        high_var_threshold: Variance above this triggers a HIGH VARIANCE flag.
        score_columns: Metric score columns to show in flagged-prompt samples.
            Defaults to :func:`get_score_columns` if not given. Unused when
            ``verbose`` is False.
        n_samples: Number of sample rows to print per flagged prompt.
        verbose: If True (default), print the ranking table plus sample
            outputs for any flagged prompt, matching the CLI report. Set to
            False to compute silently, e.g. when a caller (such as a GUI)
            renders the returned DataFrame itself.

    Returns:
        DataFrame indexed by prompt with columns ``['mean_score', 'variance',
        'high_variance']``, sorted by ``mean_score`` descending.
    """
    stats = (
        df.groupby("prompt")["average_score"]
        .agg(["mean", "var"])
        .rename(columns={"mean": "mean_score", "var": "variance"})
        .sort_values("mean_score", ascending=False)
    )
    stats["variance"] = stats["variance"].fillna(0.0)
    stats["high_variance"] = stats["variance"] > high_var_threshold

    if not verbose:
        return stats

    print("Prompt comparison (ranked by mean score):")
    print(f"  {'Prompt':<30} {'Mean':>6}  {'Var':>7}")
    print(f"  {'-'*30} {'-'*6}  {'-'*7}")
    for prompt, row in stats.iterrows():
        flag = "  <- HIGH VARIANCE" if row["high_variance"] else ""
        print(f"  {prompt:<30} {row['mean_score']:>6.3f}  {row['variance']:>7.4f}{flag}")
    print()

    samples = get_high_variance_samples(df, stats, score_columns, n_samples)
    for prompt, sample in samples.items():
        print("=" * 70)
        print(f"WARNING: '{prompt}' has high variance ({stats.loc[prompt, 'variance']:.4f})")
        print("  -> Score swings a lot across texts — inspect outputs before trusting its rank.")
        print()
        _print_samples(sample, score_columns or get_score_columns(df))
    if not samples:
        print("All prompts have consistent scores — no candidates for further inspection.")
        print()

    return stats


def verify_best_prompt(
    df: pd.DataFrame,
    prompt_stats: pd.DataFrame,
    score_columns: List[str],
    n_samples: int = N_SAMPLES,
    verbose: bool = True,
) -> Dict[str, object]:
    """Sample outputs from the top-ranked prompt for manual review.

    Args:
        df: Long-format DataFrame with ``prompt``, score, and ``*_reason`` columns.
        prompt_stats: DataFrame returned by :func:`compare_prompts`.
        score_columns: Metric score columns to display per sample.
        n_samples: Number of sample rows to show.
        verbose: If True (default), print the samples as before. Set to False
            to compute silently, e.g. for a GUI that renders the result itself.

    Returns:
        Dict with keys ``best_prompt`` (name), ``mean_score``, and ``samples``
        (DataFrame of sample rows for that prompt).
    """
    best = prompt_stats.index[0]
    mean = prompt_stats.loc[best, "mean_score"]

    reason_cols = [c for c in df.columns if c.endswith("_reason")]
    subset = df[df["prompt"] == best]
    display_cols = ["original_text", "transcribed_text"] + score_columns + reason_cols
    sample = subset[display_cols].sample(min(n_samples, len(subset)))

    if verbose:
        print(f"Best prompt: '{best}'  (mean={mean:.3f})")
        print("Read the outputs below — does this look like good simplification?\n")
        _print_samples(sample, score_columns)

    return {"best_prompt": best, "mean_score": mean, "samples": sample}


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
    df = pd.read_csv(DATA_PATH)

    df = normalise_fk_grade(df)
    score_columns = get_score_columns(df)

    # TODO: have the weights be changable in the metrics loop somehow.

    # Increase weight on readability metrics — they matter most for the end user.
    weights = {
        "fk_normalized_score": 5.0,
        "accessible_and_simple_score": 5.0,
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
