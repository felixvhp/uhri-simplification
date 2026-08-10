"""Tests for prompt_improvement_workflow.py — pure-function coverage only.

Run from code/:
    python -m pytest tests/test_prompt_improvement_workflow.py
"""

import pandas as pd
import pytest
from src.prompt_improvement_workflow import (
    get_score_columns,
    normalise_fk_grade,
    compute_weighted_average,
    compare_prompts,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _score_df(*rows) -> pd.DataFrame:
    """Build a minimal DataFrame with two score columns from (a, b) tuples."""
    return pd.DataFrame(rows, columns=["a_score", "b_score"])


def _prompt_df(a_scores, b_scores) -> pd.DataFrame:
    """Build a DataFrame with prompt and average_score columns."""
    rows = (
        [{"prompt": "A", "average_score": s} for s in a_scores]
        + [{"prompt": "B", "average_score": s} for s in b_scores]
    )
    return pd.DataFrame(rows)


# ── get_score_columns ─────────────────────────────────────────────────────────


def test_get_score_columns_picks_up_score_cols():
    df = pd.DataFrame({"a_score": [1], "a_reason": ["ok"], "b_score": [0], "other": [3]})
    assert set(get_score_columns(df)) == {"a_score", "b_score"}


def test_get_score_columns_ignores_non_score_cols():
    # "score_prefix" starts with score but doesn't end with _score; "unrelated" has no score at all
    df = pd.DataFrame({"score_prefix": [1], "unrelated": [2]})
    assert get_score_columns(df) == []


def test_get_score_columns_empty_df():
    df = pd.DataFrame({"name": ["x"]})
    assert get_score_columns(df) == []


# ── normalise_fk_grade ────────────────────────────────────────────────────────


def test_normalise_fk_grade_min_is_zero():
    df = pd.DataFrame({"fk_grade_delta": [2.0, 4.0, 6.0]})
    assert normalise_fk_grade(df)["fk_normalized_score"].min() == pytest.approx(0.0)


def test_normalise_fk_grade_max_is_one():
    df = pd.DataFrame({"fk_grade_delta": [2.0, 4.0, 6.0]})
    assert normalise_fk_grade(df)["fk_normalized_score"].max() == pytest.approx(1.0)


def test_normalise_fk_grade_midpoint():
    df = pd.DataFrame({"fk_grade_delta": [0.0, 5.0, 10.0]})
    assert normalise_fk_grade(df)["fk_normalized_score"].iloc[1] == pytest.approx(0.5)


# ── compute_weighted_average ──────────────────────────────────────────────────


def test_equal_weights_is_arithmetic_mean():
    df = _score_df((0.8, 0.4))
    result = compute_weighted_average(df, ["a_score", "b_score"])
    assert result["average_score"].iloc[0] == pytest.approx(0.6)


def test_none_weights_defaults_to_equal():
    df = _score_df((1.0, 0.0))
    result = compute_weighted_average(df, ["a_score", "b_score"], weights=None)
    assert result["average_score"].iloc[0] == pytest.approx(0.5)


def test_custom_weights_change_the_average():
    # (0.8 * 3 + 0.0 * 1) / (3 + 1) = 0.6
    df = _score_df((0.8, 0.0))
    result = compute_weighted_average(df, ["a_score", "b_score"], {"a_score": 3.0, "b_score": 1.0})
    assert result["average_score"].iloc[0] == pytest.approx(0.6)


def test_weight_key_not_in_dict_defaults_to_one():
    # b_score not in weights → treated as 1.0, so plain mean of 0.0 and 1.0 = 0.5
    df = _score_df((0.0, 1.0))
    result = compute_weighted_average(df, ["a_score", "b_score"], {"a_score": 1.0})
    assert result["average_score"].iloc[0] == pytest.approx(0.5)


def test_extreme_weight_dominates():
    # b_score weight = 1000 >> a_score weight = 1, so average ≈ b_score value
    df = _score_df((0.0, 1.0))
    result = compute_weighted_average(df, ["a_score", "b_score"], {"a_score": 1.0, "b_score": 1000.0})
    assert result["average_score"].iloc[0] == pytest.approx(1000.0 / 1001.0, rel=1e-3)


def test_all_zeros_give_zero_average():
    df = _score_df((0.0, 0.0))
    result = compute_weighted_average(df, ["a_score", "b_score"])
    assert result["average_score"].iloc[0] == pytest.approx(0.0)


def test_all_ones_give_one_average_regardless_of_weights():
    df = _score_df((1.0, 1.0))
    result = compute_weighted_average(df, ["a_score", "b_score"], {"a_score": 10.0, "b_score": 1.0})
    assert result["average_score"].iloc[0] == pytest.approx(1.0)


def test_multiple_rows_are_computed_independently():
    df = _score_df((1.0, 0.0), (0.0, 1.0))
    result = compute_weighted_average(df, ["a_score", "b_score"])
    assert result["average_score"].iloc[0] == pytest.approx(0.5)
    assert result["average_score"].iloc[1] == pytest.approx(0.5)


def test_single_score_column_returns_that_value():
    df = pd.DataFrame({"a_score": [0.7]})
    result = compute_weighted_average(df, ["a_score"])
    assert result["average_score"].iloc[0] == pytest.approx(0.7)


# ── compare_prompts ───────────────────────────────────────────────────────────


def test_compare_prompts_ranks_by_mean(capsys):
    df = _prompt_df([0.9, 0.8], [0.5, 0.6])  # A mean=0.85, B mean=0.55
    stats = compare_prompts(df)
    assert stats.index[0] == "A"
    assert stats.index[1] == "B"


def test_compare_prompts_mean_values_are_correct(capsys):
    df = _prompt_df([1.0, 0.0], [0.4, 0.6])
    stats = compare_prompts(df)
    assert stats.loc["A", "mean_score"] == pytest.approx(0.5)
    assert stats.loc["B", "mean_score"] == pytest.approx(0.5)


def test_compare_prompts_returns_variance(capsys):
    # Prompt A has variance 0 (constant scores), B has some variance
    df = _prompt_df([0.5, 0.5], [0.0, 1.0])
    stats = compare_prompts(df)
    assert stats.loc["A", "variance"] == pytest.approx(0.0)
    assert stats.loc["B", "variance"] > 0


if __name__ == "__main__":
    tests = [
        test_get_score_columns_picks_up_score_cols,
        test_get_score_columns_ignores_non_score_cols,
        test_get_score_columns_empty_df,
        test_normalise_fk_grade_min_is_zero,
        test_normalise_fk_grade_max_is_one,
        test_normalise_fk_grade_midpoint,
        test_equal_weights_is_arithmetic_mean,
        test_none_weights_defaults_to_equal,
        test_custom_weights_change_the_average,
        test_weight_key_not_in_dict_defaults_to_one,
        test_extreme_weight_dominates,
        test_all_zeros_give_zero_average,
        test_all_ones_give_one_average_regardless_of_weights,
        test_multiple_rows_are_computed_independently,
        test_single_score_column_returns_that_value,
        test_compare_prompts_ranks_by_mean,
        test_compare_prompts_mean_values_are_correct,
        test_compare_prompts_returns_variance,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
