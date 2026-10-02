"""Tests for workflow.py — pure-function coverage only.

Run from the repo root:
    python -m pytest tests/test_workflow.py
"""

import pandas as pd
import pytest
from src.dashboard.workflow import (
    get_score_columns,
    normalise_fk_grade,
    compute_weighted_average,
    compare_prompts,
    get_high_variance_samples,
    check_metric_covariance,
)


# ── Helpers ───────────────────────────────────────────────────────────────────


def _score_df(*rows) -> pd.DataFrame:
    """Build a minimal DataFrame with two score columns from (a, b) tuples."""
    return pd.DataFrame(rows, columns=["a_score", "b_score"])


def _prompt_df(a_scores, b_scores) -> pd.DataFrame:
    """Build a DataFrame with prompt and average_score columns.

    Includes dummy original_text/transcribed_text columns: verbose
    compare_prompts prints samples for high-variance prompts, which
    needs them present.
    """
    rows = (
        [{"prompt": "A", "average_score": s, "original_text": "orig", "transcribed_text": "simp"} for s in a_scores]
        + [{"prompt": "B", "average_score": s, "original_text": "orig", "transcribed_text": "simp"} for s in b_scores]
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


def test_normalise_fk_grade_no_spread_is_zero_not_nan():
    # Every delta equal -> nothing to normalise against. The naive min-max
    # formula divides by zero here; must come back as 0.0, not NaN/inf.
    df = pd.DataFrame({"fk_grade_delta": [3.0, 3.0, 3.0]})
    result = normalise_fk_grade(df)["fk_normalized_score"]
    assert (result == 0.0).all()


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


def test_nan_metric_is_excluded_from_that_rows_average_not_treated_as_zero():
    # A row missing one metric (e.g. a custom metric never scored for a
    # prompt added later) should average over what it does have, not have
    # the missing metric's weight silently deflate the result.
    df = pd.DataFrame({"a_score": [1.0, 1.0], "b_score": [1.0, float("nan")]})
    result = compute_weighted_average(df, ["a_score", "b_score"])
    assert result["average_score"].iloc[0] == pytest.approx(1.0)  # (1+1)/2
    assert result["average_score"].iloc[1] == pytest.approx(1.0)  # 1/1, not (1+0)/2=0.5


def test_all_metrics_nan_gives_nan_average():
    df = pd.DataFrame({"a_score": [float("nan")], "b_score": [float("nan")]})
    result = compute_weighted_average(df, ["a_score", "b_score"])
    assert pd.isna(result["average_score"].iloc[0])


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


def test_compare_prompts_single_sample_group_variance_is_zero_not_nan():
    # A prompt evaluated on only one text has an undefined sample variance
    # (pandas .var() gives NaN for n=1) — this must come back as 0.0, not NaN,
    # so it doesn't trip high_variance or poison downstream comparisons.
    df = _prompt_df([0.9], [0.5, 0.6])
    stats = compare_prompts(df)
    assert stats.loc["A", "variance"] == pytest.approx(0.0)
    assert stats.loc["A", "high_variance"] == False  # noqa: E712


def test_compare_prompts_flags_high_variance_prompts():
    df = _prompt_df([0.5, 0.5], [0.0, 1.0])  # B: variance 0.5, well above default threshold
    stats = compare_prompts(df)
    assert stats.loc["A", "high_variance"] == False  # noqa: E712
    assert stats.loc["B", "high_variance"] == True  # noqa: E712


def test_compare_prompts_high_var_threshold_is_tunable():
    df = _prompt_df([0.5, 0.5], [0.4, 0.6])  # B: variance 0.02
    loose = compare_prompts(df, high_var_threshold=0.5, verbose=False)
    strict = compare_prompts(df, high_var_threshold=0.01, verbose=False)
    assert loose.loc["B", "high_variance"] == False  # noqa: E712
    assert strict.loc["B", "high_variance"] == True  # noqa: E712


def test_compare_prompts_verbose_false_prints_nothing(capsys):
    df = _prompt_df([0.5, 0.5], [0.0, 1.0])
    compare_prompts(df, verbose=False)
    assert capsys.readouterr().out == ""


def test_compare_prompts_verbose_true_prints_report_and_samples(capsys):
    df = pd.DataFrame(
        {
            "prompt": ["A", "A", "B", "B"],
            "average_score": [0.5, 0.5, 0.0, 1.0],
            "original_text": ["orig1", "orig2", "orig3", "orig4"],
            "transcribed_text": ["t1", "t2", "t3", "t4"],
        }
    )
    compare_prompts(df, score_columns=[])
    out = capsys.readouterr().out
    assert "Prompt comparison" in out
    assert "HIGH VARIANCE" in out
    assert "'B' has high variance" in out


# ── get_high_variance_samples ─────────────────────────────────────────────────


def test_get_high_variance_samples_only_returns_flagged_prompts():
    df = pd.DataFrame(
        {
            "prompt": ["A", "A", "B", "B"],
            "average_score": [0.5, 0.5, 0.0, 1.0],
            "original_text": ["orig1", "orig2", "orig3", "orig4"],
            "transcribed_text": ["t1", "t2", "t3", "t4"],
        }
    )
    stats = compare_prompts(df, verbose=False)
    samples = get_high_variance_samples(df, stats, score_columns=[])
    assert set(samples.keys()) == {"B"}
    assert (samples["B"]["prompt"] == "B").all()


def test_get_high_variance_samples_empty_when_none_flagged():
    df = _prompt_df([0.5, 0.5], [0.5, 0.5])
    stats = compare_prompts(df, verbose=False)
    df_full = df.assign(original_text="x", transcribed_text="y")
    samples = get_high_variance_samples(df_full, stats, score_columns=[])
    assert samples == {}


# ── check_metric_covariance ───────────────────────────────────────────────────


def test_check_metric_covariance_flags_perfectly_correlated_pair():
    df = _score_df((1, 1), (0, 0), (1, 1), (0, 0))
    result = check_metric_covariance(df, ["a_score", "b_score"], threshold=0.8, verbose=False)
    assert len(result["high_covariance_pairs"]) == 1
    metric_a, metric_b, corr = result["high_covariance_pairs"][0]
    assert {metric_a, metric_b} == {"a_score", "b_score"}
    assert corr == pytest.approx(1.0)


def test_check_metric_covariance_ignores_unrelated_metrics():
    df = _score_df((1, 0), (0, 1), (1, 1), (0, 0))
    result = check_metric_covariance(df, ["a_score", "b_score"], threshold=0.8, verbose=False)
    assert result["high_covariance_pairs"] == []


def test_check_metric_covariance_threshold_is_tunable():
    df = _score_df((1, 1), (0, 0), (1, 0), (0, 1))  # corr = 0.0
    assert check_metric_covariance(df, ["a_score", "b_score"], threshold=0.01, verbose=False)["high_covariance_pairs"] == []


def test_check_metric_covariance_diagonal_is_nan_not_self_flagged():
    df = _score_df((1, 1), (0, 0), (1, 1), (0, 0))
    result = check_metric_covariance(df, ["a_score", "b_score"], threshold=0.8, verbose=False)
    assert pd.isna(result["correlations"].loc["a_score", "a_score"])
    assert all(a != b for a, b, _ in result["high_covariance_pairs"])


def test_check_metric_covariance_sorts_by_absolute_correlation_descending():
    df = pd.DataFrame({
        "a_score": [1, 0, 1, 0, 1, 0],
        "b_score": [1, 0, 1, 0, 0, 1],  # perfectly correlated with a
        "c_score": [1, 1, 0, 0, 1, 0],  # weaker correlation with a
    })
    result = check_metric_covariance(df, ["a_score", "b_score", "c_score"], threshold=0.0, verbose=False)
    pairs = result["high_covariance_pairs"]
    assert abs(pairs[0][2]) >= abs(pairs[-1][2])


def test_check_metric_covariance_verbose_false_prints_nothing(capsys):
    df = _score_df((1, 1), (0, 0))
    check_metric_covariance(df, ["a_score", "b_score"], verbose=False)
    assert capsys.readouterr().out == ""


def test_check_metric_covariance_verbose_true_prints_report(capsys):
    df = _score_df((1, 1), (0, 0), (1, 1), (0, 0))
    check_metric_covariance(df, ["a_score", "b_score"], threshold=0.8, verbose=True)
    out = capsys.readouterr().out
    assert "highly correlated" in out


if __name__ == "__main__":
    tests = [
        test_get_score_columns_picks_up_score_cols,
        test_get_score_columns_ignores_non_score_cols,
        test_get_score_columns_empty_df,
        test_normalise_fk_grade_min_is_zero,
        test_normalise_fk_grade_max_is_one,
        test_normalise_fk_grade_midpoint,
        test_normalise_fk_grade_no_spread_is_zero_not_nan,
        test_equal_weights_is_arithmetic_mean,
        test_none_weights_defaults_to_equal,
        test_custom_weights_change_the_average,
        test_weight_key_not_in_dict_defaults_to_one,
        test_extreme_weight_dominates,
        test_all_zeros_give_zero_average,
        test_all_ones_give_one_average_regardless_of_weights,
        test_multiple_rows_are_computed_independently,
        test_single_score_column_returns_that_value,
        test_nan_metric_is_excluded_from_that_rows_average_not_treated_as_zero,
        test_all_metrics_nan_gives_nan_average,
        test_compare_prompts_ranks_by_mean,
        test_compare_prompts_mean_values_are_correct,
        test_compare_prompts_returns_variance,
        test_compare_prompts_single_sample_group_variance_is_zero_not_nan,
        test_compare_prompts_flags_high_variance_prompts,
        test_compare_prompts_high_var_threshold_is_tunable,
        test_compare_prompts_verbose_false_prints_nothing,
        test_compare_prompts_verbose_true_prints_report_and_samples,
        test_get_high_variance_samples_only_returns_flagged_prompts,
        test_get_high_variance_samples_empty_when_none_flagged,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
