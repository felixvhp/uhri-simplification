"""Tests for the human-vs-judge calibration module."""
import json
import math

import pandas as pd

from src.feedback_form.calibration import (
    calibrate,
    cohen_kappa,
    join_pairs,
    judge_column,
    load_review_records,
)


def _judge_df():
    """Two records x one prompt, two dimensions, double-underscore columns."""
    return pd.DataFrame([
        {
            "original_text": "States must act.", "prompt": "prompt_felix",
            "transcribed_text": "Countries must act.",
            "deontic_force__score": 1, "deontic_force__reason": "kept must",
            "plain_vocabulary__score": 0, "plain_vocabulary__reason": "jargon kept",
        },
        {
            "original_text": "The Committee urges reform.", "prompt": "prompt_felix",
            "transcribed_text": "The Committee wants reform.",
            "deontic_force__score": 0, "deontic_force__reason": "urges softened",
            "plain_vocabulary__score": 1, "plain_vocabulary__reason": "plain",
        },
    ])


def _review(original, **human):
    rec = {"reviewed_variant": "prompt_felix", "original": original, "verdict": "good"}
    rec.update(human)
    return rec


def test_judge_column_prefers_double_underscore_then_legacy():
    df = pd.DataFrame(columns=["a__score", "b_score"])
    assert judge_column(df, "a") == "a__score"
    assert judge_column(df, "b") == "b_score"
    assert judge_column(df, "c") is None


def test_join_pairs_matches_on_original_and_variant():
    pairs, unmatched = join_pairs(
        [_review("States must act.", human_deontic_force=1, human_plain_vocabulary=1)],
        _judge_df(),
    )
    assert unmatched == 0
    assert {(p["dim"], p["human"], p["judge"]) for p in pairs} == {
        ("deontic_force", 1, 1), ("plain_vocabulary", 1, 0),
    }


def test_join_pairs_skips_blank_answers_and_counts_unmatched():
    pairs, unmatched = join_pairs(
        [
            _review("States must act.", human_deontic_force="", human_plain_vocabulary=0),
            _review("Not in the CSV.", human_deontic_force=1),
        ],
        _judge_df(),
    )
    assert unmatched == 1
    assert [(p["dim"], p["human"]) for p in pairs] == [("plain_vocabulary", 0)]


def test_cohen_kappa_perfect_and_undefined():
    assert cohen_kappa([(1, 1), (0, 0), (1, 1), (0, 0)]) == 1.0
    assert math.isnan(cohen_kappa([(1, 1), (1, 1)]))  # both raters constant
    assert math.isnan(cohen_kappa([]))


def test_cohen_kappa_known_value():
    # 3 agreements / 4; marginals: human 2/4 pass, judge 3/4 pass -> pe = 0.5
    pairs = [(1, 1), (1, 1), (0, 0), (0, 1)]
    assert abs(cohen_kappa(pairs) - 0.5) < 1e-9


def test_calibrate_reports_per_dim_and_disagreements():
    reviews = [
        _review("States must act.", human_deontic_force=1, human_plain_vocabulary=1),
        _review("The Committee urges reform.", human_deontic_force=0, human_plain_vocabulary=1),
    ]
    result = calibrate(reviews, _judge_df())
    dims = result["per_dim"]
    assert dims["deontic_force"]["agreement"] == 1.0
    assert dims["plain_vocabulary"]["agreement"] == 0.5
    assert result["unmatched"] == 0
    assert [(d["dim"], d["judge_reason"]) for d in result["disagreements"]] == [
        ("plain_vocabulary", "jargon kept"),
    ]


def test_calibrate_handles_legacy_single_underscore_columns():
    df = _judge_df().rename(columns={
        "deontic_force__score": "deontic_force_score",
        "deontic_force__reason": "deontic_force_reason",
    })
    result = calibrate([_review("States must act.", human_deontic_force=1)], df)
    assert result["per_dim"]["deontic_force"]["n"] == 1


def test_load_review_records_reads_files_and_dirs(tmp_path):
    (tmp_path / "a.json").write_text(json.dumps({"records": [{"x": 1}]}))
    sub = tmp_path / "exports"
    sub.mkdir()
    (sub / "b.json").write_text(json.dumps({"records": [{"x": 2}, {"x": 3}]}))
    assert len(load_review_records([tmp_path / "a.json", sub])) == 3
