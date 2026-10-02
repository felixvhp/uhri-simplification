"""Tests for the feedback form's review axes.

Run from the repo root:
    python -m pytest tests/test_feedback_form.py
"""

from unittest.mock import patch

import pytest

from src.feedback_form.build import GROUP_ACCESS, GROUP_FIDELITY, build_axes

METRICS = {
    "actors_exact": {"name": "Actors exact", "description": " Same actors? ", "active": True},
    "plain_vocabulary": {"name": "Plain vocabulary", "description": "Plain words?", "active": True,
                         "group": "accessibility"},
    "old_one": {"name": "Old", "description": "Retired question", "active": False},
}


def test_build_axes_uses_active_metrics_with_names_and_groups():
    with patch("src.feedback_form.build.load_metrics", return_value=METRICS):
        axes = build_axes()

    assert axes == [
        {"key": "actors_exact", "t": "Actors exact", "d": "Same actors?", "group": GROUP_FIDELITY},
        {"key": "plain_vocabulary", "t": "Plain vocabulary", "d": "Plain words?", "group": GROUP_ACCESS},
    ]


def test_build_axes_exits_when_nothing_is_active():
    with patch("src.feedback_form.build.load_metrics", return_value={"old_one": METRICS["old_one"]}):
        with pytest.raises(SystemExit):
            build_axes()


def test_build_axes_works_on_the_real_settings():
    assert build_axes()
