"""Tests for the feedback form's review axes.

Run from the repo root:
    python -m pytest tests/test_feedback_form.py
"""

from unittest.mock import patch

import pytest

from src.feedback_form.build import GROUP_ACCESS, GROUP_FIDELITY, build_axes, build_form_config

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


PROMPTS = {"a": {"name": "Draft A", "template": "x"}, "b": {"name": "Draft B", "template": "y"}}


def test_build_form_config_follows_selected_prompts_and_prompt_names():
    with patch("src.feedback_form.build.load_prompts", return_value=PROMPTS), \
         patch("src.feedback_form.build.load_settings", return_value={"selected_prompts": ["b", "a"]}):
        assert build_form_config() == {"primary": "b", "labels": {"a": "Draft A", "b": "Draft B"}}


def test_build_form_config_without_a_selection():
    with patch("src.feedback_form.build.load_prompts", return_value=PROMPTS), \
         patch("src.feedback_form.build.load_settings", return_value={}):
        assert build_form_config()["primary"] == ""


def test_build_form_config_on_the_real_settings_points_at_a_defined_prompt():
    cfg = build_form_config()
    assert cfg["primary"] in cfg["labels"]
