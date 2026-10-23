"""Tests for the feedback form's review-data extraction.

Run from the repo root:
    python -m pytest tests/test_make_data.py
"""

from src.feedback_form.make_data import strip_markdown


def test_strip_markdown_cleans_model_headings():
    assert strip_markdown("**(a)** Build homes.\n# Heading\ntext") == "(a) Build homes.\nHeading\ntext"
    assert strip_markdown("  plain  ") == "plain"
