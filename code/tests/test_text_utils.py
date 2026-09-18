"""Tests for text_utils.py.

Run from code/:
    python -m pytest tests/test_text_utils.py
"""

import pytest

from src.text_utils import slugify, format_legal_bullets


def test_slugify_lowercases_and_joins_with_underscores():
    assert slugify("Tone: formal enough?") == "tone_formal_enough"


def test_slugify_collapses_repeated_separators():
    assert slugify("  extra   spaces -- here  ") == "extra_spaces_here"


def test_slugify_rejects_names_with_no_usable_characters():
    with pytest.raises(ValueError):
        slugify("???")


# ── format_legal_bullets ──────────────────────────────────────────────────────


def test_format_legal_bullets_splits_lettered_clauses_onto_own_lines():
    text = (
        "The Committee recommends that the State Party: "
        "(a) Do the first thing; (b) Do the second thing; (c) Do the third thing."
    )
    result = format_legal_bullets(text)
    lines = result.split("\n\n")
    assert lines[0] == "The Committee recommends that the State Party:"
    assert lines[1].startswith("(a) Do the first thing")
    assert lines[2].startswith("(b) Do the second thing")
    assert lines[3].startswith("(c) Do the third thing")


def test_format_legal_bullets_handles_single_newline_before_marker():
    # Matches the real data shape: a lone "\n" (not "\n\n") before "(a)",
    # which Markdown would otherwise render as just a space.
    text = "Intro sentence.\n(a) First clause.\n(b) Second clause."
    result = format_legal_bullets(text)
    assert result == "Intro sentence.\n\n(a) First clause.\n\n(b) Second clause."


def test_format_legal_bullets_leaves_text_with_no_markers_unchanged():
    text = "Just a plain sentence with no lettered clauses at all."
    assert format_legal_bullets(text) == text


def test_format_legal_bullets_leaves_a_single_marker_unchanged():
    # One "(a)" is more likely a stray parenthetical than a list start.
    text = "See the earlier finding (a) 2019 report for background."
    assert format_legal_bullets(text) == text


def test_format_legal_bullets_passes_through_non_string_input():
    assert format_legal_bullets(None) is None
    nan = float("nan")
    assert format_legal_bullets(nan) is nan


if __name__ == "__main__":
    tests = [
        test_slugify_lowercases_and_joins_with_underscores,
        test_slugify_collapses_repeated_separators,
        test_slugify_rejects_names_with_no_usable_characters,
        test_format_legal_bullets_splits_lettered_clauses_onto_own_lines,
        test_format_legal_bullets_handles_single_newline_before_marker,
        test_format_legal_bullets_leaves_text_with_no_markers_unchanged,
        test_format_legal_bullets_leaves_a_single_marker_unchanged,
        test_format_legal_bullets_passes_through_non_string_input,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
