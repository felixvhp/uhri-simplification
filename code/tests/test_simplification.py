"""Tests for simplification.py — simplify_text() and generate_new_prompt().

Both the OpenAI client (simplification) and Anthropic client (evaluation's
judge, called internally by generate_new_prompt) are mocked — no real API
calls happen in this test file.

Run from code/:
    python -m pytest tests/test_simplification.py
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.simplification import (
    _MissingContext,
    generate_new_prompt,
    save_custom_prompt_rows_to_csv,
    simplify_text,
)

JUDGE_RESPONSE = """{
    "content_preserved":     {"score": true,  "reason": "ok"},
    "actors_preserved":      {"score": true,  "reason": "ok"},
    "no_meaning_change":     {"score": true,  "reason": "ok"},
    "no_hallucination":      {"score": true,  "reason": "ok"},
    "accessible_and_simple": {"score": true,  "reason": "ok"}
}"""


def _mock_openai_client(output_text):
    mock_response = MagicMock()
    mock_response.output_text = output_text
    mock_client = MagicMock()
    mock_client.responses.create.return_value = mock_response
    return mock_client


def _mock_anthropic_client(judge_response_text):
    mock_msg = MagicMock()
    mock_msg.content[0].text = judge_response_text
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


# ── _MissingContext ───────────────────────────────────────────────────────────


def test_missing_context_fills_known_key():
    assert "{text}".format_map(_MissingContext(text="hello")) == "hello"


def test_missing_context_fills_unknown_key_with_placeholder_note():
    assert _MissingContext(text="hello")["country"] == "(not available for this prompt)"


# ── simplify_text ─────────────────────────────────────────────────────────────


def test_simplify_text_fills_text_placeholder():
    client = _mock_openai_client("simplified!")
    with patch("src.simplification.openai_client", client):
        result = simplify_text("original text", "Simplify: {text}")
    assert result == "simplified!"
    sent_prompt = client.responses.create.call_args.kwargs["input"]
    assert "original text" in sent_prompt


def test_simplify_text_handles_placeholders_a_csv_row_cant_fill():
    client = _mock_openai_client("ok")
    with patch("src.simplification.openai_client", client):
        # {country} isn't available from a long-format row (no UHRIRecord) —
        # must not raise, and should say so rather than silently dropping it.
        simplify_text("txt", "Simplify for {country}: {text}")
    sent_prompt = client.responses.create.call_args.kwargs["input"]
    assert "(not available for this prompt)" in sent_prompt


# ── generate_new_prompt ───────────────────────────────────────────────────────


def test_generate_new_prompt_adds_one_row_per_unique_text_not_per_existing_row():
    # "text A" appears twice (under two existing prompts) — should only be
    # simplified once for the new prompt, not once per existing row.
    df = pd.DataFrame(
        {
            "original_text": ["text A", "text A", "text B"],
            "prompt": ["existing_1", "existing_2", "existing_1"],
            "transcribed_text": ["a1", "a2", "b1"],
            "fk_grade_delta": [1.0, 2.0, 3.0],
        }
    )
    openai_client = _mock_openai_client("simplified output")
    with patch("src.simplification.openai_client", openai_client), \
         patch("src.evaluation.eval_client", _mock_anthropic_client(JUDGE_RESPONSE)):
        result = generate_new_prompt(df, "My New Prompt", "Simplify: {text}")

    new_rows = result[result["prompt"] == "my_new_prompt"]
    assert len(new_rows) == 2
    assert set(new_rows["original_text"]) == {"text A", "text B"}
    assert (new_rows["transcribed_text"] == "simplified output").all()
    assert openai_client.responses.create.call_count == 2
    assert len(result) == len(df) + 2  # original rows preserved, not replaced


def test_generate_new_prompt_recomputes_fk_normalized_score_across_combined_rows():
    df = pd.DataFrame(
        {
            "original_text": ["text A"],
            "prompt": ["existing_1"],
            "transcribed_text": ["a1"],
            "fk_grade_delta": [1.0],
        }
    )
    with patch("src.simplification.openai_client", _mock_openai_client("out")), \
         patch("src.evaluation.eval_client", _mock_anthropic_client(JUDGE_RESPONSE)):
        result = generate_new_prompt(df, "New", "Simplify: {text}")

    assert "fk_normalized_score" in result.columns
    assert result["fk_normalized_score"].notna().all()


def test_generate_new_prompt_reports_progress_once_per_unique_text():
    df = pd.DataFrame(
        {
            "original_text": ["a", "a", "b"],
            "prompt": ["p", "p", "p"],
            "transcribed_text": ["x", "y", "z"],
            "fk_grade_delta": [1.0, 1.0, 1.0],
        }
    )
    progress_calls = []
    with patch("src.simplification.openai_client", _mock_openai_client("out")), \
         patch("src.evaluation.eval_client", _mock_anthropic_client(JUDGE_RESPONSE)):
        generate_new_prompt(
            df, "New", "{text}",
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )
    assert progress_calls == [(0, 2), (1, 2), (2, 2)]  # 2 unique texts, not 3 rows


# ── save_custom_prompt_rows_to_csv ────────────────────────────────────────────


def test_save_custom_prompt_rows_to_csv_appends_rows(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame(
        {"original_text": ["a", "b"], "prompt": ["p1", "p2"], "transcribed_text": ["x", "y"]}
    ).to_csv(csv_path, index=False)

    new_rows = [
        {"original_text": "a", "prompt": "bullet_points", "transcribed_text": "simplified a"},
        {"original_text": "b", "prompt": "bullet_points", "transcribed_text": "simplified b"},
    ]
    save_custom_prompt_rows_to_csv(str(csv_path), new_rows)

    result = pd.read_csv(csv_path)
    assert len(result) == 4
    assert set(result["prompt"]) == {"p1", "p2", "bullet_points"}


def test_save_custom_prompt_rows_to_csv_rejects_duplicate_prompt(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame(
        {"original_text": ["a"], "prompt": ["bullet_points"], "transcribed_text": ["x"]}
    ).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        save_custom_prompt_rows_to_csv(
            str(csv_path), [{"original_text": "a", "prompt": "bullet_points", "transcribed_text": "y"}]
        )

    # Nothing should have been appended on the rejected call.
    result = pd.read_csv(csv_path)
    assert len(result) == 1


if __name__ == "__main__":
    tests = [
        test_missing_context_fills_known_key,
        test_missing_context_fills_unknown_key_with_placeholder_note,
        test_simplify_text_fills_text_placeholder,
        test_simplify_text_handles_placeholders_a_csv_row_cant_fill,
        test_generate_new_prompt_adds_one_row_per_unique_text_not_per_existing_row,
        test_generate_new_prompt_recomputes_fk_normalized_score_across_combined_rows,
        test_generate_new_prompt_reports_progress_once_per_unique_text,
        # These two need pytest's tmp_path fixture — see test_prompts.py's
        # __main__ block for why they'll show FAIL called bare like this.
        test_save_custom_prompt_rows_to_csv_appends_rows,
        test_save_custom_prompt_rows_to_csv_rejects_duplicate_prompt,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
