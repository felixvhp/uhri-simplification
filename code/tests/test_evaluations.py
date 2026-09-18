"""Tests for the evaluation module."""

import json
from datetime import date
from unittest.mock import patch, MagicMock

import pandas as pd

import pytest

from src.evaluation import (
    build_eval_prompt,
    readability,
    evaluate,
    evaluate_new_metric,
    load_all_metric_definitions,
    load_custom_metric_definitions,
    save_custom_metric_definition,
    save_custom_metric_scores_to_csv,
    EVAL_DIMENSIONS,
)


# --- Pure function tests ---

def test_build_eval_prompt_contains_inputs():
    orig = "States must protect children."
    simp = "Countries need to keep children safe."
    prompt = build_eval_prompt(orig, simp)
    assert orig in prompt
    assert simp in prompt


def test_build_eval_prompt_contains_all_dimensions():
    prompt = build_eval_prompt("a", "b")
    for dim in EVAL_DIMENSIONS:
        assert dim in prompt, f"Missing dimension: {dim}"


def test_readability_returns_expected_keys():
    result = readability("The cat sat on the mat.")
    assert "fk_grade" in result
    assert "fk_ease" in result
    assert isinstance(result["fk_grade"], float)


# --- LLM-mocked tests ---

MOCK_LLM_RESPONSE = """some random gibberish {
    "content_preserved":     {"score": true,  "reason": "ok"},
    "actors_preserved":      {"score": true,  "reason": "ok"},
    "no_meaning_change":     {"score": false, "reason": "changed must->should"},
    "no_hallucination":      {"score": true,  "reason": "ok"},
    "accessible_and_simple": {"score": true,  "reason": "ok"}
}"""


def _mock_client(response_text):
    mock_msg = MagicMock()
    mock_msg.content[0].text = response_text
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def test_evaluate_parses_valid_json():
    with patch("src.evaluation.eval_client", _mock_client(MOCK_LLM_RESPONSE)):
        result = evaluate("original", "simplified")
    assert result["content_preserved"]["score"] == 1
    assert result["no_meaning_change"]["score"] == 0


def test_evaluate_handles_bad_json():
    with patch("src.evaluation.eval_client", _mock_client("not json")):
        result = evaluate("original", "simplified")
    for dim in EVAL_DIMENSIONS:
        assert result[dim]["score"] == -1


def test_evaluate_handles_missing_dimension():
    # Response missing one dimension — that dim should get score -1
    partial = MOCK_LLM_RESPONSE.replace(
        '"no_hallucination":      {"score": true,  "reason": "ok"},\n', ""
    )
    with patch("src.evaluation.eval_client", _mock_client(partial)):
        result = evaluate("original", "simplified")
    assert result["no_hallucination"]["score"] == -1

def build_prompt_context():
    prompt_mock = """
    Original data:
    {original}
     
    Simplified text:
    {simplified}

    Evaluate on these dimensions:
    {explanations}

    {{
        {eval_dimensions}
    }}
    """
    dimensions_mock = {
            "dimension_one": "First dimension",
            "dimension_two": "Second dimension",
            "dimension_three": "Third dimension",
    }
    
    # Test the build_eval_prompt() using the mock prompt and dimensions.
    raw = {"Text": "Test original text.", 
            "Body": "Trans European Signature Treaty",
            "Country": "Switzerland",
            }
    simp = "Test simplified text."
    return prompt_mock, dimensions_mock, raw, simp

def test_eval_prompt_gets_built_correctly():
    prompt_mock, dimensions_mock, raw, simp = build_prompt_context()

    orig = str(raw)

    # Testing with a mocked prompt
    with patch("src.evaluation.EVAL_PROMPT", prompt_mock), \
         patch("src.evaluation.EVAL_DIMENSIONS", dimensions_mock):
        mocked_prompt = build_eval_prompt(orig, simp)

    print(mocked_prompt)

    prompt_built_test(mocked_prompt, orig, simp, dimensions_mock)


def test_eval_prompt_works_correctly():
    prompt_mock, dimensions_mock, orig, simp = build_prompt_context()

    orig = str(orig)

    # Testing with the real prompt
    with patch("src.evaluation.EVAL_DIMENSIONS", dimensions_mock):
        prompt = build_eval_prompt(orig, simp)

    prompt_built_test(prompt, orig, simp, dimensions_mock)

def test_eval_dims_work_correctly():
    prompt_mock, dimensions_mock, raw, simp = build_prompt_context()

    orig = str(raw)

    # Testing with the real prompt
    with patch("src.evaluation.EVAL_PROMPT", prompt_mock):
        prompt = build_eval_prompt(orig, simp)

    print(prompt)

    prompt_built_test(prompt, orig, simp, EVAL_DIMENSIONS)


# --- evaluate() with a custom `dimensions` override ---


def test_evaluate_with_custom_dimensions_scores_only_that_dimension():
    response = '{"tone_formal": {"score": true, "reason": "reads as formal"}}'
    with patch("src.evaluation.eval_client", _mock_client(response)):
        result = evaluate("original", "simplified", dimensions={"tone_formal": "Is it formal?"})
    assert list(result.keys()) == ["tone_formal"]
    assert result["tone_formal"]["score"] == 1


def test_build_eval_prompt_custom_dimensions_excludes_the_standard_five():
    prompt = build_eval_prompt("a", "b", dimensions={"tone_formal": "Is it formal?"})
    assert "tone_formal" in prompt
    for dim in EVAL_DIMENSIONS:
        assert dim not in prompt


# --- evaluate_new_metric ---
# (slugify() itself is tested in test_text_utils.py, not here)


def test_evaluate_new_metric_adds_score_and_reason_columns():
    df = pd.DataFrame(
        {
            "original_text": ["orig A", "orig B"],
            "transcribed_text": ["simp A", "simp B"],
        }
    )
    response = '{"tone_formal": {"score": true, "reason": "formal enough"}}'
    with patch("src.evaluation.eval_client", _mock_client(response)):
        result = evaluate_new_metric(df, "Tone: formal?", "Is it formal?")

    assert list(result["tone_formal_score"]) == [1, 1]
    assert list(result["tone_formal_reason"]) == ["formal enough", "formal enough"]


def test_evaluate_new_metric_reports_progress_per_row():
    df = pd.DataFrame({"original_text": ["a", "b", "c"], "transcribed_text": ["x", "y", "z"]})
    response = '{"m": {"score": false, "reason": "no"}}'
    progress_calls = []
    with patch("src.evaluation.eval_client", _mock_client(response)):
        evaluate_new_metric(df, "m", "d", on_progress=lambda done, total: progress_calls.append((done, total)))
    assert progress_calls == [(0, 3), (1, 3), (2, 3), (3, 3)]


# --- metric definition persistence (metrics.json: standard + custom, together) ---
# All of these pass an explicit tmp_path file, never the real METRICS_PATH/a
# real CSV, so they can't touch anything tracked.


def test_load_all_metric_definitions_empty_when_file_missing(tmp_path):
    assert load_all_metric_definitions(tmp_path / "does_not_exist.json") == {}


def test_load_custom_metric_definitions_empty_when_file_missing(tmp_path):
    assert load_custom_metric_definitions(tmp_path / "does_not_exist.json") == {}


def test_load_custom_metric_definitions_excludes_standard_entries(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "content_preserved": {"name": "content_preserved", "description": "d", "standard": True, "date_added": None},
        "tone_formal": {"name": "Tone", "description": "d2", "standard": False, "date_added": "2026-09-13"},
    }))
    custom_only = load_custom_metric_definitions(path)
    assert set(custom_only) == {"tone_formal"}
    assert set(load_all_metric_definitions(path)) == {"content_preserved", "tone_formal"}


def test_save_then_load_custom_metric_definition_roundtrips(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("tone_formal", "Tone: formal?", "Is it formal?", path, date_added="2026-09-13")
    assert load_custom_metric_definitions(path) == {
        "tone_formal": {
            "name": "Tone: formal?",
            "description": "Is it formal?",
            "standard": False,
            "date_added": "2026-09-13",
        }
    }


def test_save_custom_metric_definition_defaults_date_added_to_today(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("tone_formal", "Tone: formal?", "Is it formal?", path)
    saved = load_custom_metric_definitions(path)["tone_formal"]
    assert saved["date_added"] == date.today().isoformat()


def test_save_custom_metric_definition_merges_without_clobbering(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("a", "A", "desc a", path)
    save_custom_metric_definition("b", "B", "desc b", path)
    assert set(load_custom_metric_definitions(path)) == {"a", "b"}


def test_save_custom_metric_definition_overwrites_same_slug(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("a", "A", "first version", path)
    save_custom_metric_definition("a", "A", "second version", path)
    defs = load_custom_metric_definitions(path)
    assert len(defs) == 1
    assert defs["a"]["description"] == "second version"


def test_save_custom_metric_definition_preserves_standard_entries(tmp_path):
    # Regression guard: saving a new custom metric must read/write the
    # *whole* file, not just the custom subset, or it would silently drop
    # the standard 5 that live in the same metrics.json.
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "content_preserved": {"name": "content_preserved", "description": "d", "standard": True, "date_added": None},
    }))
    save_custom_metric_definition("tone_formal", "Tone", "d2", path, date_added="2026-09-18")
    all_defs = load_all_metric_definitions(path)
    assert "content_preserved" in all_defs and all_defs["content_preserved"]["standard"] is True
    assert "tone_formal" in all_defs and all_defs["tone_formal"]["standard"] is False


def test_eval_dimensions_excludes_custom_metrics():
    # EVAL_DIMENSIONS is derived from metrics.json at import time, filtered
    # to standard: true — it must never silently pick up a custom metric
    # saved to the real file, or every simplification run would start
    # scoring it too.
    assert set(EVAL_DIMENSIONS) <= set(load_all_metric_definitions())
    for slug in EVAL_DIMENSIONS:
        assert load_all_metric_definitions()[slug]["standard"] is True


def test_save_custom_metric_scores_to_csv_writes_new_columns(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({"original_text": ["a", "b"], "prompt": ["p1", "p2"]}).to_csv(csv_path, index=False)

    save_custom_metric_scores_to_csv(str(csv_path), "tone_formal", [1, 0], ["ok", "not ok"])

    result = pd.read_csv(csv_path)
    assert list(result["tone_formal_score"]) == [1, 0]
    assert list(result["tone_formal_reason"]) == ["ok", "not ok"]
    assert list(result["original_text"]) == ["a", "b"]  # existing columns untouched


def test_save_custom_metric_scores_to_csv_rejects_length_mismatch(tmp_path):
    csv_path = tmp_path / "cleaned.csv"
    pd.DataFrame({"original_text": ["a", "b"], "prompt": ["p1", "p2"]}).to_csv(csv_path, index=False)

    with pytest.raises(ValueError):
        save_custom_metric_scores_to_csv(str(csv_path), "tone_formal", [1], ["ok"])

    # Nothing should have been written on the rejected call.
    result = pd.read_csv(csv_path)
    assert "tone_formal_score" not in result.columns


def prompt_built_test(prompt, orig, simp, dimensions_mock):
    assert orig in prompt
    assert simp in prompt
    # No template placeholders should survive formatting
    assert "{original}" not in prompt
    assert "{simplified}" not in prompt
    assert "{explanations}" not in prompt
    assert "{eval_dimensions}" not in prompt
    # Each mock dimension should appear as a JSON key with reason/score scaffold

    for dim, exp in dimensions_mock.items():
        assert f'"{dim}"' in prompt, f"Dimension not formatted as JSON key: {dim}"
        assert f'{exp}' in prompt
    assert '"reason"' in prompt
    assert '"score"' in prompt


if __name__ == "__main__":
    tests = [
        test_build_eval_prompt_contains_inputs,
        test_build_eval_prompt_contains_all_dimensions,
        test_readability_returns_expected_keys,
        test_evaluate_parses_valid_json,
        test_evaluate_handles_bad_json,
        test_evaluate_handles_missing_dimension,
        test_eval_prompt_gets_built_correctly,
        test_eval_prompt_works_correctly,
        test_eval_dims_work_correctly,
        test_evaluate_with_custom_dimensions_scores_only_that_dimension,
        test_build_eval_prompt_custom_dimensions_excludes_the_standard_five,
        test_evaluate_new_metric_adds_score_and_reason_columns,
        test_evaluate_new_metric_reports_progress_per_row,
        test_eval_dimensions_excludes_custom_metrics,
        # The tmp_path-based persistence tests below need pytest's fixture
        # injection to run — they'll show FAIL here (no t() args to supply
        # it with), same as this file's existing capsys-requiring tests.
        # Run via `pytest` for these to actually execute.
        test_load_all_metric_definitions_empty_when_file_missing,
        test_load_custom_metric_definitions_empty_when_file_missing,
        test_load_custom_metric_definitions_excludes_standard_entries,
        test_save_then_load_custom_metric_definition_roundtrips,
        test_save_custom_metric_definition_defaults_date_added_to_today,
        test_save_custom_metric_definition_merges_without_clobbering,
        test_save_custom_metric_definition_overwrites_same_slug,
        test_save_custom_metric_definition_preserves_standard_entries,
        test_save_custom_metric_scores_to_csv_writes_new_columns,
        test_save_custom_metric_scores_to_csv_rejects_length_mismatch,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
