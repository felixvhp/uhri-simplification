"""Tests for the evaluation module."""

import json
from unittest.mock import patch, MagicMock

import pandas as pd

import pytest

from src.evaluation import (
    build_eval_prompt,
    readability,
    evaluate,
    evaluate_missing_dimensions,
    active_metric_definitions,
    load_all_metric_definitions,
    save_custom_metric_definition,
    save_custom_metric_scores_to_csv,
    set_metric_active,
)


# --- Pure function tests ---

def test_build_eval_prompt_contains_inputs():
    orig = "States must protect children."
    simp = "Countries need to keep children safe."
    prompt = build_eval_prompt(orig, simp, active_metric_definitions())
    assert orig in prompt
    assert simp in prompt


def test_build_eval_prompt_contains_all_dimensions():
    prompt = build_eval_prompt("a", "b", active_metric_definitions())
    for dim in active_metric_definitions():
        assert dim in prompt, f"Missing dimension: {dim}"


def test_readability_returns_expected_keys():
    result = readability("The cat sat on the mat.")
    assert "fk_grade" in result
    assert "fk_ease" in result
    assert isinstance(result["fk_grade"], float)


# --- LLM-mocked tests ---

MOCK_LLM_RESPONSE = """some random gibberish {
    "obligations_complete":       {"score": true,  "reason": "ok"},
    "actors_exact":               {"score": true,  "reason": "ok"},
    "deontic_force":              {"score": false, "reason": "changed must->should"},
    "terms_of_art":               {"score": true,  "reason": "ok"},
    "references_precise":         {"score": true,  "reason": "ok"},
    "scope_and_qualifiers":       {"score": true,  "reason": "ok"},
    "additions_explanatory_only": {"score": true,  "reason": "ok"},
    "plain_vocabulary":           {"score": true,  "reason": "ok"},
    "standalone_comprehension":   {"score": true,  "reason": "ok"}
}"""


def _mock_client(response_text):
    mock_msg = MagicMock()
    mock_msg.content[0].text = response_text
    mock_client = MagicMock()
    mock_client.messages.create.return_value = mock_msg
    return mock_client


def test_evaluate_parses_valid_json():
    with patch("src.evaluation.eval_client", _mock_client(MOCK_LLM_RESPONSE)):
        result = evaluate("original", "simplified", active_metric_definitions())
    assert result["obligations_complete"]["score"] == 1
    assert result["deontic_force"]["score"] == 0


def test_evaluate_handles_bad_json():
    with patch("src.evaluation.eval_client", _mock_client("not json")):
        result = evaluate("original", "simplified", active_metric_definitions())
    for dim in active_metric_definitions():
        assert result[dim]["score"] == -1


def test_evaluate_handles_missing_dimension():
    # Response missing one dimension — that dim should get score -1
    partial = MOCK_LLM_RESPONSE.replace(
        '"references_precise":         {"score": true,  "reason": "ok"},\n', ""
    )
    with patch("src.evaluation.eval_client", _mock_client(partial)):
        result = evaluate("original", "simplified", active_metric_definitions())
    assert result["references_precise"]["score"] == -1


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
    with patch("src.evaluation.EVAL_PROMPT", prompt_mock):
        mocked_prompt = build_eval_prompt(orig, simp, dimensions_mock)

    prompt_built_test(mocked_prompt, orig, simp, dimensions_mock)


def test_eval_prompt_works_correctly():
    prompt_mock, dimensions_mock, orig, simp = build_prompt_context()
    orig = str(orig)

    # Testing with the real EVAL_PROMPT template
    prompt = build_eval_prompt(orig, simp, dimensions_mock)

    prompt_built_test(prompt, orig, simp, dimensions_mock)


# --- evaluate() with an explicit `dimensions` subset ---


def test_evaluate_with_custom_dimensions_scores_only_that_dimension():
    response = '{"tone_formal": {"score": true, "reason": "reads as formal"}}'
    with patch("src.evaluation.eval_client", _mock_client(response)):
        result = evaluate("original", "simplified", dimensions={"tone_formal": "Is it formal?"})
    assert list(result.keys()) == ["tone_formal"]
    assert result["tone_formal"]["score"] == 1


def test_build_eval_prompt_custom_dimensions_excludes_other_active_metrics():
    prompt = build_eval_prompt("a", "b", dimensions={"tone_formal": "Is it formal?"})
    assert "tone_formal" in prompt
    for dim in active_metric_definitions():
        assert dim not in prompt


# --- evaluate_missing_dimensions ---
# (slugify() itself is tested in test_text_utils.py, not here.)


def test_evaluate_missing_dimensions_adds_columns_for_a_slug_with_no_data():
    df = pd.DataFrame({"original_text": ["a", "b"], "transcribed_text": ["x", "y"]})
    response = '{"new_dim": {"score": true, "reason": "ok"}}'
    with patch("src.evaluation.eval_client", _mock_client(response)):
        result = evaluate_missing_dimensions(df, {"new_dim": "Is it new?"})
    assert list(result["new_dim_score"]) == [1, 1]
    assert list(result["new_dim_reason"]) == ["ok", "ok"]


def test_evaluate_missing_dimensions_skips_rows_already_fully_scored():
    df = pd.DataFrame({
        "original_text": ["a", "b"],
        "transcribed_text": ["x", "y"],
        "dim_score": [1, float("nan")],
        "dim_reason": ["already judged", None],
    })
    response = '{"dim": {"score": false, "reason": "freshly judged"}}'
    with patch("src.evaluation.eval_client", _mock_client(response)) as client:
        result = evaluate_missing_dimensions(df, {"dim": "Some question?"})
    # Only the second (NaN) row needed a call.
    assert client.messages.create.call_count == 1
    assert list(result["dim_score"]) == [1, 0]
    assert list(result["dim_reason"]) == ["already judged", "freshly judged"]


def test_evaluate_missing_dimensions_makes_no_calls_when_nothing_is_missing():
    df = pd.DataFrame({
        "original_text": ["a"],
        "transcribed_text": ["x"],
        "dim_score": [1],
        "dim_reason": ["ok"],
    })
    with patch("src.evaluation.eval_client", _mock_client("unused")) as client:
        result = evaluate_missing_dimensions(df, {"dim": "Some question?"})
    assert client.messages.create.call_count == 0
    assert list(result["dim_score"]) == [1]


def test_evaluate_missing_dimensions_reports_progress_over_rows_needing_work_only():
    df = pd.DataFrame({
        "original_text": ["a", "b", "c"],
        "transcribed_text": ["x", "y", "z"],
        "dim_score": [1, float("nan"), float("nan")],
        "dim_reason": ["ok", None, None],
    })
    response = '{"dim": {"score": true, "reason": "ok"}}'
    progress_calls = []
    with patch("src.evaluation.eval_client", _mock_client(response)):
        evaluate_missing_dimensions(
            df, {"dim": "Some question?"},
            on_progress=lambda done, total: progress_calls.append((done, total)),
        )
    # 1 row already had data, so only 2 need work.
    assert progress_calls == [(0, 2), (1, 2), (2, 2)]


def test_evaluate_missing_dimensions_only_asks_for_a_rows_specific_gaps():
    # Row 0 is missing only "b", row 1 is missing only "a" — each call must
    # ask for just that row's gap, not both dimensions every time.
    df = pd.DataFrame({
        "original_text": ["a", "b"],
        "transcribed_text": ["x", "y"],
        "a_score": [1, float("nan")],
        "a_reason": ["ok", None],
        "b_score": [float("nan"), 1],
        "b_reason": [None, "ok"],
    })
    seen_prompts = []

    def fake_create(**kwargs):
        seen_prompts.append(kwargs["messages"][0]["content"])
        mock_msg = MagicMock()
        mock_msg.content[0].text = '{"a": {"score": true, "reason": "ok"}, "b": {"score": true, "reason": "ok"}}'
        return mock_msg

    mock_client = MagicMock()
    mock_client.messages.create.side_effect = fake_create
    with patch("src.evaluation.eval_client", mock_client):
        evaluate_missing_dimensions(df, {"a": "Question A?", "b": "Question B?"})

    assert len(seen_prompts) == 2
    assert "Question B?" in seen_prompts[0] and "Question A?" not in seen_prompts[0]
    assert "Question A?" in seen_prompts[1] and "Question B?" not in seen_prompts[1]


def test_evaluate_missing_dimensions_rows_restricts_which_rows_are_touched():
    # A subset picked in the GUI (e.g. a random sample of a huge dataset) —
    # only rows named in `rows` may be judged, even if others outside it
    # also have gaps.
    df = pd.DataFrame({"original_text": ["a", "b", "c"], "transcribed_text": ["x", "y", "z"]})
    response = '{"dim": {"score": true, "reason": "ok"}}'
    with patch("src.evaluation.eval_client", _mock_client(response)) as client:
        result = evaluate_missing_dimensions(df, {"dim": "Some question?"}, rows=[0, 2])
    assert client.messages.create.call_count == 2
    assert list(result["dim_score"].notna()) == [True, False, True]


def test_evaluate_missing_dimensions_rows_still_skips_already_judged_rows():
    df = pd.DataFrame({
        "original_text": ["a", "b"],
        "transcribed_text": ["x", "y"],
        "dim_score": [1, float("nan")],
        "dim_reason": ["already judged", None],
    })
    with patch("src.evaluation.eval_client", _mock_client('{"dim": {"score": true, "reason": "ok"}}')) as client:
        evaluate_missing_dimensions(df, {"dim": "Some question?"}, rows=[0, 1])
    # Row 0 is in `rows` but already judged — still costs nothing.
    assert client.messages.create.call_count == 1


# --- metric definition persistence (metrics.json: every metric, together) ---
# All of these pass an explicit tmp_path file, never the real METRICS_PATH/a
# real CSV, so they can't touch anything tracked.


def test_load_all_metric_definitions_empty_when_file_missing(tmp_path):
    assert load_all_metric_definitions(tmp_path / "does_not_exist.json") == {}


def test_save_then_load_metric_definition_roundtrips(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("tone_formal", "Tone: formal?", "Is it formal?", path)
    assert load_all_metric_definitions(path) == {
        "tone_formal": {
            "name": "Tone: formal?",
            "description": "Is it formal?",
            "active": True,
        }
    }


def test_save_custom_metric_definition_merges_without_clobbering(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("a", "A", "desc a", path)
    save_custom_metric_definition("b", "B", "desc b", path)
    assert set(load_all_metric_definitions(path)) == {"a", "b"}


def test_save_custom_metric_definition_overwrites_same_slug(tmp_path):
    path = tmp_path / "metrics.json"
    save_custom_metric_definition("a", "A", "first version", path)
    save_custom_metric_definition("a", "A", "second version", path)
    defs = load_all_metric_definitions(path)
    assert len(defs) == 1
    assert defs["a"]["description"] == "second version"


def test_save_custom_metric_definition_preserves_other_entries(tmp_path):
    # Regression guard: saving a new metric must read/write the *whole*
    # file, not just one entry, or it would silently drop whatever else
    # lives in the same metrics.json.
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "content_preserved": {"name": "content_preserved", "description": "d", "active": True},
    }))
    save_custom_metric_definition("tone_formal", "Tone", "d2", path)
    all_defs = load_all_metric_definitions(path)
    assert "content_preserved" in all_defs
    assert "tone_formal" in all_defs and all_defs["tone_formal"]["description"] == "d2"


def test_active_metric_definitions_excludes_retired_metrics():
    # active_metric_definitions() is what evaluate() judges when a caller
    # asks for "every active metric" — it must never silently include one
    # marked active: false in the real metrics.json.
    active = active_metric_definitions()
    assert set(active) <= set(load_all_metric_definitions())
    for slug, entry in load_all_metric_definitions().items():
        if not entry.get("active", True):
            assert slug not in active


def test_set_metric_active_retires_without_deleting(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "content_preserved": {"name": "content_preserved", "description": "d", "active": True},
    }))
    set_metric_active("content_preserved", False, path)
    all_defs = load_all_metric_definitions(path)
    assert all_defs["content_preserved"]["active"] is False
    # Everything else about the entry survives untouched, and the entry
    # itself is never dropped from load_all_metric_definitions().
    assert all_defs["content_preserved"]["description"] == "d"
    assert "content_preserved" in all_defs
    assert "content_preserved" not in active_metric_definitions(path)


def test_set_metric_active_can_reactivate(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({
        "content_preserved": {"name": "content_preserved", "description": "d", "active": False},
    }))
    set_metric_active("content_preserved", True, path)
    assert load_all_metric_definitions(path)["content_preserved"]["active"] is True
    assert "content_preserved" in active_metric_definitions(path)


def test_set_metric_active_raises_for_unknown_slug(tmp_path):
    path = tmp_path / "metrics.json"
    path.write_text(json.dumps({}))
    with pytest.raises(KeyError):
        set_metric_active("does_not_exist", False, path)


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
        test_evaluate_with_custom_dimensions_scores_only_that_dimension,
        test_build_eval_prompt_custom_dimensions_excludes_other_active_metrics,
        test_evaluate_missing_dimensions_adds_columns_for_a_slug_with_no_data,
        test_evaluate_missing_dimensions_skips_rows_already_fully_scored,
        test_evaluate_missing_dimensions_makes_no_calls_when_nothing_is_missing,
        test_evaluate_missing_dimensions_reports_progress_over_rows_needing_work_only,
        test_evaluate_missing_dimensions_only_asks_for_a_rows_specific_gaps,
        test_evaluate_missing_dimensions_rows_restricts_which_rows_are_touched,
        test_evaluate_missing_dimensions_rows_still_skips_already_judged_rows,
        test_active_metric_definitions_excludes_retired_metrics,
        # The tmp_path-based persistence tests below need pytest's fixture
        # injection to run — they'll show FAIL here (no t() args to supply
        # it with), same as this file's existing capsys-requiring tests.
        # Run via `pytest` for these to actually execute.
        test_load_all_metric_definitions_empty_when_file_missing,
        test_save_then_load_metric_definition_roundtrips,
        test_save_custom_metric_definition_merges_without_clobbering,
        test_save_custom_metric_definition_overwrites_same_slug,
        test_save_custom_metric_definition_preserves_other_entries,
        test_set_metric_active_retires_without_deleting,
        test_set_metric_active_can_reactivate,
        test_set_metric_active_raises_for_unknown_slug,
        test_save_custom_metric_scores_to_csv_writes_new_columns,
        test_save_custom_metric_scores_to_csv_rejects_length_mismatch,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
