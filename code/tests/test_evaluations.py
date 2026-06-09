"""Tests for the evaluation module."""

from unittest.mock import patch, MagicMock
from src.evaluation import (
    build_eval_prompt,
    readability,
    evaluate,
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
    orig = "Test original text."
    simp = "Test simplified text."
    return prompt_mock, dimensions_mock, orig, simp

def test_eval_prompt_gets_built_correctly():
    prompt_mock, dimensions_mock, orig, simp = build_prompt_context()


    # Testing with a mocked prompt
    with patch("src.evaluation.EVAL_PROMPT", prompt_mock), \
         patch("src.evaluation.EVAL_DIMENSIONS", dimensions_mock):
        mocked_prompt = build_eval_prompt(orig, simp)

    print(mocked_prompt)

    prompt_built_test(mocked_prompt, orig, simp, dimensions_mock)


def test_eval_prompt_works_correctly():
    prompt_mock, dimensions_mock, orig, simp = build_prompt_context()

    # Testing with the real prompt
    with patch("src.evaluation.EVAL_DIMENSIONS", dimensions_mock):
        prompt = build_eval_prompt(orig, simp)

    prompt_built_test(prompt, orig, simp, dimensions_mock)

def test_eval_dims_work_correctly():
    prompt_mock, dimensions_mock, orig, simp = build_prompt_context()

    # Testing with the real prompt
    with patch("src.evaluation.EVAL_PROMPT", prompt_mock):
        prompt = build_eval_prompt(orig, simp)

    print(prompt)

    prompt_built_test(prompt, orig, simp, EVAL_DIMENSIONS)


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
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
