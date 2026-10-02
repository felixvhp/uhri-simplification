"""Tests for src/simplification — the input-to-output interface.

The OpenAI call is mocked everywhere (no real API calls): the fake returns
"<first line of the prompt> :: simplified", so a test can tell which
template produced which output.

Run from the repo root:
    python -m pytest tests/test_simplification.py
"""

import json
import warnings
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from src.simplification import (
    MultiplePromptsWarning,
    read_rows,
    simplify,
    simplify_file,
    simplify_rows,
    simplify_text,
    write_rows,
)
from src.simplification.core import _MissingContext, build_context, output_columns, resolve_prompts

ONE = {"plain": "PLAIN {text}"}
TWO = {"plain": "PLAIN {text}", "short": "SHORT {text}"}

ROWS = [
    {"OHCHR Annotation Id": "A-1", "Text": "First legal text.", "Countries Concerned": "- Chile"},
    {"OHCHR Annotation Id": "A-2", "Text": "Second legal text.", "Countries Concerned": "- Peru"},
]


@pytest.fixture
def fake_llm():
    """Replace the OpenAI call with an echo of the built prompt."""
    with patch("src.simplification.core._call_openai",
               side_effect=lambda prompt, model, effort: f"{prompt} :: simplified") as mock:
        yield mock


# ── Prompt templates and context ──────────────────────────────────────────────


def test_missing_context_fills_unknown_key_with_placeholder_note():
    assert "{text} {country}".format_map(_MissingContext(text="hi")) == "hi (not available)"


def test_build_context_reads_and_cleans_ohchr_columns():
    row = {"Text": "t", "Countries Concerned": " - Chile ", "Reccomending Body": "- CEDAW", "Type": "- Recommendations"}
    assert build_context(row, "Text") == {
        "text": "t", "country": "Chile", "body": "CEDAW", "type": "recommendation",
    }


def test_build_context_maps_other_types_to_observation():
    assert build_context({"Text": "t", "Type": "- Concerns/Observations"}, "Text")["type"] == "observation"


def test_build_context_skips_columns_the_row_does_not_have():
    assert build_context({"body_text": "t"}, "body_text") == {"text": "t"}


def test_simplify_text_sends_the_built_prompt_to_the_configured_model():
    client = MagicMock()
    client.responses.create.return_value.output_text = "simplified!"
    with patch("src.simplification.core.openai_client", client):
        result = simplify_text("legal text", "Rewrite for {country}: {text}", {"country": "Chile"}, model="m-1")

    assert result == "simplified!"
    kwargs = client.responses.create.call_args.kwargs
    assert kwargs["input"] == "Rewrite for Chile: legal text"
    assert kwargs["model"] == "m-1"


def test_resolve_prompts_defaults_to_the_selected_ones():
    with patch("src.simplification.core.selected_prompts", return_value=ONE):
        assert resolve_prompts() == ONE


def test_resolve_prompts_looks_up_slugs_and_rejects_unknown_ones():
    definitions = {"plain": {"template": "PLAIN {text}"}}
    with patch("src.simplification.core.load_prompts", return_value=definitions):
        assert resolve_prompts("plain") == ONE
        assert resolve_prompts(["plain"]) == ONE
        with pytest.raises(ValueError, match="Unknown prompt"):
            resolve_prompts(["nope"])


def test_resolve_prompts_rejects_an_empty_selection():
    with patch("src.simplification.core.selected_prompts", return_value={}):
        with pytest.raises(ValueError, match="No prompt selected"):
            resolve_prompts()


def test_output_columns_naming():
    assert output_columns(ONE) == {"plain": "simplified"}
    assert output_columns(TWO) == {"plain": "simplified_plain", "short": "simplified_short"}


# ── simplify_rows: one prompt ─────────────────────────────────────────────────


def test_single_prompt_keeps_every_original_column_and_adds_simplified(fake_llm):
    result = simplify_rows(ROWS, prompts=ONE)

    assert len(result) == 2
    for original, out in zip(ROWS, result):
        assert {k: out[k] for k in original} == original  # every column, key included
        assert list(out) == list(original) + ["simplified"]
    assert result[0]["simplified"] == "PLAIN First legal text. :: simplified"
    assert fake_llm.call_count == 2


def test_input_rows_are_not_modified(fake_llm):
    before = [dict(row) for row in ROWS]
    simplify_rows(ROWS, prompts=ONE)
    assert ROWS == before


def test_single_prompt_raises_no_warning(fake_llm):
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        simplify_rows(ROWS, prompts=ONE)


def test_uses_the_selected_prompt_when_none_is_given(fake_llm):
    with patch("src.simplification.core.selected_prompts", return_value={"chosen": "CHOSEN {text}"}):
        result = simplify_rows(ROWS)
    assert result[0]["simplified"].startswith("CHOSEN ")


# ── simplify_rows: several prompts ────────────────────────────────────────────


def test_several_prompts_warn_before_any_call_and_add_one_column_each(fake_llm):
    def fail_if_called_before_warning(*args):
        raise AssertionError("API called before the warning was raised")

    fake_llm.side_effect = fail_if_called_before_warning
    with pytest.raises(MultiplePromptsWarning):
        with warnings.catch_warnings():
            warnings.simplefilter("error", MultiplePromptsWarning)
            simplify_rows(ROWS, prompts=TWO)
    assert fake_llm.call_count == 0


def test_several_prompts_output(fake_llm):
    with pytest.warns(MultiplePromptsWarning, match="2 prompts are selected"):
        result = simplify_rows(ROWS, prompts=TWO)

    assert list(result[0]) == list(ROWS[0]) + ["simplified_plain", "simplified_short"]
    assert result[1]["simplified_short"] == "SHORT Second legal text. :: simplified"
    assert fake_llm.call_count == 4


# ── simplify_rows: layout, text column, edge cases ────────────────────────────


def test_long_layout_has_one_row_per_prompt_with_dashboard_columns(fake_llm):
    with pytest.warns(MultiplePromptsWarning):
        result = simplify_rows(ROWS, prompts=TWO, layout="long")

    assert len(result) == 4
    first = result[0]
    assert first["OHCHR Annotation Id"] == "A-1"  # originals still there
    assert first["original_text"] == "First legal text."
    assert first["prompt"] == "plain"
    assert first["transcribed_text"] == "PLAIN First legal text. :: simplified"
    assert [row["prompt"] for row in result] == ["plain", "short", "plain", "short"]


def test_unknown_layout_is_rejected(fake_llm):
    with pytest.raises(ValueError, match="layout"):
        simplify_rows(ROWS, prompts=ONE, layout="sideways")


def test_explicit_text_column(fake_llm):
    result = simplify_rows([{"id": 1, "body_text": "Some text."}], prompts=ONE, text_column="body_text")
    assert result[0]["simplified"] == "PLAIN Some text. :: simplified"


def test_text_column_is_auto_detected(fake_llm):
    assert simplify_rows([{"text": "lower"}], prompts=ONE)[0]["simplified"].startswith("PLAIN lower")


def test_unknown_text_column_lists_the_available_columns(fake_llm):
    with pytest.raises(ValueError, match="'nope' not found.*id"):
        simplify_rows([{"id": 1, "Text": "x"}], prompts=ONE, text_column="nope")


def test_no_recognisable_text_column_asks_for_one(fake_llm):
    with pytest.raises(ValueError, match="text_column="):
        simplify_rows([{"id": 1, "content": "x"}], prompts=ONE)


def test_blank_text_costs_no_call_and_gets_none(fake_llm):
    rows = [{"Text": "Real text."}, {"Text": ""}, {"Text": None}, {"Text": float("nan")}]
    result = simplify_rows(rows, prompts=ONE)
    assert [row["simplified"] is None for row in result] == [False, True, True, True]
    assert fake_llm.call_count == 1


def test_existing_output_column_is_not_overwritten(fake_llm):
    with pytest.raises(ValueError, match="already has column"):
        simplify_rows([{"Text": "x", "simplified": "old"}], prompts=ONE)
    assert fake_llm.call_count == 0


def test_empty_input_returns_empty_output(fake_llm):
    assert simplify_rows([], prompts=ONE) == []


def test_progress_counts_calls(fake_llm):
    seen = []
    with pytest.warns(MultiplePromptsWarning):
        simplify_rows(ROWS, prompts=TWO, on_progress=lambda done, total: seen.append((done, total)))
    assert seen == [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)]


def test_row_context_reaches_the_template(fake_llm):
    result = simplify_rows(ROWS, prompts={"ctx": "{country}: {text}"})
    assert result[0]["simplified"] == "Chile: First legal text. :: simplified"


# ── simplify(): same shape out as in ──────────────────────────────────────────


def test_simplify_string_returns_string(fake_llm):
    assert simplify("Legal text.", prompts=ONE) == "PLAIN Legal text. :: simplified"


def test_simplify_string_with_several_prompts_returns_one_per_prompt(fake_llm):
    with pytest.warns(MultiplePromptsWarning):
        result = simplify("Legal text.", prompts=TWO)
    assert set(result) == {"simplified_plain", "simplified_short"}


def test_simplify_dict_returns_dict_with_original_keys(fake_llm):
    record = {"id": 7, "Text": "Legal text.", "nested": {"themes": ["a", "b"]}}
    result = simplify(record, prompts=ONE)
    assert result["id"] == 7 and result["nested"] == {"themes": ["a", "b"]}
    assert result["simplified"].startswith("PLAIN ")


def test_simplify_list_returns_list(fake_llm):
    result = simplify(ROWS, prompts=ONE)
    assert isinstance(result, list) and len(result) == 2


def test_simplify_dataframe_returns_dataframe_with_all_columns(fake_llm):
    df = pd.DataFrame(ROWS)
    result = simplify(df, prompts=ONE)
    assert isinstance(result, pd.DataFrame)
    assert list(result.columns) == list(df.columns) + ["simplified"]
    assert list(df.columns) == list(ROWS[0])  # input frame untouched


def test_simplify_rejects_unsupported_types(fake_llm):
    with pytest.raises(TypeError):
        simplify(42, prompts=ONE)


# ── Files: formats in, formats out ────────────────────────────────────────────


@pytest.mark.parametrize("suffix", [".csv", ".tsv", ".xlsx", ".json", ".jsonl"])
def test_rows_roundtrip_through_every_format(tmp_path, suffix):
    path = tmp_path / f"data{suffix}"
    write_rows(ROWS, path)
    assert read_rows(path) == ROWS


def test_tabular_files_keep_ids_as_written(tmp_path):
    path = tmp_path / "ids.csv"
    path.write_text("id,Text\n007,Some text\n")
    assert read_rows(path) == [{"id": "007", "Text": "Some text"}]


def test_json_must_be_a_list_of_objects(tmp_path):
    path = tmp_path / "wrapped.json"
    path.write_text(json.dumps({"results": ROWS}))
    with pytest.raises(ValueError, match="list of objects"):
        read_rows(path)


def test_unsupported_extension_is_rejected(tmp_path):
    with pytest.raises(ValueError, match="Unsupported file type"):
        read_rows(tmp_path / "data.parquet")


def test_simplify_file_defaults_output_next_to_input(tmp_path, fake_llm):
    source = tmp_path / "recs.csv"
    write_rows(ROWS, source)

    output = simplify_file(source, prompts=ONE)

    assert output == tmp_path / "recs.simplified.csv"
    written = read_rows(output)
    assert [row["OHCHR Annotation Id"] for row in written] == ["A-1", "A-2"]
    assert written[0]["simplified"] == "PLAIN First legal text. :: simplified"


def test_simplify_file_converts_between_formats(tmp_path, fake_llm):
    source = tmp_path / "recs.xlsx"
    write_rows(ROWS, source)

    output = simplify_file(source, tmp_path / "out" / "recs.json", prompts=ONE)

    written = json.loads(output.read_text())
    assert written[1]["Countries Concerned"] == "- Peru"
    assert "simplified" in written[1]


def test_simplify_file_sample_is_reproducible(tmp_path, fake_llm):
    source = tmp_path / "recs.jsonl"
    write_rows([{"id": i, "text": f"Text {i}."} for i in range(20)], source)

    first = read_rows(simplify_file(source, tmp_path / "a.jsonl", sample_size=5, seed=1, prompts=ONE))
    second = read_rows(simplify_file(source, tmp_path / "b.jsonl", sample_size=5, seed=1, prompts=ONE))

    assert len(first) == 5
    assert [row["id"] for row in first] == [row["id"] for row in second]


def test_simplify_path_dispatches_to_simplify_file(tmp_path, fake_llm):
    source = tmp_path / "recs.csv"
    write_rows(ROWS, source)
    assert simplify(source, prompts=ONE) == tmp_path / "recs.simplified.csv"
