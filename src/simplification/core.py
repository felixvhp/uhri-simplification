"""
Pkg 3: Simplifying UHRI texts with an LLM [last updated 2026-10-02]

Usage:
    from src.simplification import simplify, simplify_file

    simplify_file("data/email_recs.xlsx", "data/email_recs.simplified.csv")
    rows = simplify([{"id": 7, "Text": "The Committee recommends ..."}])
    df = simplify(my_dataframe, text_column="recommendation")
    text = simplify("The Committee recommends ...")

    Command line: python -m src.simplification --help

Whatever goes in comes back in the same shape with every original column
(or key) untouched, plus one new column per prompt:

    one prompt selected     ->  "simplified"
    several selected        ->  "simplified_<prompt>" each, after a warning

Which prompts run is ``selected_prompts`` in the shared settings (see
src/settings); pass ``prompts=`` to override for one call. The model comes
from the settings too.

Requirements:
    OPENAI_API_KEY in a .env file at the repo root (see .env.example) or as
    a shell env var. Every row costs one live OpenAI call per prompt.

Open Issues:
    - Calls run one after another; a large file takes a while.
    - A failed call aborts the run; nothing is written for the rows done so far.

Resumption:
    A new input shape goes in simplify()'s dispatch; a new file format in
    io.py. Judging the output is not this folder's job — the dashboard
    does that from the file written here (use layout="long").
"""

import random
import warnings
from pathlib import Path
from typing import Callable, Optional, Union

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from ..settings import load_prompts, load_settings, selected_prompts
from .io import read_rows, write_rows

load_dotenv()  # loads OPENAI_API_KEY from .env if present

# Tried in order when no text column is given or configured.
TEXT_COLUMNS = ("Text", "text", "original_text")

# Row columns that fill a template's {country}/{body}/{type} slots, first
# match wins. "Reccomending Body" is OHCHR's own typo in the export.
_CONTEXT_COLUMNS = {
    "country": ("Countries Concerned", "country"),
    "body": ("Reccomending Body", "Recommending Body", "recommending_body", "body"),
    "type": ("Type", "type"),
}

# Column names of layout="long" — the format the dashboard and the feedback
# form read: one row per (input row, prompt).
LONG_ORIGINAL, LONG_PROMPT, LONG_SIMPLIFIED = "original_text", "prompt", "transcribed_text"

openai_client: Optional[OpenAI] = None  # built on first use, so importing needs no API key


class MultiplePromptsWarning(UserWarning):
    """More than one prompt is selected, so every row gets several simplifications."""


class _MissingContext(dict):
    """`str.format_map` helper: an unset placeholder renders as a note instead of raising.

    A row from a generic file may have no country/body/type to fill
    {country}/{body}/{type} with, unlike a row of the OHCHR export.
    """

    def __missing__(self, key):
        return "(not available)"


def _get_openai_client() -> OpenAI:
    """Return the shared OpenAI client, creating it on first call."""
    global openai_client
    if openai_client is None:
        openai_client = OpenAI()
    return openai_client


def _call_openai(prompt: str, model: str, reasoning_effort: str) -> str:
    """Send a fully-built prompt to OpenAI and return the generated text."""
    response = _get_openai_client().responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": reasoning_effort},  # "minimal" | "low" | "medium" | "high"
    )
    return response.output_text


def _is_blank(value) -> bool:
    """True for None, NaN and empty/whitespace-only values."""
    if value is None:
        return True
    if isinstance(value, float) and value != value:  # NaN
        return True
    return not str(value).strip()


def _clean(value) -> str:
    """Strip whitespace and the "- " list prefix the OHCHR export puts on values."""
    return str(value).strip().lstrip("- ")


def build_context(row: dict, text_column: str) -> dict:
    """Collect what a prompt template can refer to for one row.

    Args:
        row: One input row.
        text_column: Name of the column holding the text to simplify.

    Returns:
        Dict with ``text`` and whichever of ``country``/``body``/``type``
        the row has a column for (see _CONTEXT_COLUMNS).
    """
    context = {"text": row[text_column]}
    for slot, candidates in _CONTEXT_COLUMNS.items():
        for column in candidates:
            if column in row and not _is_blank(row[column]):
                context[slot] = _clean(row[column])
                break
    if "type" in context:
        context["type"] = "recommendation" if context["type"] == "Recommendations" else "observation"
    return context


def simplify_text(text: str, template: str, context: Optional[dict] = None, model: Optional[str] = None) -> str:
    """Simplify one text with one prompt template.

    Args:
        text: The original text.
        template: Prompt template with a {text} slot and optionally
            {type}/{body}/{country}.
        context: Values for the optional slots. A slot with no value
            renders as "(not available)".
        model: OpenAI model ID. Defaults to the settings' model.

    Returns:
        The model's simplified text.
    """
    config = load_settings()["simplification"]
    prompt = template.format_map(_MissingContext({**(context or {}), "text": text}))
    return _call_openai(prompt, model or config["model"], config["reasoning_effort"])


def resolve_prompts(prompts: Union[None, str, list, dict] = None) -> dict:
    """Turn a ``prompts`` argument into slug -> template.

    Args:
        prompts: None for the prompts selected in the settings, one slug, a
            list of slugs defined in prompts.json, or a ready
            slug -> template dict.

    Returns:
        Dict of slug -> template, in the order given.

    Raises:
        ValueError: If nothing is selected or a slug isn't defined.
    """
    if prompts is None:
        resolved = selected_prompts()
    elif isinstance(prompts, dict):
        resolved = dict(prompts)
    else:
        slugs = [prompts] if isinstance(prompts, str) else list(prompts)
        definitions = load_prompts()
        unknown = [slug for slug in slugs if slug not in definitions]
        if unknown:
            raise ValueError(f"Unknown prompt(s) {unknown}. Defined: {sorted(definitions)}")
        resolved = {slug: definitions[slug]["template"] for slug in slugs}
    if not resolved:
        raise ValueError("No prompt selected — set selected_prompts in the settings or pass prompts=.")
    return resolved


def output_columns(prompts: dict) -> dict:
    """Name of the added column for each prompt: slug -> column name."""
    if len(prompts) == 1:
        return {slug: "simplified" for slug in prompts}
    return {slug: f"simplified_{slug}" for slug in prompts}


def _find_text_column(rows: list, text_column: Optional[str]) -> str:
    """Pick the column holding the text: explicit, then configured, then a known name."""
    columns = list(rows[0])
    wanted = text_column or load_settings()["simplification"].get("text_column")
    if wanted:
        if wanted not in columns:
            raise ValueError(f"Text column '{wanted}' not found. Columns: {columns}")
        return wanted
    for candidate in TEXT_COLUMNS:
        if candidate in columns:
            return candidate
    raise ValueError(
        f"Could not tell which column holds the text (looked for {list(TEXT_COLUMNS)}). "
        f"Pass text_column=. Columns: {columns}"
    )


def simplify_rows(
    rows: list,
    prompts: Union[None, str, list, dict] = None,
    text_column: Optional[str] = None,
    model: Optional[str] = None,
    layout: str = "wide",
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> list:
    """Simplify the text of every row, keeping everything else the row has.

    Args:
        rows: List of dicts, one per record. Not modified.
        prompts: Which prompts to run — see resolve_prompts(). Defaults to
            the ones selected in the settings.
        text_column: Column holding the text. Defaults to the settings'
            ``text_column``, then the first of TEXT_COLUMNS present.
        model: OpenAI model ID. Defaults to the settings' model.
        layout: "wide" returns one row per input row with one added column
            per prompt (see output_columns()). "long" returns one row per
            (input row, prompt) with ``original_text``, ``prompt`` and
            ``transcribed_text`` added — what the dashboard and the
            feedback form read.
        on_progress: Optional ``callback(done, total)`` counting API calls.

    Returns:
        New list of dicts: every original key, plus the added columns. A row
        with no text gets None instead of a simplification (and costs no call).

    Raises:
        ValueError: Unknown layout or prompt, no text column found, or an
            added column name already exists in the data.

    Warns:
        MultiplePromptsWarning: Before any call is made, if more than one
            prompt will run.
    """
    if layout not in ("wide", "long"):
        raise ValueError(f"layout must be 'wide' or 'long', not '{layout}'")
    templates = resolve_prompts(prompts)
    if not rows:
        return []
    text_column = _find_text_column(rows, text_column)

    added = list(output_columns(templates).values()) if layout == "wide" else [LONG_PROMPT, LONG_SIMPLIFIED]
    clash = [column for column in added if column in rows[0]]
    if clash:
        raise ValueError(f"The data already has column(s) {clash}; rename them before simplifying.")

    if len(templates) > 1:
        warnings.warn(
            f"{len(templates)} prompts are selected ({', '.join(templates)}): every row gets "
            f"{len(templates)} simplifications. Select a single prompt for one output.",
            MultiplePromptsWarning,
            stacklevel=2,
        )

    columns = output_columns(templates)
    total = sum(1 for row in rows if not _is_blank(row[text_column])) * len(templates)
    done = 0
    if on_progress:
        on_progress(0, total)

    result = []
    for row in rows:
        text = row[text_column]
        out = dict(row)
        for slug, template in templates.items():
            simplified = None
            if not _is_blank(text):
                simplified = simplify_text(text, template, build_context(row, text_column), model)
                done += 1
                if on_progress:
                    on_progress(done, total)
            if layout == "wide":
                out[columns[slug]] = simplified
            else:
                result.append({**row, LONG_ORIGINAL: text, LONG_PROMPT: slug, LONG_SIMPLIFIED: simplified})
        if layout == "wide":
            result.append(out)
    return result


def simplify_file(
    input_path: Union[str, Path],
    output_path: Union[str, Path, None] = None,
    sample_size: Optional[int] = None,
    seed: int = 42,
    **options,
) -> Path:
    """Simplify a data file and write the result.

    Input and output formats are chosen by file extension and need not
    match (.xlsx, .csv, .tsv, .json, .jsonl — see io.py).

    Args:
        input_path: File with one record per row.
        output_path: Where to write. Defaults to
            ``<input name>.simplified.<input extension>`` next to the input.
        sample_size: Only simplify a random subset of this many rows.
        seed: Fixes which rows get sampled, so runs are reproducible.
        **options: Passed to simplify_rows() (prompts, text_column, model,
            layout, on_progress).

    Returns:
        The path written.
    """
    input_path = Path(input_path)
    output_path = Path(output_path) if output_path else input_path.with_name(
        f"{input_path.stem}.simplified{input_path.suffix}"
    )
    rows = read_rows(input_path)
    if sample_size is not None and sample_size < len(rows):
        rows = random.Random(seed).sample(rows, sample_size)
    write_rows(simplify_rows(rows, **options), output_path)
    return output_path


def simplify(data, **options):
    """Simplify whatever is handed in and return it in the same shape.

    Args:
        data: One of
            - str: a single text. Returns the simplified text (or a
              prompt -> text dict when several prompts run).
            - dict: a single record. Returns the record with added keys.
            - list of dicts: records. Returns a new list.
            - pandas DataFrame. Returns a new DataFrame.
            - pathlib.Path: a data file. Returns the output path (see
              simplify_file(); pass output_path= to choose it).
        **options: See simplify_rows() / simplify_file().

    Returns:
        Same shape as ``data``, with the simplification column(s) added.
    """
    if isinstance(data, Path):
        return simplify_file(data, **options)
    if isinstance(data, str):
        row = simplify_rows([{"text": data}], text_column="text", **{**options, "layout": "wide"})[0]
        added = {key: value for key, value in row.items() if key != "text"}
        return added["simplified"] if list(added) == ["simplified"] else added
    if isinstance(data, dict):
        result = simplify_rows([data], **options)
        return result[0] if options.get("layout", "wide") == "wide" else result
    if isinstance(data, pd.DataFrame):
        return pd.DataFrame(simplify_rows(data.to_dict("records"), **options))
    if isinstance(data, list):
        return simplify_rows(data, **options)
    raise TypeError(f"Don't know how to simplify a {type(data).__name__}")
