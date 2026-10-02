"""
Pkg 3: Natural language translation [last updated 2026-09-13]

The goal of this document is to test out using LLMs for simplifying the legalese texts.
Motivation:
    Using the OpenAI API allows access to frontier models without local hardware.
    Since presumably original text will always be provided also hallucinations should be less of an issue.
    Hallucinations also get reduced by the small context window, so no need for relying on background information.
    Enables easy tweaking of the "translation" through tweaking eg. the model or the system prompt.

Requirements:
    pip install openai python-dotenv
    Set OPENAI_API_KEY in a .env file at the repo root (see .env.example) or as a shell env var.

Input:
    A UHRI export file at DATA_PATH, loaded via src.Dataloader.load_records.

Output:
    Simplified text written to OUTPUT_PATH as CSV.

Open issues:
    The code needs to be modularized and organized.
    Work more on pre-processing (adding context)
    Spending some time on choosing the best model might be wise.

Resumption:
    generate_new_prompt() (added 2026-09-13) lets a caller — the
    prompt-improvement Streamlit GUI, in particular — try a brand-new prompt
    template against the existing dataset without a full UHRIRecord (a
    long-format CSV row only has flattened text, no country/body/type). See
    its docstring for what that costs in fidelity.

    save_custom_prompt_rows_to_csv() (added 2026-09-13) lets the GUI persist
    a custom prompt's generated rows permanently instead of them living only
    in st.session_state — the prompt's name/template itself is saved
    separately, via prompts.save_custom_prompt_definition() (git-tracked
    custom_prompts.json next to prompts.py), mirroring how a custom metric's
    definition/scores are split across evaluation.py's two save functions.
"""

import random

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from src.Dataloader import load_records
from src.evaluation import readability, score_simplification
from src.prompt_improvement_workflow import normalise_fk_grade
from src.prompts import PROMPTS, DEFAULT_PROMPT, build_prompt
from src.text_utils import slugify
from src.uhri_record import UHRIRecord

load_dotenv()  # loads OPENAI_API_KEY from .env if present

DATA_PATH = "../data/email_recs.xlsx"
MODEL = "gpt-5.5" # Find others at: https://platform.openai.com/docs/models
OUTPUT_PATH = "../data/cleaned.csv"  # tidy/long format, one row per (text, prompt)
FULL_TESTS = True

# Pre-processing knobs (previously done by hand in scripts/analyse_simplification.ipynb)
SAMPLE_SIZE = None   # None = evaluate on all records; set an int to draw a random subset
RANDOM_SEED = 42     # fixes which records get sampled, so runs are reproducible

openai_client = OpenAI()


class _MissingContext(dict):
    """`str.format_map` helper: an unset placeholder renders as a note instead of raising.

    Used by simplify_text() for custom templates run against a long-format
    CSV row, which only has flattened text — no country/recommending_body/type
    to fill {country}/{body}/{type} with, unlike a full UHRIRecord.
    """

    def __missing__(self, key):
        return "(not available for this prompt)"


def _call_openai(prompt: str, model: str) -> str:
    """Send a fully-built prompt to OpenAI and return the generated text."""
    response = openai_client.responses.create(
        model=model,
        input=prompt,
        reasoning={"effort": "medium"},  # "minimal" | "low" | "medium" | "high"
    )
    return response.output_text


def simplify(record: UHRIRecord, model: str = MODEL, template: str = DEFAULT_PROMPT) -> str:
    """
    Simplify a single UHRI record.

    Builds the prompt with build_prompt() and sends it to the given model via
    the OpenAI responses API. No post-processing.

    Args:
        record: The UHRIRecord to simplify.
        model: OpenAI model ID (e.g. "gpt-5.5", "gpt-4o")
        template: Prompt template passed through to build_prompt().
                  Defaults to the module-level DEFAULT_PROMPT.

    Returns:
        The model's generated simplified text.
    """
    return _call_openai(build_prompt(record, template), model)


def simplify_text(text: str, template: str, model: str = MODEL) -> str:
    """
    Simplify raw text with a custom prompt template, without a full UHRIRecord.

    For use when only flattened text is available (e.g. one row of the
    long-format CSV): any {country}/{recommending_body}/{type} placeholder in
    `template` — used by the built-in PROMPTS — renders as
    "(not available for this prompt)" instead of raising, so a template that
    references them still runs; a new template should stick to {text} where
    it can. See simplify() for the full-context version.

    Args:
        text: The original text to simplify.
        template: Prompt template string.
        model: OpenAI model ID.

    Returns:
        The model's generated simplified text.
    """
    prompt = template.format_map(_MissingContext(text=text))
    return _call_openai(prompt, model)


def _build_row(original_text: str, prompt_name: str, conversion: str, metrics: dict) -> dict:
    """Assemble one long-format row from a simplification and its metrics.

    Shared by run_experiment() (the batch CLI pipeline) and
    generate_new_prompt() (the GUI's live "add a prompt" path), so the two
    ways of producing a cleaned.csv-shaped row can't drift apart — in
    particular the double-underscore-to-single-underscore flattening of
    score_simplification()'s ``dim__score``/``dim__reason`` keys.
    """
    row = {"original_text": original_text, "prompt": prompt_name, "transcribed_text": conversion}
    row.update({k.replace("__", "_"): v for k, v in metrics.items()})
    return row


def generate_new_prompt(
    df: pd.DataFrame,
    name: str,
    template: str,
    model: str = MODEL,
    on_progress=None,
) -> pd.DataFrame:
    """Try a new prompt template against every unique text in a long-format DataFrame.

    A long-format DataFrame has one row per (original_text, existing prompt)
    pair, so a text that already appears under several prompts is simplified
    only once here — not once per existing row — to avoid redundant calls.

    Each unique text costs two live API calls: one to OpenAI (simplify_text,
    this module) and one to Anthropic (score_simplification, evaluation.py,
    scoring the standard EVAL_DIMENSIONS). Intended for interactive use
    (the GUI's "add a prompt" form), not a hot loop.

    Args:
        df: Long-format DataFrame with an ``original_text`` column.
        name: Human-readable prompt name; slugified via
            :func:`src.text_utils.slugify` into the new rows' ``prompt`` value.
        template: Prompt template string. See simplify_text() — only {text}
            is guaranteed to be filled; {type}/{body}/{country} (used by the
            built-in PROMPTS) render as a placeholder note instead, since a
            long-format row carries no such context.
        model: OpenAI model ID for simplification.
        on_progress: Optional ``callback(done: int, total: int)``, one call
            per unique text. Called with (0, total) before the first text.

    Returns:
        A new DataFrame: df's rows plus one new row per unique original_text
        (``prompt`` set to the slug), with ``fk_normalized_score`` recomputed
        across the combined set — it's dataset-relative (see
        normalise_fk_grade), so adding rows shifts every row's value, not
        just the new ones. Any metric column df has that isn't part of the
        standard EVAL_DIMENSIONS (e.g. a custom metric added via the GUI's
        "add a metric" form) is NOT populated for the new rows.
    """
    slug = slugify(name)
    unique_texts = df["original_text"].drop_duplicates().tolist()
    total = len(unique_texts)

    if on_progress:
        on_progress(0, total)

    new_rows = []
    for i, text in enumerate(unique_texts, 1):
        conversion = simplify_text(text, template, model=model)
        orig_read = readability(text)
        metrics = score_simplification(text, conversion, orig_read)
        new_rows.append(_build_row(text, slug, conversion, metrics))

        if on_progress:
            on_progress(i, total)

    combined = pd.concat([df, pd.DataFrame(new_rows)], ignore_index=True)
    return normalise_fk_grade(combined)


def save_custom_prompt_rows_to_csv(csv_path: str, rows: list) -> None:
    """Append one custom prompt's generated rows into a CSV, in place.

    Re-reads csv_path fresh from disk — not any in-memory DataFrame — and
    appends `rows` (as produced by generate_new_prompt(), with its derived
    fk_normalized_score column already dropped — that column is
    dataset-relative and gets recomputed fresh on every load, see
    normalise_fk_grade(), so persisting a stale copy would be wrong the
    moment another prompt is added later).

    Args:
        csv_path: Path to the long-format CSV to update in place.
        rows: New rows to append, each a dict shaped like the CSV's
            existing rows (original_text, prompt, transcribed_text, metric
            score/reason columns, ...).

    Raises:
        ValueError: If any row's ``prompt`` value already appears in the
            CSV — appending would create ambiguous duplicate prompt rows.
            Remove the existing ones first (or pick a different name) if
            you actually want to replace them.
    """
    df = pd.read_csv(csv_path)
    new_prompts = {row["prompt"] for row in rows}
    existing_prompts = set(df["prompt"].unique()) if "prompt" in df.columns else set()
    clash = new_prompts & existing_prompts
    if clash:
        raise ValueError(
            f"Prompt(s) {sorted(clash)} already exist in '{csv_path}' — "
            "appending would create duplicate rows. Remove them first, or "
            "use a different name."
        )
    combined = pd.concat([df, pd.DataFrame(rows)], ignore_index=True)
    combined.to_csv(csv_path, index=False)


def select_records(records, sample_size=SAMPLE_SIZE, seed=RANDOM_SEED):
    """
    Returns all records when sample_size is None or larger than the dataset,
    otherwise a random subset of the given size.
    """
    if sample_size is None or sample_size >= len(records):
        return list(records)
    return random.Random(seed).sample(records, sample_size)


def run_experiment(records, prompts=PROMPTS, model=MODEL):
    """
    Run every prompt on every record and return long rows.

    Generates one row per (record, prompt) pair, with each metric as its own column (pivoted columns)
    """
    rows = []
    for record in records:
        print(f"\n{'=' * 60}")
        print(f"{record.recommending_body} to {record.country} [{record.type}]")
        orig_read = readability(record.text)

        for name, template in prompts.items():
            conversion = simplify(record, model=model, template=template)
            print(f"  Finished simplifying with {name}")

            metrics = score_simplification(record, conversion, orig_read)
            print(f"  -> eval: {metrics}")

            rows.append(_build_row(record.text, name, conversion, metrics))

    return rows


if __name__ == "__main__":
    records = load_records(DATA_PATH)

    if not FULL_TESTS:
        record = records[0]
        print("\n Original:")
        print(record.text)
        print("\n Prompt template:")
        print(DEFAULT_PROMPT)
        print("\n Conversion:")
        print(simplify(record))

    else:
        sample = select_records(records)
        print(f"Evaluating {len(sample)} of {len(records)} records "
              f"across {len(PROMPTS)} prompts.")

        rows = run_experiment(sample)
        # One row per (text, prompt), each metric its own column. Same
        # normalise_fk_grade() used everywhere else in pkg3 (the GUI,
        # prompt_improvement_workflow.py, generate_new_prompt() above) —
        # there used to be a separate add_fk_normalized_score() here doing
        # the identical min-max math on a list of dicts instead of a
        # DataFrame; building the DataFrame a few lines earlier instead
        # removed that duplicate.
        df = normalise_fk_grade(pd.DataFrame(rows))
        df.to_csv(OUTPUT_PATH, index=False)

        print(f"Results saved to {OUTPUT_PATH}")

