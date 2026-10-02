"""
Pkg 3: Evaluation of simplified UHRI texts [last updated 2026-05-27]
 
Evaluates a simplified text against its original using LLM-as-judge (Anthropic Claude) and
Flesch-Kincaid readability
 
Usage:
    from src.evaluation import readability, score_simplification
    orig_read = readability(record.text)
    metrics   = score_simplification(record.text, simplified, orig_read)
 
Requirements:
    anthropic, textstat, pandas

Open Issues:
    - smaller models instead one big model

Resumption:
    All metric definitions — standard ones AND any custom ones added via
    the GUI — live together in one file: metrics.json, next to this module
    (git-tracked, so it's shared with anyone who pulls the repo). An
    entry's `standard: true/false` flag is what tells the two apart, not
    which file it's in. `date_added` is null for entries that predate the
    field and an ISO date for every one added since. EVAL_DIMENSIONS is
    derived from the file at import time (see load_all_metric_definitions())
    and must stay in sync with the JSON keys in EVAL_PROMPT — if you add a
    standard dimension, add it to metrics.json with `standard: true` and
    update EVAL_PROMPT here.

    A dimension retired from the standard set (e.g. the original 5, retired
    when the judge was rewritten to the current 9 fine-grained ones) is
    never deleted from metrics.json — it's marked `"active": false`
    instead, via DefinitionsStore.set_active(). EVAL_DIMENSIONS filters to
    `standard AND active`, and the GUI's metric list (see
    prompt_improvement_app.py's _get_available_metric_columns()) filters on
    that exact same `active` flag read from this same file, so "what the
    judge scores" and "what the GUI shows as available" can't silently
    drift apart from each other the way they once did.

    evaluate_new_metric() (added 2026-09-13) lets a caller — the
    prompt-improvement Streamlit GUI, in particular — judge one ad-hoc
    dimension across a whole long-format DataFrame without touching
    EVAL_DIMENSIONS. It reuses evaluate()/build_eval_prompt() by passing a
    one-entry `dimensions` dict, so the standard five and a custom one share
    the same judge machinery and never drift apart.

    load_all_metric_definitions()/load_custom_metric_definitions()/
    save_custom_metric_definition() (added 2026-09-13, reshaped into one
    unified metrics.json 2026-09-18, delegated to the shared
    DefinitionsStore 2026-09-18) let the GUI persist a custom metric
    permanently instead of it living only in st.session_state: the
    definition (name + judge question + date_added) is merged into
    metrics.json alongside the standard 5 — save_custom_metric_definition()
    always reads and rewrites the *whole* file, never just the custom
    subset, so saving a new custom metric can never accidentally drop the
    standard ones (see definitions_store.py for the shared implementation;
    prompts.py's equivalent trio for prompts.json uses the exact same
    class). save_custom_metric_scores_to_csv() persists the computed
    scores/reasons themselves, into the data CSV instead (data/ is
    gitignored, so that part stays local). Once saved, the metric is just
    an ordinary CSV column on the next load; nothing special treats it as
    "custom" anymore.
"""

import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
import anthropic
import textstat

from src.definitions_store import DefinitionsStore
from src.text_utils import slugify

# Where ALL metric definitions live — the 5 standard ones and any custom
# ones, together. Next to this file (not relative to the cwd), tracked in
# git so a saved metric's judge question is shared with anyone who pulls
# the repo. The *scores* for a custom metric go into the data CSV instead
# (data/ is gitignored — see save_custom_metric_scores_to_csv()), since
# those are per-dataset, not source.
METRICS_PATH = Path(__file__).parent / "metrics.json"
_metrics_store = DefinitionsStore(METRICS_PATH, content_field="description")


def load_all_metric_definitions(path: Optional[Path] = None) -> dict:
    """Load every metric definition — standard AND custom, active AND retired.

    See DefinitionsStore.load_all() for the full contract; this is a thin
    wrapper so callers keep using `from src.evaluation import
    load_all_metric_definitions` rather than reaching into the store
    directly. Use load_active_metric_definitions() instead if you only want
    the ones currently in use.
    """
    return _metrics_store.load_all(path)


def load_active_metric_definitions(path: Optional[Path] = None) -> dict:
    """Load every ACTIVE metric definition — standard and custom, together.

    See DefinitionsStore.load_active() for the full contract. This is what
    EVAL_DIMENSIONS and the GUI's metric list both filter on, so neither
    can show a retired metric the other has stopped scoring.
    """
    return _metrics_store.load_active(path)


def set_metric_active(slug: str, active: bool, path: Optional[Path] = None) -> None:
    """Retire or restore one metric definition in place (see DefinitionsStore.set_active()).

    Use this instead of deleting an entry from metrics.json — its
    description and date_added survive, and it can be reactivated later by
    calling this again with active=True.
    """
    _metrics_store.set_active(slug, active, path)


load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present
 
EVAL_MODEL = "claude-sonnet-4-5"
eval_client = anthropic.Anthropic()

EVAL_PROMPT = """\
You are evaluating whether a simplified version of a UN human rights text is accurate and accessible.
 
Carefully compare the original and simplified texts. Be strict: if anything important is missing, changed, or added incorrectly, mark as false.
 
Original data:
{original}
 
Simplified text:
{simplified}
 
Evaluate on these dimensions:
{explanations}
 
Respond with ONLY a JSON object in exactly this format, with no preamble, no markdown fences, no explanation outside the JSON:
 
{{
    {eval_dimensions}
}}
 
The "score" field must be a JSON boolean (true or false), never a string. The "reason" field is one short sentence explaining the score.\
"""
 
# The subset of metrics.json marked `standard: true` AND `active: true` —
# flattened to slug -> description, which is the shape
# build_eval_prompt()/evaluate() (below) and every existing caller/test
# expect. This is what gets judged on *every* simplification by default; a
# custom (standard: false) metric in the same file is judged only on
# demand, via evaluate_new_metric(). A standard metric retired via
# set_metric_active(slug, False) stays in metrics.json (so its description/
# history survives) but drops out of here and out of the GUI's metric list
# at the same time, since both read the same `active` flag from the same
# file.
#
# The 9 fine-grained dimensions below replaced the original 5 (now
# retired, active: false, still in metrics.json) — they live in
# metrics.json like every other standard entry, rather than as a hardcoded
# dict, so they go through the same load/save path as a custom metric.
EVAL_DIMENSIONS = {
    slug: entry["description"]
    for slug, entry in load_all_metric_definitions().items()
    if entry.get("standard") and entry.get("active", True)
}
 
def evaluate(
    raw_record,
    simplified: str,
    model: str = EVAL_MODEL,
    dimensions: Optional[dict] = None,
) -> dict:
    """
    Evaluate a simplified text against its original with an LLM judge.

    Args:
        original: The original source text, and associated meta data.
        simplified: The simplified output to evaluate.
        model: Anthropic model name for the judge.
        dimensions: Mapping of dimension name to judge question. Defaults to
            EVAL_DIMENSIONS (the standard 5). Pass a one-entry dict to judge
            a single custom dimension in isolation — e.g. evaluate_new_metric()
            does this so an ad-hoc metric doesn't re-score the standard five.

    Returns:
        Dict keyed by dimension name.
        Score conventions: 1 = yes/pass, 0 = no/fail, -1 = parse error.
    """
    dimensions = dimensions or EVAL_DIMENSIONS

    prompt = build_eval_prompt(raw_record, simplified, dimensions)

    response = eval_client.messages.create(
        model=model,
        max_tokens=600,
        messages=[{"role": "user", "content": prompt}],
    )

    raw = response.content[0].text.strip()

    # in case llm dumb and adds stuff around the JSON brackets
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end > start:
        raw = raw[start : end + 1]

    # Parse JSON
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        print(f"  Failed to parse judge response as JSON. Raw output:\n{raw}")
        parsed = {}

    # ensure every expected dimension is present, give score 1 or 0, -1 if error
    result = {}
    for dim in dimensions:
        entry = parsed.get(dim)
        if not isinstance(entry, dict) or "score" not in entry:
            result[dim] = {"score": -1, "reason": "missing"}
            continue

        score_val = entry["score"]
        if score_val is True:
            score_int = 1
        elif score_val is False:
            score_int = 0
        else:
            score_int = -1

        result[dim] = {"score": score_int, "reason": entry.get("reason", "reason missing")}

    return result

def build_eval_prompt(
    raw_record,
    simplified: str,
    dimensions: Optional[dict] = None,
):
    """
    Build the prompt for evaluating the query.

    params:
        raw_record: the original record loaded from the database with all the context.
        simplified: the simplified record.
        dimensions: Mapping of dimension name to judge question. Defaults to
            EVAL_DIMENSIONS.
    """
    dimensions = dimensions or EVAL_DIMENSIONS
    original = str(raw_record)

    # the numerator simply counts out the list.
    numerator = 1
    formated_dim_explanations = ""
    dim_explanations_template = "{num}. {dim}: {desc} \n"


    formated_eval_dimensions = ""
    for dim, description in dimensions.items():
        # The explanation of each item.
        explanation = dim_explanations_template.format(
            num = numerator,
            dim = dim,
            desc = description
        )
        numerator += 1
        formated_dim_explanations += explanation

        # The json format template for the evaluation.
        formated_eval = '"%s": {"reason": "<one short sentence>", "score": true}, \n' % dim
        formated_eval_dimensions += formated_eval


    prompt = EVAL_PROMPT.format(original=original,
                                simplified=simplified,
                                explanations = formated_dim_explanations,
                                eval_dimensions = formated_eval_dimensions
                               )

    return prompt


def evaluate_new_metric(
    df: pd.DataFrame,
    name: str,
    description: str,
    model: str = EVAL_MODEL,
    on_progress=None,
) -> pd.DataFrame:
    """Judge every row of a long-format DataFrame on one new custom dimension.

    Makes one Anthropic API call per row — this is a live, billed, slow
    operation (roughly one judge call's latency times ``len(df)``), unlike
    the rest of prompt_improvement_workflow.py, which is instant local
    pandas math. Intended for interactive use (e.g. the GUI's "add a
    metric" form), not a hot loop.

    Args:
        df: Long-format DataFrame with ``original_text`` and
            ``transcribed_text`` columns (see prompt_improvement_workflow.py).
        name: Human-readable metric name; slugified via
            :func:`src.text_utils.slugify` into the new columns' prefix.
        description: The judge question for this dimension, e.g. "Is the
            tone appropriately formal for a legal audience?"
        model: Anthropic model name for the judge.
        on_progress: Optional ``callback(done: int, total: int)`` invoked
            after each row — e.g. to drive a progress bar in a GUI. Called
            with (0, total) before the first row so callers can render an
            initial 0% state.

    Returns:
        df with two new columns added, ``f"{slug}_score"`` and
        ``f"{slug}_reason"`` (modifies in place and returns).
    """
    slug = slugify(name)
    dimensions = {slug: description}
    total = len(df)

    if on_progress:
        on_progress(0, total)

    scores, reasons = [], []
    for i, row in enumerate(df.itertuples(), 1):
        result = evaluate(row.original_text, row.transcribed_text, model=model, dimensions=dimensions)
        scores.append(result[slug]["score"])
        reasons.append(result[slug]["reason"])
        if on_progress:
            on_progress(i, total)

    df[f"{slug}_score"] = scores
    df[f"{slug}_reason"] = reasons
    return df


def load_custom_metric_definitions(path: Optional[Path] = None) -> dict:
    """Load saved CUSTOM metric definitions only — the standard 5 excluded.

    "Custom" means anything in metrics.json *not* marked `standard: true` —
    i.e. whatever's been added via the GUI's "Add a new metric" form. Use
    load_all_metric_definitions() instead if you need the standard ones too.
    See DefinitionsStore.load_custom() for the full contract.
    """
    return _metrics_store.load_custom(path)


def save_custom_metric_definition(
    slug: str,
    name: str,
    description: str,
    path: Optional[Path] = None,
    date_added: Optional[str] = None,
) -> None:
    """Persist one custom metric's definition (name + judge question).

    See DefinitionsStore.save_custom() for the full contract, in particular
    why this can never accidentally drop the standard 5 sharing the file.

    Args:
        slug: The metric's column-prefix slug (see text_utils.slugify).
        name: Human-readable name, as typed in the GUI.
        description: The judge question.
        path: JSON file to write. Defaults to METRICS_PATH.
        date_added: ISO date string (YYYY-MM-DD). Defaults to today — the
            normal case, a fresh save from the GUI. Pass this explicitly
            only to backfill an older entry.
    """
    _metrics_store.save_custom(slug, name, description, path, date_added)


def save_custom_metric_scores_to_csv(csv_path: str, slug: str, scores: list, reasons: list) -> None:
    """Write one custom metric's scores/reasons into a CSV as new columns.

    Re-reads csv_path fresh from disk — not any in-memory DataFrame — so
    only the file's actual base rows are touched. Anything a caller has
    additionally derived in-session (a weighted average, generated-prompt
    rows appended in memory, and so on) is never written back; this only
    ever adds two columns.

    Args:
        csv_path: Path to the long-format CSV to update in place.
        slug: The metric's column-prefix slug.
        scores: One score per row, in the same order as the CSV's rows.
        reasons: One reason per row, same order.

    Raises:
        ValueError: If ``scores``/``reasons`` don't have exactly one entry
            per row currently in the CSV — e.g. because the metric was
            judged while extra rows (from a generated prompt) were also in
            play. Writing anyway would silently misalign every row after
            the mismatch.
    """
    df = pd.read_csv(csv_path)
    if len(scores) != len(df) or len(reasons) != len(df):
        raise ValueError(
            f"Have {len(scores)} scores / {len(reasons)} reasons for "
            f"{len(df)} rows in '{csv_path}' — this metric was likely judged "
            "against a different set of rows (e.g. with generated prompts "
            "active). Not writing, to avoid misaligning every row."
        )
    df[f"{slug}_score"] = scores
    df[f"{slug}_reason"] = reasons
    df.to_csv(csv_path, index=False)

 

def readability(text: str) -> dict[str, float]:
    """Flesch-Kincaid grade and ease, after normalizing for legalese formatting."""
    text = text.replace(";", ".")
    text = re.sub(r"\([a-z]\)\s*", "", text) 
    text = re.sub(r"^\d+\.\s*", "", text)  
    return {
        "fk_grade": textstat.flesch_kincaid_grade(text),
        "fk_ease": textstat.flesch_reading_ease(text),
    }
 
def score_simplification(
    raw_record,
    simplified: str,
    orig_readability: dict[str, float],
) -> dict:
    """
    Compute all evaluation metrics for one simplified output.
 
    Bundles the judge scores and readability deltas into a single flat dict
    so the caller doesn't have to assemble them by hand.
 
    Args:
        original: The original source text.
        simplified: The simplified output to score.
        orig_readability: Readability dict for the original text 
 
    Returns:
        Flat dict of metrics
    """
    simp_read = readability(simplified)
    scores = evaluate(raw_record, simplified)
 
    out = {
        "fk_grade": simp_read["fk_grade"],
        # "fk_ease": simp_read["fk_ease"],
        "fk_grade_delta": orig_readability["fk_grade"] - simp_read["fk_grade"],
    }
    for dim, entry in scores.items():
        out[f"{dim}__score"] = entry["score"]
        out[f"{dim}__reason"] = entry["reason"]

    return out

