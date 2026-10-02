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
    Every metric definition — the ones judged by default and any added via
    the GUI — lives in one file: metrics.json, next to this module
    (git-tracked, so it's shared with anyone who pulls the repo). There's no
    "standard vs custom" flag: every entry is just a metric, told apart only
    by its `active: true/false` flag (see DefinitionsStore for the full
    shape). active_metric_definitions() reads the file fresh on every call
    (never cached), flattened to slug -> description — this is what
    evaluate()/score_simplification() judge by default when no explicit
    `dimensions` is given. Its JSON keys must stay in sync with EVAL_PROMPT
    — if you add a metric meant to be judged by default, add it to
    metrics.json and make sure EVAL_PROMPT still makes sense for it.

    Retiring a metric (`set_metric_active(slug, False)`) never deletes it:
    its description survives, it stops being judged on any *new*
    simplification, and an already-computed CSV column for it still
    displays normally in the GUI. metrics.json is the GUI's ground truth
    for which metrics exist at all — the GUI seeds an empty column for
    every *active* entry a loaded CSV doesn't already have, but never does
    that for a retired one, which only reappears if the CSV already has
    real data for it.

    A caller that wants something other than "every currently active
    metric" — e.g. the GUI threading through whatever's currently toggled
    on in its own "Active metrics" selection — builds its own `dimensions`
    dict and passes it explicitly; active_metric_definitions() is only the
    zero-config fallback for a caller (the CLI script, in particular) that
    has no more specific selection of its own.

    evaluate_missing_dimensions() judges one or more dimensions across a
    whole long-format DataFrame, optionally restricted to a row subset
    (e.g. a random sample, so a large dataset can be judged a little at a
    time). It reuses evaluate()/build_eval_prompt() by passing whatever
    `dimensions` dict is needed, so every metric shares the same judge
    machinery and never drifts apart — a brand-new metric is just the case
    where every row is a gap, since the column doesn't exist yet.

    load_all_metric_definitions()/save_custom_metric_definition() let the
    GUI persist a new metric permanently instead of it living only in
    st.session_state: the definition (name + judge question) is merged into
    metrics.json — save_custom_metric_definition() always reads and
    rewrites the *whole* file, never a subset, so saving a new metric can
    never accidentally drop any other entry (see definitions_store.py for
    the shared implementation; prompts.py's equivalent trio for
    prompts.json uses the exact same class). save_custom_metric_scores_to_csv()
    persists the computed scores/reasons themselves, into the data CSV
    instead (data/ is gitignored, so that part stays local). Once saved, a
    metric is just an ordinary CSV column on the next load.
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

# Where ALL metric definitions live — the standard ones and any custom
# ones, together. Next to this file (not relative to the cwd), tracked in
# git so a saved metric's judge question is shared with anyone who pulls
# the repo. The *scores* for a custom metric go into the data CSV instead
# (data/ is gitignored — see save_custom_metric_scores_to_csv()), since
# those are per-dataset, not source.
METRICS_PATH = Path(__file__).parent / "metrics.json"
_metrics_store = DefinitionsStore(METRICS_PATH, content_field="description", include_standard=False)


def load_all_metric_definitions(path: Optional[Path] = None) -> dict:
    """Load every metric definition, active AND retired.

    See DefinitionsStore.load_all() for the full contract; this is a thin
    wrapper so callers keep using `from src.evaluation import
    load_all_metric_definitions` rather than reaching into the store
    directly. Use active_metric_definitions() instead if you only want the
    ones currently judged by default.
    """
    return _metrics_store.load_all(path)


def active_metric_definitions(path: Optional[Path] = None) -> dict:
    """Load the currently-active metric definitions, flattened to slug -> description.

    Read fresh on every call (never cached at import time), so retiring or
    adding a metric takes effect on the very next call. This is what
    evaluate()/score_simplification() judge by default when no explicit
    `dimensions` is given — a caller with a more specific selection of its
    own (e.g. the GUI threading through whatever's currently toggled active
    in its "Active metrics" dropdown) passes that instead.
    """
    return {
        slug: entry["description"]
        for slug, entry in load_all_metric_definitions(path).items()
        if entry.get("active", True)
    }


def set_metric_active(slug: str, active: bool, path: Optional[Path] = None) -> None:
    """Retire or restore one metric definition in place (see DefinitionsStore.set_active()).

    Use this instead of deleting an entry from metrics.json — its
    description survives, and it can be reactivated later by calling this
    again with active=True.
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
 
def evaluate(
    raw_record,
    simplified: str,
    dimensions: dict,
    model: str = EVAL_MODEL,
) -> dict:
    """
    Evaluate a simplified text against its original with an LLM judge.

    There's no implicit "judge everything active" fallback: judging only
    ever happens when a caller explicitly says which dimensions to judge —
    in the GUI, that's always a user-triggered action (a button press) over
    whatever's currently selected, never automatic. Pass
    active_metric_definitions() yourself if you specifically want "every
    currently-active metric" (e.g. the CLI script's batch pipeline does).

    Args:
        original: The original source text, and associated meta data.
        simplified: The simplified output to evaluate.
        dimensions: Mapping of dimension name to judge question — exactly
            the set to judge, nothing implied beyond it.
        model: Anthropic model name for the judge.

    Returns:
        Dict keyed by dimension name.
        Score conventions: 1 = yes/pass, 0 = no/fail, -1 = parse error.
    """

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
    dimensions: dict,
):
    """
    Build the prompt for evaluating the query.

    params:
        raw_record: the original record loaded from the database with all the context.
        simplified: the simplified record.
        dimensions: Mapping of dimension name to judge question — exactly
            the set to judge, nothing implied. Pass active_metric_definitions()
            yourself for "every currently-active metric".
    """
    original = str(raw_record)

    explanations = "\n".join(
        f"{i}. {dim}: {desc} " for i, (dim, desc) in enumerate(dimensions.items(), 1)
    )
    eval_dimensions = "\n".join(
        f'"{dim}": {{"reason": "<one short sentence>", "score": true}}, ' for dim in dimensions
    )

    return EVAL_PROMPT.format(
        original=original, simplified=simplified,
        explanations=explanations, eval_dimensions=eval_dimensions,
    )


def evaluate_missing_dimensions(
    df: pd.DataFrame,
    dimensions: dict,
    model: str = EVAL_MODEL,
    on_progress=None,
    rows: Optional[list] = None,
) -> pd.DataFrame:
    """Judge whichever of ``dimensions`` a row doesn't already have a score for.

    A row already scored on every one of ``dimensions`` costs nothing; a
    brand-new or partially-scored row costs exactly one Anthropic call
    covering just that row's gaps (whichever of ``dimensions`` are missing
    for it, scored together via evaluate()/build_eval_prompt() like any
    other judge call). A brand-new custom metric is just the case where
    every row is a gap, since its column doesn't exist yet.

    This is what lets the GUI's "Run missing metrics over full dataset"
    action catch a CSV up to the current, active metrics.json entries
    (e.g. after a judge-prompt rewrite added new standard dimensions, or a
    custom metric was defined before this CSV existed) without re-billing
    rows already judged on them.

    Args:
        df: Long-format DataFrame with ``original_text`` and
            ``transcribed_text`` columns — only those two, no full
            UHRIRecord needed.
        dimensions: Mapping of slug -> judge question to check/fill gaps
            for. Typically active_metric_definitions() (to catch up every
            active metric), or a single ``{slug: description}`` for one
            specific metric.
        model: Anthropic model name for the judge.
        on_progress: Optional ``callback(done: int, total: int)`` — total
            counts only rows that actually need a call, so it's 0
            immediately if every row is already fully judged.
        rows: Restrict which row labels are even considered — e.g. a random
            subset picked in the GUI so a huge dataset can be judged a
            little at a time instead of all at once. Defaults to every row
            in ``df``. A row in here that's already fully judged still
            costs nothing (the gap check applies either way); a row NOT in
            here is left untouched even if it has gaps.

    Returns:
        df with a ``f"{slug}_score"``/``f"{slug}_reason"`` column for every
        slug in ``dimensions``, filled for every row in ``rows`` that had a
        gap (modifies in place and returns). A slug with no existing column
        starts as NaN for every row before being filled in — including rows
        outside ``rows``, which then stay NaN (still correctly "not yet
        judged" rather than missing the column entirely).
    """
    for slug in dimensions:
        if f"{slug}_score" not in df.columns:
            df[f"{slug}_score"] = float("nan")
            df[f"{slug}_reason"] = None

    candidates = df.index if rows is None else rows
    rows_needing_work = [
        i for i in candidates if any(pd.isna(df.at[i, f"{slug}_score"]) for slug in dimensions)
    ]
    total = len(rows_needing_work)

    if on_progress:
        on_progress(0, total)

    for done, i in enumerate(rows_needing_work, 1):
        missing = {slug: desc for slug, desc in dimensions.items() if pd.isna(df.at[i, f"{slug}_score"])}
        result = evaluate(df.at[i, "original_text"], df.at[i, "transcribed_text"], model=model, dimensions=missing)
        for slug, entry in result.items():
            df.at[i, f"{slug}_score"] = entry["score"]
            df.at[i, f"{slug}_reason"] = entry["reason"]
        if on_progress:
            on_progress(done, total)

    return df


def save_custom_metric_definition(
    slug: str,
    name: str,
    description: str,
    path: Optional[Path] = None,
) -> None:
    """Persist one metric's definition (name + judge question) to metrics.json.

    See DefinitionsStore.save_custom() for the full contract, in particular
    why this can never accidentally drop any other entry sharing the file.

    Args:
        slug: The metric's column-prefix slug (see text_utils.slugify).
        name: Human-readable name, as typed in the GUI.
        description: The judge question.
        path: JSON file to write. Defaults to METRICS_PATH.
    """
    _metrics_store.save_custom(slug, name, description, path)


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
    dimensions: dict,
) -> dict:
    """
    Compute all evaluation metrics for one simplified output.

    Bundles the judge scores and readability deltas into a single flat dict
    so the caller doesn't have to assemble them by hand.

    Args:
        raw_record: The original record (and metadata) to evaluate against.
        simplified: The simplified output to score.
        orig_readability: Readability dict for the original text
        dimensions: Mapping of dimension name to judge question — passed
            straight through to evaluate(). No implicit default: the caller
            decides what gets judged (e.g. simplification.py's CLI pipeline
            passes active_metric_definitions(); the GUI passes whatever is
            currently selected).

    Returns:
        Flat dict of metrics
    """
    simp_read = readability(simplified)
    scores = evaluate(raw_record, simplified, dimensions)

    out = {
        "fk_grade": simp_read["fk_grade"],
        "fk_grade_delta": orig_readability["fk_grade"] - simp_read["fk_grade"],
    }
    for dim, entry in scores.items():
        out[f"{dim}__score"] = entry["score"]
        out[f"{dim}__reason"] = entry["reason"]

    return out

