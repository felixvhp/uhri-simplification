"""
Pkg 3: Evaluation of simplified UHRI texts [last updated 2026-10-02]

Evaluates a simplified text against its original using LLM-as-judge (Anthropic Claude) and
Flesch-Kincaid readability

Usage:
    from src.dashboard.evaluation import evaluate_missing_dimensions, add_readability
    df = add_readability(df)
    df = evaluate_missing_dimensions(df, active_metrics())

Requirements:
    anthropic, textstat, pandas
    ANTHROPIC_API_KEY in a .env file at the repo root or as a shell env var.

Open Issues:
    - smaller models instead one big model

Resumption:
    Metric definitions live in the shared settings (src/settings,
    metrics.json): every entry is just a metric, told apart only by its
    `active` flag. Their keys must stay in sync with EVAL_PROMPT — if you
    add a metric, make sure EVAL_PROMPT still makes sense for it.

    Nothing is judged automatically. `dimensions` is a required argument
    everywhere here: the caller says exactly what to judge (the GUI passes
    what's selected; pass settings.active_metrics() for "everything active").

    evaluate_missing_dimensions() judges one or more dimensions across a
    whole long-format DataFrame, optionally restricted to a row subset
    (e.g. a random sample, so a large dataset can be judged a little at a
    time). A brand-new metric is just the case where every row is a gap,
    since the column doesn't exist yet. save_metric_scores_to_csv()
    persists the computed scores/reasons into the data CSV (data/ is
    gitignored, so that part stays local).
"""

import json
import re
from typing import Optional

import pandas as pd
from dotenv import load_dotenv
import anthropic
import textstat

from ..settings import load_settings

load_dotenv()  # loads ANTHROPIC_API_KEY from .env if present

EVAL_MODEL = load_settings()["dashboard"]["judge_model"]
eval_client: Optional[anthropic.Anthropic] = None  # built on first use, so importing needs no API key


def _get_eval_client() -> anthropic.Anthropic:
    """Return the shared Anthropic client, creating it on first call."""
    global eval_client
    if eval_client is None:
        eval_client = anthropic.Anthropic()
    return eval_client


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
    whatever's currently selected, never automatic.

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

    response = _get_eval_client().messages.create(
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
            the set to judge, nothing implied.
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
    other judge call). A brand-new metric is just the case where every row
    is a gap, since its column doesn't exist yet.

    This is what lets the GUI's "Run missing metrics over full dataset"
    action catch a CSV up to the current, active metrics (e.g. a fresh
    simplification output, or after a new metric was defined) without
    re-billing rows already judged on them.

    Args:
        df: Long-format DataFrame with ``original_text`` and
            ``transcribed_text`` columns.
        dimensions: Mapping of slug -> judge question to check/fill gaps
            for. Typically settings.active_metrics() (to catch up every
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


def save_metric_scores_to_csv(csv_path: str, slug: str, scores: list, reasons: list) -> None:
    """Write one metric's scores/reasons into a CSV as columns.

    Re-reads csv_path fresh from disk — not any in-memory DataFrame — so
    only the file's actual rows are touched. Anything a caller has
    additionally derived in-session (a weighted average, and so on) is
    never written back; this only ever adds or replaces two columns.

    Args:
        csv_path: Path to the long-format CSV to update in place.
        slug: The metric's column-prefix slug.
        scores: One score per row, in the same order as the CSV's rows.
        reasons: One reason per row, same order.

    Raises:
        ValueError: If ``scores``/``reasons`` don't have exactly one entry
            per row currently in the CSV. Writing anyway would silently
            misalign every row after the mismatch.
    """
    df = pd.read_csv(csv_path)
    if len(scores) != len(df) or len(reasons) != len(df):
        raise ValueError(
            f"Have {len(scores)} scores / {len(reasons)} reasons for "
            f"{len(df)} rows in '{csv_path}' — this metric was likely judged "
            "against a different set of rows. Not writing, to avoid "
            "misaligning every row."
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


def add_readability(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``fk_grade``/``fk_grade_delta`` to a long-format DataFrame that lacks them.

    A file straight from the simplification step has only texts, no
    readability numbers — this computes them locally (no API call), so the
    dashboard can score reading-level improvement for any such file.

    Args:
        df: Long-format DataFrame with ``original_text`` and
            ``transcribed_text`` columns.

    Returns:
        df with ``fk_grade`` (of the simplified text) and ``fk_grade_delta``
        (original minus simplified; positive = easier to read). Returned
        unchanged if ``fk_grade_delta`` is already there. Rows missing
        either text get NaN.
    """
    if "fk_grade_delta" in df.columns:
        return df

    def grade(text) -> float:
        return readability(text)["fk_grade"] if isinstance(text, str) and text.strip() else float("nan")

    df["fk_grade"] = df["transcribed_text"].map(grade)
    df["fk_grade_delta"] = df["original_text"].map(grade) - df["fk_grade"]
    return df
