"""
Pkg 3: Prompt improvement workflow — Streamlit GUI [last updated 2026-09-13]

Usage:
    cd code
    streamlit run src/prompt_improvement_app.py

    Reads the same long-format CSV as prompt_improvement_workflow.py
    (default data/cleaned.csv, editable in the sidebar) and walks through the
    same four-step evaluation report interactively in the browser instead of
    via CLI `input()` prompts.

Deliberately thin: this file is UI only. Every number it shows comes from
calling prompt_improvement_workflow.py's step functions with verbose=False,
which hands back plain DataFrames/dicts instead of printed text — see that
module's "Design note" docstring section. If the underlying computation
ever needs to change (new metric, new threshold logic, new step), change it
there once; both this GUI and the CLI script pick it up automatically.

Both the "Metrics" and "Prompts" sidebar sections follow the same four-part
pattern — tweak / remove / view / add — for the same reason: a metric and a
prompt are the two "dimensions" a user compares things along, so their
controls are deliberately symmetric. Where the two sides were byte-for-byte
identical logic (only "remove", via _select_active_items — see its
docstring) they share one function; where they differ enough to matter
(_render_add_metric_form / _render_add_prompt_form, "add") they stay two
mirrored functions rather than being forced into one with a pile of
conditionals.

    - Tweak: a weight slider per active metric (metrics only — a prompt
      doesn't have a "weight" the way a metric does; a prompt is a thing
      being compared, not a thing that's combined with others).
    - Remove: the "Active metrics"/"Active prompts" dropdown hides an item
      from the weighted average / variance chart / prompt comparison
      without touching the underlying data — reversible any time by
      reselecting it. The Metric Variance and Prompt Comparison tabs'
      flagged-item expanders also get a one-click "Remove" button wired to
      the same selection.
    - View: "📋 View metric description" / "📋 View prompt template" — pick
      one from a dropdown and see its judge question or full template text.
      Pure inspection, meant to make the tweak/remove/add decisions above
      easier without having to go read evaluation.py/prompts.py.
    - Add: an "➕ Add a new ..." expander sits directly under each dropdown.
      Both forms call a live LLM: "Add a new metric" judges a new dimension
      once per row (evaluation.py); "Add a new prompt" simplifies *and*
      judges once per unique text (simplification.py) — double the API
      calls, across two providers. Both are gated behind an explicit
      confirmation checkbox, and both always process the full dataset
      rather than a sample, so every metric/prompt stays fully populated
      and the rest of the workflow never has to deal with partial coverage
      or NaN-handling. Added items live only in st.session_state (see
      _reattach_custom_metrics/_reattach_custom_prompts) until a "💾 Save"
      button persists them (_render_save_metrics_section /
      _render_save_prompts_section): the definition (judge question / prompt
      template) is merged into metrics.json / prompts.json, next to
      evaluation.py / prompts.py — one unified file per domain holding both
      the standard definitions and any custom ones (an `standard: true/false`
      flag tells them apart, not which file they're in), git-tracked so it's
      shared with anyone who pulls the repo. The actual data — computed
      scores as new columns, generated rows appended — is written into the
      CSV itself instead (data/ is gitignored, so that part stays local).
      Once saved, a metric or prompt is just ordinary CSV content on the
      next load — nothing "custom" about it anymore. Unsaved items disappear
      if the CSV path, mtime, or row count changes underneath them.

Settings persistence (weights, active metrics/prompts, thresholds, etc.):
    A "💾 Save current settings" button up top (_save_settings/_load_settings)
    writes every sidebar knob — Samples to show, Auto-format toggle, both
    variance thresholds, which metrics/prompts are active, and every weight
    slider — into a `<csv name>.settings.json` sidecar next to the CSV. This
    is a *personal workspace preference*, not a shared research artifact
    like a saved metric/prompt definition, so — unlike metrics.json/
    prompts.json — it lives under data/ (gitignored) rather than next to
    evaluation.py/prompts.py: never git-tracked, never shared.
    Loaded back automatically the moment this CSV is opened, but only ever
    *seeds* each widget's session_state once per session (the same pattern
    active_metrics/active_prompts already use) — a saved value never
    clobbers a change you've since made live. active_metrics/active_prompts
    each save their "known" universe alongside the active list, for the
    same reason _select_active_items tracks known_<kind>: to tell "this
    metric existed and I deliberately excluded it" apart from "this metric
    didn't exist yet when I saved, default it to included".

Open Issues:
    - Single-file/single-tab layout. Fine at today's scale (4 steps, one CSV).
    - Saving a prompt appends rows, changing the CSV's row count — so it
      also retires any *other* still-unsaved custom metric or prompt (their
      stored base row count no longer matches). Saving a metric only adds
      columns, so it doesn't have this side effect on other pending items.
      Save one thing at a time if you have several pending, to be safe.
    - No "reset to defaults" button for settings yet — only save/auto-load.
    - Settings only reliably auto-load for whatever CSV is at
      DEFAULT_DATA_PATH (i.e. whatever's already showing the moment the
      session starts) — every "seed once" setting (n_samples,
      auto_format_original, both thresholds, every weight, active_metrics,
      active_prompts) shares the same "only seeds if the session_state key
      doesn't already exist" guard, seeded against whatever data_path was
      showing on that very first render. Typing in a *different* CSV path
      mid-session won't re-seed anything from that other file's sidecar —
      confirmed while testing this feature, not new to it: switching paths
      mid-session already didn't reset active_metrics/known_metrics before
      settings persistence existed. Properly fixing it means detecting a
      data_path change and re-running every seeding step (and clearing the
      now-stale session_state) — worth doing if multi-CSV-per-session
      becomes a real workflow; skipped for now since one CSV per session is
      today's actual usage.

Resumption — how to scale this up when it outgrows a single file:
    - New step: write one `render_<step>_tab(...)` function below and add one
      more entry to the `STEPS` list. Nothing else needs to change.
    - New metric in the CSV: no code change needed — weight sliders and
      variance tables are generated from `get_score_columns(df)`, so a new
      `*_score` column just shows up. This includes a metric retired from
      metrics.json (`active: false`, see evaluation.py's docstring): its
      already-computed column keeps showing here with its description
      intact (metric_descriptions below reads ALL of metrics.json, not
      just the active subset) — `active` only gates whether it's part of
      EVAL_DIMENSIONS for a *new* judging run, not whether existing data
      for it is still viewable/selectable here.
    - Growing past a handful of steps/tabs: split each render_* function out
      into its own module under src/gui/ and import them here; `st.tabs` can
      be swapped for `st.navigation`/multipage app structure without touching
      the workflow module.
    - Multiple people using it at once / needing persistence across runs:
      move `st.session_state` usage to a small results cache keyed by
      (data path, weights, thresholds) so re-runs don't require re-picking
      the same sliders.
"""

import json
import sys
from pathlib import Path

# Same trick as prompt_improvement_workflow.py: when run as
# `streamlit run src/prompt_improvement_app.py` from code/, we need code/ on
# sys.path so `from src...` imports resolve.
sys.path.insert(0, str(Path(__file__).parent.parent))

import altair as alt
import pandas as pd
import streamlit as st

from src.prompt_improvement_workflow import (
    LOW_VARIANCE_THRESHOLD,
    HIGH_VARIANCE_THRESHOLD,
    N_SAMPLES,
    get_score_columns,
    normalise_fk_grade,
    compute_weighted_average,
    check_metric_variance,
    show_sanity_samples,
    compare_prompts,
    get_high_variance_samples,
    verify_best_prompt,
)
from src.prompts import load_all_prompt_definitions, load_custom_prompt_definitions, save_custom_prompt_definition
from src.text_utils import format_legal_bullets, slugify

# evaluation.py and simplification.py each build a live API client (Anthropic,
# OpenAI respectively) at import time, and raise if the matching key isn't
# configured. The rest of this GUI (weights, removing metrics, the four
# report tabs) needs neither, so both are imported defensively — each import
# failure only disables its one "add a new ..." form, not the whole app.
# prompts.py's prompts.json loading (above) has no such dependency, so it's
# imported unconditionally — the "view prompt template" panel works either way.
# EVAL_DIMENSIONS/metrics.json loading live in evaluation.py (both standard
# and custom metric definitions are read from the same file — see that
# module's docstring), so "view metric description" needs a working
# Anthropic client too, same as everything else behind LLM_JUDGE_AVAILABLE.
try:
    from src.evaluation import (
        evaluate_new_metric,
        load_all_metric_definitions,
        load_custom_metric_definitions,
        save_custom_metric_definition,
        save_custom_metric_scores_to_csv,
        EVAL_MODEL,
    )
    LLM_JUDGE_AVAILABLE = True
    LLM_JUDGE_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - deliberately broad, see comment above
    LLM_JUDGE_AVAILABLE = False
    LLM_JUDGE_IMPORT_ERROR = str(e)

try:
    from src.simplification import generate_new_prompt, save_custom_prompt_rows_to_csv, MODEL as SIMPLIFY_MODEL
    PROMPT_GENERATION_AVAILABLE = True
    PROMPT_GENERATION_IMPORT_ERROR = None
except Exception as e:  # noqa: BLE001 - deliberately broad, see comment above
    PROMPT_GENERATION_AVAILABLE = False
    PROMPT_GENERATION_IMPORT_ERROR = str(e)

DEFAULT_DATA_PATH = "../data/cleaned.csv"

# Metrics that get a higher default weight slider — readability is what the
# end user actually experiences, so it's weighted above the other (mostly
# binary pass/fail) judged dimensions out of the box. Purely a starting
# point; every slider is still adjustable.
_READABILITY_METRICS = ("fk_normalized_score", "accessible_and_simple_score")


# ── Settings persistence (weights, in/excluded metrics & prompts, thresholds) ─
# Unlike a saved metric/prompt definition (a shared research artifact, so it
# goes in code/src/*.json — git-tracked), these are personal workspace
# preferences: which items you like excluded, how you've tuned the weights.
# They live in a JSON sidecar next to the CSV itself, under data/ (already
# gitignored), so saving/loading them never touches git.


def _settings_path(data_path: str) -> Path:
    """Where this dataset's saved GUI settings live: `<csv name>.settings.json`,
    next to the CSV itself."""
    p = Path(data_path)
    return p.parent / f"{p.stem}.settings.json"


def _load_settings(data_path: str) -> dict:
    """Load this dataset's saved GUI settings. Empty dict if none saved yet."""
    path = _settings_path(data_path)
    if not path.exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _save_settings(data_path: str, settings: dict) -> None:
    """Persist the current GUI settings as this dataset's sidecar JSON."""
    with open(_settings_path(data_path), "w") as f:
        json.dump(settings, f, indent=2, sort_keys=True)
        f.write("\n")


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading data…")
def load_data(path: str, mtime: float) -> pd.DataFrame:
    """Load and score the long-format CSV.

    Args:
        path: Path to the long-format CSV (see prompt_improvement_workflow.py).
        mtime: File modification time. Deliberately part of the cache key (no
            leading underscore) so editing the CSV on disk — e.g. re-running
            the LLM judge on more rows — invalidates the cache automatically;
            without it, Streamlit would keep serving the first-loaded copy
            for the rest of the session.

    Returns:
        DataFrame with ``fk_normalized_score`` added.
    """
    df = pd.read_csv(path)
    return normalise_fk_grade(df)


# ── Metric management: tweak (weights), remove (active-metric toggle), ──────
# ── add (live LLM judge) ──────────────────────────────────────────────────────


def _progress_callback(progress_widget, unit: str):
    """Build an `on_progress(done, total)` callback that drives a `st.progress` widget.

    Shared by the "Add a new metric" and "Add a new prompt" forms — both run
    a slow, per-item live API loop and want the same "N/total <unit> done"
    bar while it runs.
    """
    def on_progress(done: int, total: int) -> None:
        progress_widget.progress(done / total if total else 1.0, text=f"{done}/{total} {unit} done")
    return on_progress


def _render_clear_button(state_key: str, label: str) -> None:
    """Render a "🗑️ Clear <label> (names)" button for a session_state dict, if it's non-empty.

    Shared by both "Add a new ..." forms, each of which stores what it's
    added this session as `st.session_state[state_key]` (a dict keyed by
    slug) and needs a one-click way to drop all of it again.
    """
    items = st.session_state.get(state_key)
    if not items:
        return
    names = ", ".join(items)
    if st.button(f"🗑️ Clear {label} ({names})"):
        st.session_state[state_key] = {}
        st.rerun()


def _reattach_custom_metrics(df: pd.DataFrame, data_path: str, mtime: float) -> pd.DataFrame:
    """Re-attach any LLM-judged custom metrics added earlier this session.

    load_data() is cached on (path, mtime) and returns a plain fresh
    DataFrame from disk whenever either changes — it knows nothing about
    metrics added via the "Add a new metric" form, since those never touch
    the CSV on disk. Without this, a custom metric would vanish the moment
    any other widget triggered a rerun.

    Each stored metric remembers the (path, mtime, row count) it was judged
    against; if any of those have since drifted — a different file, an
    edited file, a changed row count — it's dropped rather than silently
    reattached to the wrong rows.
    """
    key = (data_path, mtime, len(df))
    for slug, meta in list(st.session_state.get("custom_metrics", {}).items()):
        if (meta["path"], meta["mtime"], meta["n_rows"]) != key:
            del st.session_state["custom_metrics"][slug]
            continue
        df[f"{slug}_score"] = meta["scores"]
        df[f"{slug}_reason"] = meta["reasons"]
    return df


def _select_active_items(all_items: list, kind: str, help_text: str) -> list:
    """Dropdown of which metrics or prompts are active — shared by both sidebar sections.

    "Active" is the "remove" action for either kind — deselecting one here
    only hides it from this session's analysis (weights/variance chart for
    a metric; variance chart/comparison for a prompt); the underlying data
    is untouched and reselecting brings it right back. A new item (a CSV
    column or prompt value not seen before, or a freshly added custom one)
    defaults to included.

    Bound to ``st.session_state[f"active_{kind}"]`` via ``key=`` rather than
    ``default=`` so other widgets — the "remove" button on a flagged
    metric/prompt in its tab — can change the selection by writing to that
    key and calling ``st.rerun()``. A plain ``default=`` is only honored on
    a widget's very first render; it silently ignores later programmatic
    changes, which would make that remove button a no-op.

    Args:
        all_items: Every currently-available metric or prompt (all_metrics/
            all_prompts in main()).
        kind: "metrics" or "prompts" — used to build both the session_state
            keys (``active_<kind>``/``known_<kind>``) and the multiselect's
            label ("Active <kind>").
        help_text: Tooltip text, specific to what "active" means for this kind.
    """
    active_key, known_key = f"active_{kind}", f"known_{kind}"
    known = st.session_state.get(known_key, [])
    if active_key not in st.session_state:
        st.session_state[active_key] = list(all_items)
    else:
        current = [i for i in st.session_state[active_key] if i in all_items]
        current += [i for i in all_items if i not in known]
        st.session_state[active_key] = current
    st.session_state[known_key] = list(all_items)

    return st.multiselect(f"Active {kind}", options=all_items, key=active_key, help=help_text)


def _render_add_metric_form(df: pd.DataFrame, data_path: str, mtime: float) -> None:
    """Expander form to judge one new custom metric across every row.

    Calls evaluate_new_metric() (evaluation.py): one live Anthropic API call
    per row, so — unlike everything else in this sidebar — it's slow and
    billed. Gated behind an explicit confirmation checkbox for that reason,
    and always scores the full dataset rather than a sample, so every
    metric stays fully populated and the rest of the workflow (which
    assumes dense score columns) never has to deal with partially-scored
    metrics or NaN-handling.
    """
    with st.expander("➕ Add a new metric (LLM judge)"):
        if not LLM_JUDGE_AVAILABLE:
            st.warning(f"LLM judge unavailable: {LLM_JUDGE_IMPORT_ERROR}")
            return

        saved_defs = load_custom_metric_definitions()
        picked = st.selectbox(
            "📂 Load a saved metric definition (optional)",
            ["— new metric —"] + list(saved_defs),
            help="Metrics saved to disk (see the '💾 Save' button after "
                 "judging) — pick one to reuse its name/question instead of "
                 "retyping them.",
        )
        default_name, default_description = "", ""
        if picked != "— new metric —":
            default_name = saved_defs[picked]["name"]
            default_description = saved_defs[picked]["description"]

        name = st.text_input("Metric name", value=default_name, placeholder="e.g. Tone: formal enough?")
        description = st.text_area(
            "Judge question", value=default_description,
            placeholder="e.g. Is the tone appropriately formal for a legal audience?",
        )
        st.warning(
            f"⚠️ **This costs money.** Scoring all {len(df)} rows makes {len(df)} live "
            f"Anthropic API calls ({EVAL_MODEL}) — that's {len(df)} calls billed to "
            f"your API credits, taking a few minutes to finish. It is **not** free "
            f"or local, unlike the rest of this app.",
            icon="💸",
        )
        confirmed = st.checkbox("I understand this spends API credits — go ahead")

        if st.button("Run judge", disabled=not (name and description and confirmed)):
            progress = st.progress(0.0, text="Starting…")
            try:
                slug = slugify(name)
                scored = evaluate_new_metric(
                    df.copy(), name, description,
                    on_progress=_progress_callback(progress, "rows judged"),
                )
            except Exception as e:  # noqa: BLE001 - surface any judge/parse/API failure to the user
                st.error(f"Judge run failed: {e}")
                return

            st.session_state.setdefault("custom_metrics", {})[slug] = {
                "name": name,
                "description": description,
                "path": data_path,
                "mtime": mtime,
                "n_rows": len(df),
                "scores": scored[f"{slug}_score"].tolist(),
                "reasons": scored[f"{slug}_reason"].tolist(),
            }
            st.success(f"Added metric '{slug}_score' — scored {len(df)} rows.")
            st.rerun()


def _render_save_metrics_section(data_path: str, base_n_rows: int) -> None:
    """One-click persistence for each custom metric added this session.

    Merges the definition into metrics.json (git-tracked, so it's shared
    with anyone who pulls the repo — alongside the standard 5 in the same
    file) and writes the computed scores into data_path's CSV in place
    (data/ is gitignored, so that part only ever affects your local copy) —
    so this metric no longer needs re-judging (or re-billing) next session.

    After a successful save, the metric naturally drops out of
    st.session_state on the next rerun: saving changes the CSV's mtime,
    and _reattach_custom_metrics's own staleness check (path, mtime, row
    count) treats that as "the file changed underneath it" and retires the
    now-redundant in-session copy — the column is already there from the
    fresh CSV read, so nothing is lost.
    """
    if not LLM_JUDGE_AVAILABLE:
        return
    custom_metrics = st.session_state.get("custom_metrics", {})
    if not custom_metrics:
        return

    for slug, meta in custom_metrics.items():
        if meta["path"] != data_path or meta["n_rows"] != base_n_rows:
            st.caption(
                f"⚠️ Can't save '{slug}' — it was judged against a different "
                "CSV or row count than what's currently loaded."
            )
            continue
        if st.button(f"💾 Save '{slug}' to disk", key=f"save_{slug}"):
            try:
                save_custom_metric_definition(slug, meta.get("name", slug), meta["description"])
                save_custom_metric_scores_to_csv(data_path, slug, meta["scores"], meta["reasons"])
            except Exception as e:  # noqa: BLE001 - surface any I/O failure to the user
                st.error(f"Save failed: {e}")
                continue
            st.success(f"Saved '{slug}': definition in metrics.json, scores written into {data_path}.")
            st.rerun()


def _render_definition_viewer(
    active_items: list,
    contents: dict,
    dates: dict,
    *,
    kind: str,
    expander_title: str,
    selectbox_label: str,
    not_available: str,
    as_code: bool = False,
) -> None:
    """Let the user look up an item's stored content, and when it was added.

    Shared by the "View metric description" / "View prompt template" panels
    — pure inspection, makes it easier to decide how to weight/remove a
    metric or compare/draft a prompt without going to read evaluation.py /
    prompts.py directly. `contents`/`dates` are keyed the same way as
    `active_items` (full ``*_score`` column names for metrics, prompt
    values for prompts) so a lookup never mismatches.

    `dates` is best-effort: a computed-not-judged metric (fk_normalized_score),
    a standard metric/prompt (date_added: null in metrics.json/prompts.json),
    or an unsaved this-session custom item all have no date to show, so the
    caption is just omitted for those.

    Args:
        active_items: The currently-active metrics or prompts to choose from.
        contents: slug/prompt -> description or template text.
        dates: slug/prompt -> ISO date string, only for entries that have one.
        kind: "metric" or "prompt" — used for the selectbox's session_state key.
        expander_title: e.g. "📋 View metric description".
        selectbox_label: e.g. "Metric".
        not_available: Fallback text when `chosen` isn't in `contents`.
        as_code: Render the content with st.code() (for a prompt template)
            instead of st.write() (for a metric's plain-text description).
    """
    if not active_items:
        return
    with st.expander(expander_title):
        chosen = st.selectbox(selectbox_label, active_items, key=f"view_{kind}_select")
        content = contents.get(chosen, not_available)
        if as_code:
            st.code(content, language=None)
        else:
            st.write(content)
        if chosen in dates:
            st.caption(f"Added: {dates[chosen]}")


# ── Prompt management: remove (active-prompt toggle), add (live LLM) ────────


def _deactivate_prompt(prompt: str) -> None:
    """Drop one prompt from the active-prompts selection.

    on_click callback for the same reason as _deactivate_metric: the
    key-bound multiselect above has already rendered by the time a tab's
    "remove" button is clicked, so the mutation has to happen in a callback
    (which runs before the next rerun) rather than inline in the tab body.
    """
    st.session_state["active_prompts"] = [
        p for p in st.session_state["active_prompts"] if p != prompt
    ]


def _reattach_custom_prompts(df: pd.DataFrame, data_path: str, mtime: float, base_n_rows: int) -> pd.DataFrame:
    """Re-attach any prompts generated earlier this session, and refresh fk_normalized_score.

    Mirrors _reattach_custom_metrics, but appends rows instead of columns —
    generate_new_prompt() adds one row per unique original_text rather than
    a new column, so nothing here needs to survive load_data()'s cache being
    invalidated by a rerun. Each stored prompt remembers the (path, mtime,
    row count) of the base data it was generated against; if that's drifted,
    it's dropped rather than appended to the wrong base rows.

    base_n_rows must be the *original* base row count (before any generated
    prompt's rows are appended) — not simply len(df) at call time. Once a
    second custom prompt is added, df already includes the first one's rows,
    so len(df) at that point is inflated; using it here would make every
    prompt after the first look stale on the very next rerun and get dropped.

    fk_normalized_score is recomputed across the full combined set every
    time (it's dataset-relative — see normalise_fk_grade), so it's not part
    of what's stored; storing it would just be stale the moment a second
    prompt gets added.
    """
    key = (data_path, mtime, base_n_rows)
    new_row_frames = []
    for slug, meta in list(st.session_state.get("custom_prompts", {}).items()):
        if (meta["path"], meta["mtime"], meta["base_n_rows"]) != key:
            del st.session_state["custom_prompts"][slug]
            continue
        new_row_frames.append(pd.DataFrame(meta["rows"]))

    if not new_row_frames:
        return df
    return normalise_fk_grade(pd.concat([df] + new_row_frames, ignore_index=True))


def _render_add_prompt_form(df: pd.DataFrame, data_path: str, mtime: float, base_n_rows: int) -> None:
    """Expander form to try one new prompt template against every unique text.

    Calls generate_new_prompt() (simplification.py): for every unique
    original_text, one live OpenAI call to simplify it plus one live
    Anthropic call to judge it on the standard dimensions — twice the API
    surface of "Add a new metric", across two different providers/keys.
    Gated behind the same explicit confirmation pattern for that reason.

    A template only has {text} available (no country/recommending_body/type
    — see simplify_text() in simplification.py for why), and any custom
    metric added via "Add a new metric" won't be scored for this new
    prompt's rows (only the standard dimensions are); both are noted in the
    form so neither is a silent surprise.
    """
    with st.expander("➕ Add a new prompt (LLM simplify + judge)"):
        if not PROMPT_GENERATION_AVAILABLE:
            st.warning(f"Prompt generation unavailable: {PROMPT_GENERATION_IMPORT_ERROR}")
            return

        saved_defs = load_custom_prompt_definitions()
        picked = st.selectbox(
            "📂 Load a saved prompt definition (optional)",
            ["— new prompt —"] + list(saved_defs),
            help="Prompts saved to disk (see the '💾 Save' button after "
                 "generating) — pick one to reuse its name/template instead "
                 "of retyping them.",
        )
        default_name, default_template = "", ""
        if picked != "— new prompt —":
            default_name = saved_defs[picked]["name"]
            default_template = saved_defs[picked]["template"]

        name = st.text_input("Prompt name", value=default_name, placeholder="e.g. Bullet Points")
        template = st.text_area(
            "Prompt template", value=default_template,
            placeholder="e.g. Rewrite the following as short bullet points "
                        "a non-expert can follow:\n\n{text}",
            help="Only {text} is filled in — this data has no "
                 "country/recommending_body/type to offer a template, unlike "
                 "the built-in prompts in prompts.py.",
        )
        n_unique = df["original_text"].nunique()
        st.warning(
            f"⚠️ **This costs money.** Trying this on all {n_unique} unique texts makes "
            f"{n_unique} OpenAI calls ({SIMPLIFY_MODEL}) to simplify plus {n_unique} "
            f"Anthropic calls ({EVAL_MODEL if LLM_JUDGE_AVAILABLE else 'the judge model'}) "
            f"to score them — {2 * n_unique} live, billed calls total, taking a few "
            f"minutes. Any custom metric added above won't be scored for this new "
            f"prompt's rows.",
            icon="💸",
        )
        confirmed = st.checkbox("I understand this spends API credits across two providers — go ahead")

        if st.button("Generate prompt", disabled=not (name and template and confirmed)):
            progress = st.progress(0.0, text="Starting…")
            try:
                slug = slugify(name)
                combined = generate_new_prompt(
                    df.copy(), name, template,
                    on_progress=_progress_callback(progress, "texts done"),
                )
            except Exception as e:  # noqa: BLE001 - surface any generation/judge/API failure to the user
                st.error(f"Prompt generation failed: {e}")
                return

            new_rows = combined[combined["prompt"] == slug].drop(columns=["fk_normalized_score"])
            st.session_state.setdefault("custom_prompts", {})[slug] = {
                "name": name,
                "template": template,
                "path": data_path,
                "mtime": mtime,
                "base_n_rows": base_n_rows,
                "rows": new_rows.to_dict("records"),
            }
            st.success(f"Added prompt '{slug}' — generated and scored {len(new_rows)} texts.")
            st.rerun()


def _render_save_prompts_section(data_path: str, base_n_rows: int) -> None:
    """One-click persistence for each custom prompt added this session.

    Mirrors _render_save_metrics_section: merges the definition into
    prompts.json (git-tracked, shared with anyone who pulls the repo —
    alongside the standard 5 in the same file) and writes the generated
    rows into data_path's CSV in place (data/ is gitignored, so that part
    only ever affects your local copy) — so this prompt no longer needs
    re-generating (or re-billing, across *two* providers) next session.

    After a successful save, the prompt naturally drops out of
    st.session_state on the next rerun for the same reason a saved metric
    does (see _render_save_metrics_section) — except here, appending rows
    also changes the CSV's row count, so saving one prompt will also
    retire any *other* still-unsaved custom metric or prompt whose stored
    base row count no longer matches. Save one thing at a time if you have
    several pending.
    """
    if not PROMPT_GENERATION_AVAILABLE:
        return
    custom_prompts = st.session_state.get("custom_prompts", {})
    if not custom_prompts:
        return

    for slug, meta in custom_prompts.items():
        if meta["path"] != data_path or meta["base_n_rows"] != base_n_rows:
            st.caption(
                f"⚠️ Can't save '{slug}' — it was generated against a "
                "different CSV or row count than what's currently loaded."
            )
            continue
        if st.button(f"💾 Save '{slug}' to disk", key=f"save_prompt_{slug}"):
            try:
                save_custom_prompt_definition(slug, meta.get("name", slug), meta["template"])
                save_custom_prompt_rows_to_csv(data_path, meta["rows"])
            except Exception as e:  # noqa: BLE001 - surface any I/O failure to the user
                st.error(f"Save failed: {e}")
                continue
            st.success(f"Saved '{slug}': definition in prompts.json, rows appended into {data_path}.")
            st.rerun()




# Static fallback explanations for metrics that have no per-row `_reason`
# column of their own — currently just fk_normalized_score, which is a
# computed number (see normalise_fk_grade in prompt_improvement_workflow.py),
# not an LLM judgment, so there's no reason text to show without this.
_METRIC_EXPLANATIONS = {
    "fk_normalized_score": (
        "Flesch-Kincaid reading-grade improvement, min-max normalised to 0–1 "
        "across this batch — 0 is the smallest reduction in reading difficulty "
        "seen here, 1 is the largest. It's relative to this batch, not an "
        "absolute reading level, so the scale shifts if the data changes."
    ),
}


def _render_metrics_list(row: pd.Series, score_columns: list) -> None:
    """Render the `- metric = score — reason` bullet list shared by every sample view.

    Falls back to _METRIC_EXPLANATIONS for metrics without their own
    `_reason` column, so a reader unfamiliar with e.g. the FK index still
    sees what the number means instead of a bare "—".
    """
    for metric in score_columns:
        if metric not in row.index:
            continue
        reason_col = metric.replace("_score", "_reason")
        if reason_col in row.index:
            reason = row[reason_col]
        else:
            reason = _METRIC_EXPLANATIONS.get(metric, "—")
        st.markdown(f"- `{metric}` = **{row[metric]}** — {reason}")


def _render_sample(row: pd.Series, score_columns: list, auto_format_original: bool = False) -> None:
    """Render one sample row: original/simplified stacked, then metrics below.

    Args:
        auto_format_original: If True, run the original text through
            format_legal_bullets() first (see the sidebar's "✨ Auto-format
            original text" toggle) so lettered clauses get their own line.
    """
    if "prompt" in row.index:
        st.caption(f"Prompt: `{row['prompt']}`")
    if "original_text" in row.index:
        original = row["original_text"]
        if auto_format_original:
            original = format_legal_bullets(original)
        st.markdown(f"**Original:** {original}")
    if "transcribed_text" in row.index:
        st.markdown(f"**Simplified:** {row['transcribed_text']}")

    _render_metrics_list(row, score_columns)


def _render_sample_side_by_side(row: pd.Series, score_columns: list, auto_format_original: bool = False) -> None:
    """Render one sample row: original/simplified in two columns, metrics below.

    Used by the Sanity Check and Verify Best Prompt tabs, where reading the
    two texts next to each other (rather than stacked) is the point of the
    exercise — Metric Variance and Prompt Comparison keep the stacked layout
    from _render_sample, since their samples are there to illustrate one
    flagged metric/prompt rather than to be read end-to-end.

    Args:
        auto_format_original: See _render_sample.
    """
    if "prompt" in row.index:
        st.caption(f"Prompt: `{row['prompt']}`")

    original_col, simplified_col = st.columns(2)
    if "original_text" in row.index:
        with original_col:
            st.markdown("**Original**")
            original = row["original_text"]
            if auto_format_original:
                original = format_legal_bullets(original)
            st.markdown(original)
    if "transcribed_text" in row.index:
        with simplified_col:
            st.markdown("**Simplified**")
            st.markdown(row["transcribed_text"])

    _render_metrics_list(row, score_columns)


# ── Step 1: metric variance ────────────────────────────────────────────────


def _deactivate_metric(metric: str) -> None:
    """Drop one metric from the active-metrics selection.

    Used as an `on_click` callback (not inline in the render function) because
    Streamlit forbids writing to `st.session_state["active_metrics"]` after
    the key-bound multiselect widget has already been instantiated this run —
    the sidebar (where that widget lives) renders before this tab's button.
    Callbacks run before the script reruns, so the mutation lands ahead of
    the next run's widget instantiation instead of after this run's.
    """
    st.session_state["active_metrics"] = [
        m for m in st.session_state["active_metrics"] if m != metric
    ]


def render_metric_variance_tab(
    df: pd.DataFrame, score_columns: list, n_samples: int, auto_format_original: bool = False
) -> None:
    threshold = st.slider(
        "Low-variance threshold", min_value=0.0, max_value=0.25, step=0.01,
        key="low_variance_threshold",
        help="Metrics with variance below this assign nearly the same score "
             "to every output — little signal, candidate for removal.",
    )
    result = check_metric_variance(df, score_columns, threshold=threshold, n_samples=n_samples, verbose=False)

    st.bar_chart(result["variances"])

    if not result["low_variance_metrics"]:
        st.success("All metrics have sufficient variance — no candidates for removal.")
        return

    for metric in result["low_variance_metrics"]:
        var = result["variances"][metric]
        with st.expander(f"⚠️ '{metric}' has low variance ({var:.4f})", expanded=False):
            st.write("Nearly constant score across all outputs. If the samples "
                     "below look reasonable, consider removing this metric.")
            st.button(
                f"🗑️ Remove '{metric}' from active metrics",
                key=f"remove_{metric}",
                on_click=_deactivate_metric,
                args=(metric,),
            )
            for _, row in result["samples"][metric].iterrows():
                _render_sample(row, [metric], auto_format_original)
                st.divider()


# ── Step 2: sanity check ───────────────────────────────────────────────────


def render_sanity_check_tab(
    df: pd.DataFrame, score_columns: list, n_samples: int, auto_format_original: bool = False
) -> None:
    # show_sanity_samples() isn't cached, so it draws a fresh random sample on
    # every script rerun; a button click is just the conventional way to
    # trigger that rerun on demand (any other widget interaction would too).
    st.button("🔀 Resample")
    sample = show_sanity_samples(df, score_columns, n_samples=n_samples, verbose=False)
    for _, row in sample.iterrows():
        _render_sample_side_by_side(row, score_columns, auto_format_original)
        st.divider()


# ── Step 3: prompt comparison ───────────────────────────────────────────────


def render_prompt_comparison_tab(
    df: pd.DataFrame, score_columns: list, n_samples: int, auto_format_original: bool = False
) -> pd.DataFrame:
    threshold = st.slider(
        "High-variance threshold", min_value=0.0, max_value=0.25, step=0.01,
        key="high_variance_threshold",
        help="Prompts with variance above this perform inconsistently across "
             "texts — inspect before trusting the mean ranking.",
    )
    stats = compare_prompts(df, high_var_threshold=threshold, verbose=False)

    # A boxplot (not just the mean bar) so a prompt's spread is visible at a
    # glance — a tall box/whiskers reads as "inconsistent" as fast as the
    # 'variance' column does, without checking a number. Sorted to match the
    # ranking below; one flat color since the x-axis already carries prompt
    # identity — coloring it too would just repeat that, not add anything.
    boxplot = (
        alt.Chart(df)
        .mark_boxplot(size=32)
        .encode(
            x=alt.X("prompt:N", sort=list(stats.index), title=None),
            y=alt.Y("average_score:Q", title="Average score"),
            color=alt.value("#4C78A8"),
        )
    )
    st.altair_chart(boxplot, use_container_width=True)
    st.dataframe(stats.style.format({"mean_score": "{:.3f}", "variance": "{:.4f}"}))

    samples = get_high_variance_samples(df, stats, score_columns, n_samples=n_samples)
    if not samples:
        st.success("All prompts have consistent scores.")
    for prompt, sample in samples.items():
        with st.expander(f"⚠️ '{prompt}' has high variance ({stats.loc[prompt, 'variance']:.4f})"):
            st.button(
                f"🗑️ Remove '{prompt}' from active prompts",
                key=f"remove_prompt_{prompt}",
                on_click=_deactivate_prompt,
                args=(prompt,),
            )
            for _, row in sample.iterrows():
                _render_sample(row, score_columns, auto_format_original)
                st.divider()

    return stats


# ── Step 4: verify best prompt ──────────────────────────────────────────────


def render_verify_best_prompt_tab(
    df: pd.DataFrame, prompt_stats: pd.DataFrame, score_columns: list, n_samples: int,
    auto_format_original: bool = False,
) -> None:
    if prompt_stats is None:
        st.info("Open the **Prompt Comparison** tab first to rank prompts.")
        return

    # Same pattern as the Sanity Check tab: verify_best_prompt() draws a
    # fresh random sample on every call, so the button just needs to trigger
    # a rerun.
    st.button("🔀 Resample", key="resample_verify_best_prompt")
    result = verify_best_prompt(df, prompt_stats, score_columns, n_samples=n_samples, verbose=False)
    st.metric("Best prompt", result["best_prompt"], f"mean score {result['mean_score']:.3f}")
    st.write("Read the outputs below — does this look like good simplification?")
    for _, row in result["samples"].iterrows():
        _render_sample_side_by_side(row, score_columns, auto_format_original)
        st.divider()


# ── App ───────────────────────────────────────────────────────────────────────

# Each entry is (tab label, render function). Add a step by adding one tuple
# here and one render_* function above — nothing else changes.
STEPS = [
    ("1. Metric Variance", render_metric_variance_tab),
    ("2. Sanity Check", render_sanity_check_tab),
    ("3. Prompt Comparison", render_prompt_comparison_tab),
    ("4. Verify Best Prompt", render_verify_best_prompt_tab),
]


def main() -> None:
    st.set_page_config(page_title="Prompt Improvement Workflow", layout="wide")
    st.title("Prompt Improvement Workflow")

    with st.sidebar:
        st.header("Data")
        data_path = st.text_input("CSV path", DEFAULT_DATA_PATH)

        if not Path(data_path).is_file():
            st.error(f"Could not read '{data_path}': no such file")
            st.stop()
        mtime = Path(data_path).stat().st_mtime

        saved_settings = _load_settings(data_path)
        if saved_settings:
            st.caption(f"⚙️ Loaded saved settings from {_settings_path(data_path).name}")
        # Captured now, acted on at the end of this block — the actual save
        # needs active_metrics/weights/active_prompts, which don't exist yet
        # this far up, but the button itself can render anywhere.
        save_settings_clicked = st.button(
            "💾 Save current settings",
            help="Remembers Samples to show, Auto-format, both variance "
                 "thresholds, which metrics/prompts are active, and the "
                 "weight sliders — as a sidecar file next to this CSV.",
        )

        # Seed once per session — same pattern as active_metrics/active_prompts
        # below — so a saved value only applies on first load and a later
        # in-session tweak is never silently overwritten by it.
        st.session_state.setdefault("n_samples", saved_settings.get("n_samples", N_SAMPLES))
        st.session_state.setdefault(
            "auto_format_original", saved_settings.get("auto_format_original", True)
        )
        st.session_state.setdefault(
            "low_variance_threshold",
            saved_settings.get("low_variance_threshold", LOW_VARIANCE_THRESHOLD),
        )
        st.session_state.setdefault(
            "high_variance_threshold",
            saved_settings.get("high_variance_threshold", HIGH_VARIANCE_THRESHOLD),
        )

        n_samples = st.slider("Samples to show", 1, 10, key="n_samples")
        auto_format_original = st.checkbox(
            "✨ Auto-format original text", key="auto_format_original",
            help="Break lettered '(a)/(b)/(c)...' clauses in the original "
                 "text onto their own lines, matching how the simplified "
                 "text usually already looks — makes the two easier to "
                 "compare side by side. Toggle off to see the raw text.",
        )

        df = load_data(data_path, mtime).copy()
        df = _reattach_custom_metrics(df, data_path, mtime)
        base_n_rows = len(df)  # true base row count, before any generated-prompt rows
        df = _reattach_custom_prompts(df, data_path, mtime, base_n_rows)
        all_metrics = get_score_columns(df)
        all_prompts = sorted(df["prompt"].unique())

        # Built from whatever's currently available, for the "view" panels below —
        # not from active_metrics/active_prompts, so a deselected item can still be
        # inspected before deciding whether to bring it back.
        metric_descriptions = dict(_METRIC_EXPLANATIONS)
        metric_dates = {}
        if LLM_JUDGE_AVAILABLE:
            # Standard and custom metrics live together in one file,
            # metrics.json (next to evaluation.py) — load_all_metric_definitions()
            # reads both at once, no separate merge step needed. Standard
            # entries have date_added: null there, which entry.get() below
            # correctly skips rather than showing a bogus date.
            for slug, entry in load_all_metric_definitions().items():
                metric_descriptions[f"{slug}_score"] = entry["description"]
                if entry.get("date_added"):
                    metric_dates[f"{slug}_score"] = entry["date_added"]
        for slug, meta in st.session_state.get("custom_metrics", {}).items():
            metric_descriptions[f"{slug}_score"] = meta["description"]

        # Standard and custom prompts live together in one file, prompts.json
        # (next to prompts.py) — load_all_prompt_definitions() reads both at
        # once. No PROMPT_GENERATION_AVAILABLE guard needed here: unlike
        # generating a prompt, reading definitions needs no OpenAI/Anthropic
        # key. Standard entries have date_added: null, which entry.get()
        # below correctly skips rather than showing a bogus date.
        prompt_templates = {}
        prompt_dates = {}
        for slug, entry in load_all_prompt_definitions().items():
            prompt_templates[slug] = entry["template"]
            if entry.get("date_added"):
                prompt_dates[slug] = entry["date_added"]
        for slug, meta in st.session_state.get("custom_prompts", {}).items():
            prompt_templates[slug] = meta["template"]

        # Seed the "active"/"known" pair from saved settings, once per session
        # only (the same guard _select_active_items itself uses) — after
        # this, that function's own "keep what's still active + fold in
        # anything not in known" logic runs unmodified and does the right
        # thing whether it was just seeded from disk or is mid-session.
        if "active_metrics" not in st.session_state and "active_metrics" in saved_settings:
            st.session_state["active_metrics"] = saved_settings["active_metrics"]
            st.session_state["known_metrics"] = saved_settings.get("known_metrics", [])
        if "active_prompts" not in st.session_state and "active_prompts" in saved_settings:
            st.session_state["active_prompts"] = saved_settings["active_prompts"]
            st.session_state["known_prompts"] = saved_settings.get("known_prompts", [])

        st.header("Metrics")
        active_metrics = _select_active_items(
            all_metrics, "metrics",
            help_text="Deselect a metric to exclude it from the weighted average, "
                      "variance chart, and prompt comparison. It stays in the data "
                      "and can be reselected any time.",
        )
        _render_add_metric_form(df, data_path, mtime)
        _render_save_metrics_section(data_path, base_n_rows)
        _render_clear_button("custom_metrics", "custom metrics")
        _render_definition_viewer(
            active_metrics, metric_descriptions, metric_dates,
            kind="metric", expander_title="📋 View metric description", selectbox_label="Metric",
            not_available="_No description available for this metric._",
        )

        st.caption("Higher weight = more influence on a prompt's average score.")
        saved_weights = saved_settings.get("weights", {})
        weights = {}
        for col in active_metrics:
            weight_key = f"weight_{col}"
            default_weight = 5.0 if col in _READABILITY_METRICS else 1.0
            st.session_state.setdefault(weight_key, saved_weights.get(col, default_weight))
            weights[col] = st.slider(col, 0.0, 10.0, step=0.5, key=weight_key)

        st.header("Prompts")
        active_prompts = _select_active_items(
            all_prompts, "prompts",
            help_text="Deselect a prompt to exclude all its rows from the variance "
                      "chart and prompt comparison. Data stays put; reselect any time.",
        )
        _render_add_prompt_form(df, data_path, mtime, base_n_rows)
        _render_save_prompts_section(data_path, base_n_rows)
        _render_clear_button("custom_prompts", "added prompts")
        _render_definition_viewer(
            active_prompts, prompt_templates, prompt_dates,
            kind="prompt", expander_title="📋 View prompt template", selectbox_label="Prompt",
            not_available="No template available for this prompt.", as_code=True,
        )

        # Acted on here (not right where the button rendered, up top) because
        # weights/active_prompts don't exist yet that early in the script.
        if save_settings_clicked:
            _save_settings(data_path, {
                "n_samples": n_samples,
                "auto_format_original": auto_format_original,
                "low_variance_threshold": st.session_state["low_variance_threshold"],
                "high_variance_threshold": st.session_state["high_variance_threshold"],
                "active_metrics": active_metrics,
                "known_metrics": all_metrics,
                "active_prompts": active_prompts,
                "known_prompts": all_prompts,
                "weights": weights,
            })
            st.success(f"Settings saved to {_settings_path(data_path)}")

    if not active_metrics:
        st.warning("No active metrics selected — pick at least one in the sidebar.")
        st.stop()
    if not active_prompts:
        st.warning("No active prompts selected — pick at least one in the sidebar.")
        st.stop()

    df = df[df["prompt"].isin(active_prompts)].reset_index(drop=True)
    df = compute_weighted_average(df, active_metrics, weights)

    tabs = st.tabs([label for label, _ in STEPS])
    prompt_stats = st.session_state.get("prompt_stats")
    for (_, render), tab in zip(STEPS, tabs):
        with tab:
            if render is render_prompt_comparison_tab:
                prompt_stats = render(df, active_metrics, n_samples, auto_format_original)
                st.session_state["prompt_stats"] = prompt_stats
            elif render is render_verify_best_prompt_tab:
                render(df, prompt_stats, active_metrics, n_samples, auto_format_original)
            else:
                render(df, active_metrics, n_samples, auto_format_original)


if __name__ == "__main__":
    main()
