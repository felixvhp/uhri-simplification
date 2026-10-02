"""
Pkg 3: Evaluation dashboard — Streamlit GUI [last updated 2026-10-02]

Usage:
    streamlit run src/dashboard/app.py        (from the repo root)

    Reads the same long-format CSV as workflow.py (the settings' dashboard
    data_path, editable in the sidebar) and walks through the same four-step
    evaluation report interactively in the browser instead of via CLI
    `input()` prompts — plus one GUI-only tab, Metric Covariance
    (check_metric_covariance(), also in workflow.py, just not yet wired into
    that module's CLI __main__).

Deliberately thin: this file is UI only. Every number it shows comes from
calling workflow.py's step functions with verbose=False, which hands back
plain DataFrames/dicts instead of printed text. If the underlying
computation ever needs to change, change it there once; both this GUI and
the CLI script pick it up automatically.

The dashboard never calls the simplification code. It talks to it only
through the shared settings (src/settings):
    - "Add a new metric" judges it here (live Anthropic calls), and "💾 Save"
      puts its definition in the user metrics file and its scores in the CSV.
    - "Add a new prompt" only saves the template to the user prompts file
      and, if ticked, selects it. The simplification step then produces its
      outputs; "Run missing metrics" here judges them.
    - "💾 Save current settings" writes weights, active metrics/prompts,
      thresholds and the prompts selected for simplification to the user
      settings file. Defaults are never written.

New metric: just add it to metrics.json — main() seeds an empty column for
every *active* entry a loaded CSV doesn't have yet, so it shows up
immediately (flagged "0/N judged"), rather than being invisible until
someone happens to judge it for this particular CSV.

Resumption:
    - New step: write one `render_<step>_tab(...)` function and add one
      entry to `STEPS`. Nothing else needs to change.
"""

import importlib
import os
import random
import sys
from pathlib import Path
from typing import Callable

# `streamlit run src/dashboard/app.py` executes this file as a plain script,
# with no package context. Put the repo root on sys.path and name the
# package so the relative imports below resolve the same way they do when
# this is imported as src.dashboard.app.
if not __package__:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    importlib.import_module("src.dashboard")
    __package__ = "src.dashboard"

import altair as alt
import pandas as pd
import streamlit as st

from ..settings import load_metrics, load_prompts, load_settings, save_metric, save_prompt, save_settings
from .evaluation import EVAL_MODEL, add_readability, evaluate_missing_dimensions, save_metric_scores_to_csv
from .text_utils import format_legal_bullets, slugify
from .workflow import (
    get_score_columns,
    normalise_fk_grade,
    compute_weighted_average,
    check_metric_variance,
    check_metric_covariance,
    show_sanity_samples,
    compare_prompts,
    get_high_variance_samples,
    verify_best_prompt,
)

# evaluation.py loads .env on import and builds its Anthropic client lazily,
# so a missing key doesn't break the app — it only disables the live-judge
# forms. Everything else (weights, removing metrics, the report tabs) is local.
LLM_JUDGE_AVAILABLE = bool(os.environ.get("ANTHROPIC_API_KEY"))
LLM_JUDGE_IMPORT_ERROR = None if LLM_JUDGE_AVAILABLE else "ANTHROPIC_API_KEY is not set"


# ── Data loading ──────────────────────────────────────────────────────────────


@st.cache_data(show_spinner="Loading data…")
def load_data(path: str, mtime: float) -> pd.DataFrame:
    """Load and score the long-format CSV.

    Args:
        path: Path to the long-format CSV (see workflow.py).
        mtime: File modification time. Deliberately part of the cache key (no
            leading underscore) so editing the CSV on disk — e.g. re-running
            the LLM judge on more rows — invalidates the cache automatically;
            without it, Streamlit would keep serving the first-loaded copy
            for the rest of the session.

    Returns:
        DataFrame with ``fk_normalized_score`` added (and the readability
        columns it's derived from, if the file didn't have them yet).
    """
    df = pd.read_csv(path)
    return normalise_fk_grade(add_readability(df))


# ── Metric management: tweak (weights), remove (active-metric toggle), ──────
# ── add (live LLM judge) ──────────────────────────────────────────────────────


def _render_sample_controls(total: int, key_prefix: str, item_label: str) -> tuple:
    """Slider + randomize button for picking a random subset to run a live LLM action over.

    Shared by every "run the judge/generator over the dataset" action (Run
    missing metrics, Add a new metric, Add a new prompt) — all three cost
    real money per row/text, and a dataset that's small and cheap today may
    not stay that way. Defaults to the full count, so nothing changes for
    anyone who doesn't touch the slider.

    Args:
        total: How many items are available to sample from (rows with
            gaps, or unique texts — whatever the caller is about to run
            the action over).
        key_prefix: Unique per call site — becomes the slider/button's
            session_state keys, so two of these on the same page (unlikely
            today, but cheap to keep safe) never collide.
        item_label: What's being sampled, for the slider's label/help text,
            e.g. "rows with gaps" or "unique texts".

    Returns:
        (sample_size, seed): seed only changes when "🎲" is clicked, so the
        *same* random subset survives reruns from unrelated widgets (typing
        in the name field, say) instead of reshuffling under the user on
        every interaction. Pass seed straight to e.g.
        ``random.Random(seed).sample(items, sample_size)``.
    """
    size_key, seed_key = f"{key_prefix}_sample_size", f"{key_prefix}_sample_seed"
    st.session_state.setdefault(size_key, total)
    st.session_state.setdefault(seed_key, 0)

    slider_col, button_col = st.columns([5, 1])
    with slider_col:
        sample_size = st.slider(
            f"Sample size ({item_label})", min_value=1, max_value=total, key=size_key,
            help=f"Run this over a random {total}-item subset instead of all of "
                 f"them — cheaper/faster to try first, and necessary once the "
                 f"dataset is too large to run in full every time.",
        )
    with button_col:
        st.write("")  # vertical spacer so the button lines up with the slider
        if st.button("🎲", key=f"{key_prefix}_randomize", help="Pick a new random subset"):
            st.session_state[seed_key] += 1

    return sample_size, st.session_state[seed_key]


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


def _select_active_items(
    all_items: list, kind: str, help_text: str, format_func: Callable[[str], str] = str,
) -> list:
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
        format_func: How to render one item's label in the dropdown. The
            selection itself is always the real item (a column name or
            prompt value) — this only changes what's displayed, e.g. the
            metrics side appends a "not fully judged yet" warning.
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

    return st.multiselect(
        f"Active {kind}", options=all_items, key=active_key, help=help_text, format_func=format_func,
    )


def _build_metric_names(all_metrics: list, metric_defs: dict) -> dict:
    """Map each "_score" column to a human-readable display name.

    metrics.json's own "name" field is the source — the same field a
    custom metric's typed name already lives in (see _render_add_metric_form)
    — so a technical slug like "obligations_complete_score" never has to
    be shown verbatim anywhere in the GUI. A column metrics.json doesn't
    govern at all (fk_normalized_score; a one-off column added by hand)
    falls back to a generic "underscores -> Title Case" conversion of its
    own name.
    """
    names = {}
    for col in all_metrics:
        slug = col[: -len("_score")] if col.endswith("_score") else col
        entry = metric_defs.get(slug)
        names[col] = entry["name"] if entry else slug.replace("_", " ").title()
    return names


def _metric_option_label(col: str, metric_coverage: dict, metric_names: dict) -> str:
    """Multiselect format_func: show a metric's display name, flagging gaps.

    `col` is always the real column name (what gets selected/used downstream)
    — this only changes the displayed text: the human name from
    metric_names, plus a "done/total judged" warning when metric_coverage
    says this one has gaps. A column not in metric_coverage at all
    (retired, or not governed by metrics.json — fk_normalized_score) is
    shown with no warning, same as before this feature existed.
    """
    label = metric_names.get(col, col)
    coverage = metric_coverage.get(col)
    if coverage and coverage[0] < coverage[1]:
        done, total = coverage
        return f"{label}  ⚠️ {done}/{total} judged"
    return label


def _render_run_missing_metrics_section(
    df: pd.DataFrame,
    data_path: str,
    base_n_rows: int,
    metric_defs: dict,
    metric_coverage: dict,
) -> None:
    """Offer to judge whichever active metrics.json entries aren't fully scored yet.

    metrics.json is the ground truth for which metrics exist (see main()'s
    column-seeding, right before metric_coverage is computed) — a metric
    defined there but never judged for this CSV shows up here with
    0/base_n_rows judged, exactly like one only partially judged after a
    judge-prompt rewrite added it. One button handles both cases at once,
    across every incomplete active metric together: evaluate_missing_
    dimensions() makes exactly one Anthropic call per row that still has
    *any* gap, covering whichever of these dimensions that row is missing
    — not one call per metric.

    Retired (active: false) metrics never appear here; see
    metric_coverage's construction in main() for why.

    Offers a sample-size slider (see _render_sample_controls()) over
    whichever rows actually have a gap, rather than always running every
    one of them — important once the dataset is large enough that "every
    row with a gap" is itself a lot of rows.
    """
    if not LLM_JUDGE_AVAILABLE:
        return
    incomplete = {
        slug: metric_defs[slug]["description"]
        for slug in metric_defs
        if metric_coverage.get(f"{slug}_score", (base_n_rows, base_n_rows))[0] < base_n_rows
    }
    if not incomplete:
        return

    base_df = df.iloc[:base_n_rows]
    rows_needing_work = [
        i for i in base_df.index if any(pd.isna(base_df.at[i, f"{slug}_score"]) for slug in incomplete)
    ]

    with st.expander(f"⚠️ {len(incomplete)} metric(s) not fully judged over this dataset"):
        st.write(", ".join(f"**{metric_defs[slug]['name']}**" for slug in sorted(incomplete)))
        sample_size, seed = _render_sample_controls(
            len(rows_needing_work), "run_missing_metrics", "rows with gaps",
        )
        sampled_rows = random.Random(seed).sample(rows_needing_work, sample_size)

        st.warning(
            f"⚠️ **This costs money.** Filling the gaps makes {sample_size} live "
            f"Anthropic API calls ({EVAL_MODEL}) covering {len(incomplete)} metric(s) — "
            f"billed to your API credits, taking a few minutes.",
            icon="💸",
        )
        confirmed = st.checkbox(
            "I understand this spends API credits — go ahead", key="confirm_run_missing_metrics",
        )
        button_label = (
            "🧮 Run missing metrics over full dataset" if sample_size == len(rows_needing_work)
            else f"🧮 Run missing metrics over sample ({sample_size})"
        )
        if st.button(button_label, disabled=not confirmed):
            progress = st.progress(0.0, text="Starting…")
            try:
                filled = evaluate_missing_dimensions(
                    base_df.copy(), incomplete, rows=sampled_rows,
                    on_progress=_progress_callback(progress, "rows judged"),
                )
            except Exception as e:  # noqa: BLE001 - surface any judge/parse/API failure to the user
                st.error(f"Judge run failed: {e}")
                return
            try:
                # Writes f"{slug}_score"/f"{slug}_reason" columns into the
                # CSV, re-read fresh from disk so it can never clobber
                # anything else. Rows outside the sample keep whatever they
                # already had (NaN if still unjudged), since `filled` only
                # touched `sampled_rows`.
                for slug in incomplete:
                    save_metric_scores_to_csv(
                        data_path, slug,
                        filled[f"{slug}_score"].tolist(),
                        filled[f"{slug}_reason"].tolist(),
                    )
            except Exception as e:  # noqa: BLE001 - surface any I/O failure to the user
                st.error(f"Save failed: {e}")
                return
            st.success(f"Judged and saved {len(incomplete)} metric(s) across {sample_size} rows.")
            st.rerun()


def _render_add_metric_form(df: pd.DataFrame, data_path: str, mtime: float) -> None:
    """Expander form to judge one new custom metric, optionally over a sample.

    Calls evaluate_missing_dimensions() (evaluation.py) with a single
    ``{slug: description}`` dimension — a brand-new metric has no existing
    column, so every row in the chosen sample counts as a gap to fill.
    Sampling (see _render_sample_controls()) lets a new judge question be
    sanity-checked cheaply on a handful of rows before committing to the
    full, and eventually much larger, dataset; rows left out of the sample
    just stay NaN — visible as "⚠️ N/total judged" in the metrics dropdown
    once saved, and fillable later via "Run missing metrics" without
    re-billing this sample.
    """
    with st.expander("➕ Add a new metric (LLM judge)"):
        if not LLM_JUDGE_AVAILABLE:
            st.warning(f"LLM judge unavailable: {LLM_JUDGE_IMPORT_ERROR}")
            return

        saved_defs = load_metrics()
        picked = st.selectbox(
            "📂 Load a saved metric definition (optional)",
            ["— new metric —"] + list(saved_defs),
            help="Metrics saved to disk — pick one to reuse its name/"
                 "question instead of retyping them (e.g. as a starting "
                 "point for a close variant).",
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
        sample_size, seed = _render_sample_controls(len(df), "add_metric", "rows")
        sampled_rows = random.Random(seed).sample(list(df.index), sample_size)

        st.warning(
            f"⚠️ **This costs money.** Scoring {sample_size} row(s) makes {sample_size} live "
            f"Anthropic API calls ({EVAL_MODEL}) — billed to your API credits, taking a few "
            f"minutes to finish. It is **not** free or local, unlike the rest of this app.",
            icon="💸",
        )
        confirmed = st.checkbox("I understand this spends API credits — go ahead")

        if st.button("Run judge", disabled=not (name and description and confirmed)):
            progress = st.progress(0.0, text="Starting…")
            try:
                slug = slugify(name)
                scored = evaluate_missing_dimensions(
                    df.copy(), {slug: description}, rows=sampled_rows,
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
            st.success(f"Added metric '{slug}_score' — scored {sample_size} of {len(df)} rows.")
            st.rerun()


def _render_save_metrics_section(data_path: str, base_n_rows: int) -> None:
    """One-click persistence for each custom metric added this session.

    Saves the definition to the user metrics file (src/settings/user/,
    gitignored — move it into defaults/metrics.json by hand to share it) and
    writes the computed scores into data_path's CSV in place — so this
    metric no longer needs re-judging (or re-billing) next session.

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
                save_metric(slug, meta.get("name", slug), meta["description"])
                save_metric_scores_to_csv(data_path, slug, meta["scores"], meta["reasons"])
            except Exception as e:  # noqa: BLE001 - surface any I/O failure to the user
                st.error(f"Save failed: {e}")
                continue
            st.success(f"Saved '{slug}': definition in your user metrics file, scores written into {data_path}.")
            st.rerun()


def _render_definition_viewer(
    active_items: list,
    contents: dict,
    *,
    kind: str,
    expander_title: str,
    selectbox_label: str,
    not_available: str,
    as_code: bool = False,
    format_func: Callable[[str], str] = str,
) -> None:
    """Let the user look up an item's stored content.

    Shared by the "View metric description" / "View prompt template" panels
    — pure inspection, makes it easier to decide how to weight/remove a
    metric or compare/draft a prompt without opening the settings files. `contents` is keyed the same way as `active_items`
    (full ``*_score`` column names for metrics, prompt values for prompts)
    so a lookup never mismatches.

    Args:
        active_items: The currently-active metrics or prompts to choose from.
        contents: slug/prompt -> description or template text.
        kind: "metric" or "prompt" — used for the selectbox's session_state key.
        expander_title: e.g. "📋 View metric description".
        selectbox_label: e.g. "Metric".
        not_available: Fallback text when `chosen` isn't in `contents`.
        as_code: Render the content with st.code() (for a prompt template)
            instead of st.write() (for a metric's plain-text description).
        format_func: How to render one item's label in the selectbox — same
            "selection is the real item, this only changes the displayed
            text" contract as _select_active_items. Metrics pass their
            display-name lookup here; prompts (which have no separate
            technical/display split) leave it at the identity default.
    """
    if not active_items:
        return
    with st.expander(expander_title):
        chosen = st.selectbox(selectbox_label, active_items, key=f"view_{kind}_select", format_func=format_func)
        content = contents.get(chosen, not_available)
        if as_code:
            st.code(content, language=None)
        else:
            st.write(content)


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


def _simplification_command(data_path: str) -> str:
    """The command that makes the simplification step (re)write this dashboard's CSV."""
    return f"python -m src.simplification <input file> -o {data_path} --layout long"


def _render_selected_prompts(prompt_defs: dict) -> list:
    """Multiselect of which prompts the simplification step runs.

    This is the dashboard's one lever on the simplification step: the
    choice is stored as ``selected_prompts`` in the shared settings (on
    "💾 Save current settings"), which is all the simplification code reads.
    Selecting more than one makes it output one simplification per prompt,
    which it warns about — fine for comparing prompts here, usually not
    what a data consumer wants.
    """
    options = [slug for slug, entry in prompt_defs.items() if entry.get("active", True)]
    selected = st.multiselect(
        "Used by the simplification step", options=options, key="selected_prompts",
        help="Which prompt(s) the simplification step runs next time. Saved "
             "with '💾 Save current settings'.",
    )
    if len(selected) > 1:
        st.caption(f"⚠️ {len(selected)} prompts selected — every text will get {len(selected)} simplifications.")
    return selected


def _render_add_prompt_form(data_path: str) -> None:
    """Expander form to define one new prompt template.

    Only saves the definition (to the user prompts file) and, if ticked,
    selects it for the simplification step — this app never runs a
    simplification itself. The form shows the command that produces the new
    prompt's outputs; once that has run, they show up here as an ordinary
    prompt with every metric flagged "not judged", ready for "Run missing
    metrics".
    """
    with st.expander("➕ Add a new prompt"):
        saved_defs = load_prompts()
        picked = st.selectbox(
            "📂 Start from an existing prompt (optional)",
            ["— new prompt —"] + list(saved_defs),
            help="Pick one to reuse its name/template as a starting point.",
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
            help="{text} is the text to simplify. {type}, {body} and {country} "
                 "are filled in when the input data has them.",
        )
        select_it = st.checkbox("Select it for the next simplification run", value=True)

        if st.button("💾 Save prompt", disabled=not (name and template)):
            try:
                slug = slugify(name)
                save_prompt(slug, name, template)
                if select_it:
                    selected = load_settings().get("selected_prompts", [])
                    if slug not in selected:
                        save_settings({"selected_prompts": selected + [slug]})
                        st.session_state.pop("selected_prompts", None)  # re-seed from the saved value
            except Exception as e:  # noqa: BLE001 - surface any validation/I/O failure to the user
                st.error(f"Save failed: {e}")
                return
            st.session_state["prompt_saved_notice"] = slug
            st.rerun()

        if "prompt_saved_notice" in st.session_state:
            st.success(
                f"Saved '{st.session_state['prompt_saved_notice']}' to your user prompts file. "
                "To get its outputs into this dashboard, run:"
            )
            st.code(_simplification_command(data_path), language=None)




# Static fallback explanations for metrics that have no per-row `_reason`
# column of their own — currently just fk_normalized_score, which is a
# computed number (see normalise_fk_grade in workflow.py),
# not an LLM judgment, so there's no reason text to show without this.
_METRIC_EXPLANATIONS = {
    "fk_normalized_score": (
        "Flesch-Kincaid reading-grade improvement, min-max normalised to 0–1 "
        "across this batch — 0 is the smallest reduction in reading difficulty "
        "seen here, 1 is the largest. It's relative to this batch, not an "
        "absolute reading level, so the scale shifts if the data changes."
    ),
}


def _render_metrics_list(row: pd.Series, score_columns: list, metric_names: dict) -> None:
    """Render the `- Name = score — reason` bullet list shared by every sample view.

    `metric_names` (see _build_metric_names()) is what keeps a technical
    column like "obligations_complete_score" from ever being shown
    verbatim — every caller down this chain gets it from main() rather
    than re-deriving it.

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
        st.markdown(f"- **{metric_names.get(metric, metric)}** = **{row[metric]}** — {reason}")


def _render_sample(
    row: pd.Series, score_columns: list, metric_names: dict, auto_format_original: bool = False,
) -> None:
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

    _render_metrics_list(row, score_columns, metric_names)


def _render_sample_side_by_side(
    row: pd.Series, score_columns: list, metric_names: dict, auto_format_original: bool = False,
) -> None:
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

    _render_metrics_list(row, score_columns, metric_names)


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
    df: pd.DataFrame, score_columns: list, metric_names: dict, metric_descriptions: dict, n_samples: int,
    auto_format_original: bool = False,
) -> None:
    threshold = st.slider(
        "Low-variance threshold", min_value=0.0, max_value=0.25, step=0.01,
        key="low_variance_threshold",
        help="Metrics with variance below this assign nearly the same score "
             "to every output — little signal, candidate for removal.",
    )
    result = check_metric_variance(df, score_columns, threshold=threshold, n_samples=n_samples, verbose=False)

    # Renamed for display only — result["variances"]/["low_variance_metrics"]
    # keep the real column names, since those still drive the remove button's
    # on_click target below.
    st.bar_chart(result["variances"].rename(lambda col: metric_names.get(col, col)))

    if not result["low_variance_metrics"]:
        st.success("All metrics have sufficient variance — no candidates for removal.")
        return

    for metric in result["low_variance_metrics"]:
        var = result["variances"][metric]
        name = metric_names.get(metric, metric)
        with st.expander(f"⚠️ '{name}' has low variance ({var:.4f})", expanded=False):
            # The judge question itself, not sample rows — those are one click
            # away on the Sanity Check tab, and showing both here was
            # redundant with that tab's whole purpose.
            st.write(metric_descriptions.get(metric, "_No description available for this metric._"))
            st.caption("Nearly constant score across all outputs. See the **Sanity Check** "
                       "tab for example outputs across all metrics, then decide whether to remove this one.")
            st.button(
                f"🗑️ Remove '{name}' from active metrics",
                key=f"remove_{metric}",
                on_click=_deactivate_metric,
                args=(metric,),
            )


# ── Step 2: metric covariance ────────────────────────────────────────────────


def render_metric_covariance_tab(
    df: pd.DataFrame, score_columns: list, metric_names: dict, metric_descriptions: dict, n_samples: int,
    auto_format_original: bool = False,
) -> None:
    threshold = st.slider(
        "High-covariance threshold", min_value=0.0, max_value=1.0, step=0.05,
        key="covariance_threshold",
        help="Two metrics correlated above this (in either direction) tend "
             "to agree almost every time — one adds little signal beyond "
             "the other and is a candidate for removal.",
    )
    result = check_metric_covariance(df, score_columns, threshold=threshold, verbose=False)

    display_corr = result["correlations"].rename(
        index=lambda col: metric_names.get(col, col), columns=lambda col: metric_names.get(col, col),
    )
    st.dataframe(display_corr.style.format("{:.2f}", na_rep="—"))

    if not result["high_covariance_pairs"]:
        st.success("No metric pair is highly correlated — no redundancy candidates.")
        return

    for metric_a, metric_b, corr in result["high_covariance_pairs"]:
        name_a, name_b = metric_names.get(metric_a, metric_a), metric_names.get(metric_b, metric_b)
        with st.expander(f"⚠️ '{name_a}' and '{name_b}' are highly correlated ({corr:.2f})", expanded=False):
            st.caption("These two tend to agree almost every time — one adds little "
                       "over the other. Consider removing one, or merging them into "
                       "a single judge question.")
            desc_col_a, desc_col_b = st.columns(2)
            with desc_col_a:
                st.markdown(f"**{name_a}**")
                st.write(metric_descriptions.get(metric_a, "_No description available for this metric._"))
            with desc_col_b:
                st.markdown(f"**{name_b}**")
                st.write(metric_descriptions.get(metric_b, "_No description available for this metric._"))
            remove_col_a, remove_col_b = st.columns(2)
            with remove_col_a:
                st.button(
                    f"🗑️ Remove '{name_a}'", key=f"remove_covar_{metric_a}_{metric_b}_a",
                    on_click=_deactivate_metric, args=(metric_a,),
                )
            with remove_col_b:
                st.button(
                    f"🗑️ Remove '{name_b}'", key=f"remove_covar_{metric_a}_{metric_b}_b",
                    on_click=_deactivate_metric, args=(metric_b,),
                )


# ── Step 3: sanity check ───────────────────────────────────────────────────


def render_sanity_check_tab(
    df: pd.DataFrame, score_columns: list, metric_names: dict, n_samples: int,
    auto_format_original: bool = False,
) -> None:
    # show_sanity_samples() isn't cached, so it draws a fresh random sample on
    # every script rerun; a button click is just the conventional way to
    # trigger that rerun on demand (any other widget interaction would too).
    st.button("🔀 Resample")
    sample = show_sanity_samples(df, score_columns, n_samples=n_samples, verbose=False)
    for _, row in sample.iterrows():
        _render_sample_side_by_side(row, score_columns, metric_names, auto_format_original)
        st.divider()


# ── Step 4: prompt comparison ───────────────────────────────────────────────


def render_prompt_comparison_tab(
    df: pd.DataFrame, score_columns: list, metric_names: dict, n_samples: int,
    auto_format_original: bool = False,
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
    st.altair_chart(boxplot, width="stretch")
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
                _render_sample(row, score_columns, metric_names, auto_format_original)
                st.divider()

    return stats


# ── Step 5: verify best prompt ──────────────────────────────────────────────


def render_verify_best_prompt_tab(
    df: pd.DataFrame, prompt_stats: pd.DataFrame, score_columns: list, metric_names: dict, n_samples: int,
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
        _render_sample_side_by_side(row, score_columns, metric_names, auto_format_original)
        st.divider()


# ── App ───────────────────────────────────────────────────────────────────────

# Each entry is (tab label, render function). Add a step by adding one tuple
# here and one render_* function above — nothing else changes.
STEPS = [
    ("1. Metric Variance", render_metric_variance_tab),
    ("2. Metric Covariance", render_metric_covariance_tab),
    ("3. Sanity Check", render_sanity_check_tab),
    ("4. Prompt Comparison", render_prompt_comparison_tab),
    ("5. Verify Best Prompt", render_verify_best_prompt_tab),
]


REQUIRED_COLUMNS = ("original_text", "prompt", "transcribed_text")


def main() -> None:
    st.set_page_config(page_title="Simplification Evaluation Dashboard", layout="wide")
    st.title("Simplification Evaluation Dashboard")

    # Defaults with the user's saved settings merged over them (src/settings).
    saved_settings = load_settings()
    config = saved_settings["dashboard"]

    with st.sidebar:
        st.header("Data")
        data_path = st.text_input("CSV path", config["data_path"])

        if not Path(data_path).is_file():
            st.error(f"Could not read '{data_path}': no such file")
            st.caption("Create it with the simplification step:")
            st.code(_simplification_command(data_path), language=None)
            st.stop()
        missing_columns = [c for c in REQUIRED_COLUMNS if c not in pd.read_csv(data_path, nrows=0).columns]
        if missing_columns:
            st.error(f"'{data_path}' has no column(s) {missing_columns} — it isn't a long-format results file.")
            st.caption("Create one with the simplification step:")
            st.code(_simplification_command(data_path), language=None)
            st.stop()
        mtime = Path(data_path).stat().st_mtime

        # Captured now, acted on at the end of this block — the actual save
        # needs active_metrics/weights/active_prompts, which don't exist yet
        # this far up, but the button itself can render anywhere.
        save_settings_clicked = st.button(
            "💾 Save current settings",
            help="Remembers Samples to show, Auto-format, the thresholds, "
                 "which metrics/prompts are active, the weight sliders and "
                 "the prompts selected for simplification — in your user "
                 "settings file. The defaults are never changed.",
        )

        # Seed once per session — same pattern as active_metrics/active_prompts
        # below — so a saved value only applies on first load and a later
        # in-session tweak is never silently overwritten by it.
        for key in ("n_samples", "auto_format_original", "low_variance_threshold",
                    "high_variance_threshold", "covariance_threshold"):
            st.session_state.setdefault(key, config[key])

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
        base_n_rows = len(df)

        # metrics.json is the ground source of truth for which metrics exist —
        # every entry gets a column here (NaN-filled if this particular CSV
        # hasn't been judged on it yet), rather than discovering metrics
        # purely by sniffing which "*_score" columns the CSV happens to have.
        # That's what makes a fresh simplification output (texts only, no
        # scores) show every active metric as "0/N judged" right away. A
        # column metrics.json doesn't govern at all (e.g. fk_normalized_score,
        # which is locally computed, not LLM-judged) is untouched here and
        # picked up below by get_score_columns(df) same as always.
        metric_defs = load_metrics()
        for slug, entry in metric_defs.items():
            if not entry.get("active", True):
                continue  # retired: show only if this CSV already has real data for it
            col = f"{slug}_score"
            if col not in df.columns:
                df[col] = float("nan")
                df[f"{slug}_reason"] = None

        # done/total judged rows for each ACTIVE metric — retired (active:
        # false) entries are excluded here on purpose: "not judged" is
        # expected for them, not something to warn about or offer to fix.
        metric_coverage = {
            f"{slug}_score": (int(df[f"{slug}_score"].notna().sum()), base_n_rows)
            for slug, entry in metric_defs.items()
            if entry.get("active", True)
        }

        all_metrics = get_score_columns(df)
        all_prompts = sorted(df["prompt"].unique())
        metric_names = _build_metric_names(all_metrics, metric_defs)

        # Built from whatever's currently available, for the "view" panels below —
        # not from active_metrics/active_prompts, so a deselected item can still be
        # inspected before deciding whether to bring it back.
        metric_descriptions = dict(_METRIC_EXPLANATIONS)
        for slug, entry in metric_defs.items():
            metric_descriptions[f"{slug}_score"] = entry["description"]
        for slug, meta in st.session_state.get("custom_metrics", {}).items():
            metric_descriptions[f"{slug}_score"] = meta["description"]

        prompt_defs = load_prompts()
        prompt_templates = {slug: entry["template"] for slug, entry in prompt_defs.items()}

        # Seed the "active"/"known" pair from saved settings, once per session
        # only (the same guard _select_active_items itself uses) — after
        # this, that function's own "keep what's still active + fold in
        # anything not in known" logic runs unmodified and does the right
        # thing whether it was just seeded from disk or is mid-session.
        if "active_metrics" not in st.session_state and "active_metrics" in config:
            st.session_state["active_metrics"] = config["active_metrics"]
            st.session_state["known_metrics"] = config.get("known_metrics", [])
        if "active_prompts" not in st.session_state and "active_prompts" in config:
            st.session_state["active_prompts"] = config["active_prompts"]
            st.session_state["known_prompts"] = config.get("known_prompts", [])

        st.header("Metrics")
        active_metrics = _select_active_items(
            all_metrics, "metrics",
            help_text="Deselect a metric to exclude it from the weighted average, "
                      "variance chart, and prompt comparison. It stays in the data "
                      "and can be reselected any time.",
            format_func=lambda col: _metric_option_label(col, metric_coverage, metric_names),
        )
        _render_run_missing_metrics_section(df, data_path, base_n_rows, metric_defs, metric_coverage)
        _render_add_metric_form(df, data_path, mtime)
        _render_save_metrics_section(data_path, base_n_rows)
        _render_clear_button("custom_metrics", "custom metrics")
        _render_definition_viewer(
            active_metrics, metric_descriptions,
            kind="metric", expander_title="📋 View metric description", selectbox_label="Metric",
            not_available="_No description available for this metric._",
            format_func=lambda col: metric_names.get(col, col),
        )

        st.caption("Higher weight = more influence on a prompt's average score.")
        saved_weights = config.get("weights", {})
        weights = {}
        for col in active_metrics:
            weight_key = f"weight_{col}"
            st.session_state.setdefault(weight_key, float(saved_weights.get(col, config["default_weight"])))
            weights[col] = st.slider(
                metric_names.get(col, col), 0.0, 10.0, step=0.5, key=weight_key,
                help=metric_descriptions.get(col),
            )

        st.header("Prompts")
        active_prompts = _select_active_items(
            all_prompts, "prompts",
            help_text="Deselect a prompt to exclude all its rows from the variance "
                      "chart and prompt comparison. Data stays put; reselect any time.",
        )
        st.session_state.setdefault("selected_prompts", [
            slug for slug in saved_settings.get("selected_prompts", [])
            if prompt_defs.get(slug, {}).get("active", True) and slug in prompt_defs
        ])
        selected_prompts = _render_selected_prompts(prompt_defs)
        _render_add_prompt_form(data_path)
        _render_definition_viewer(
            sorted(set(all_prompts) | set(prompt_templates)), prompt_templates,
            kind="prompt", expander_title="📋 View prompt template", selectbox_label="Prompt",
            not_available="No template available for this prompt.", as_code=True,
        )

        # Acted on here (not right where the button rendered, up top) because
        # weights/active_prompts don't exist yet that early in the script.
        if save_settings_clicked:
            save_settings({
                "selected_prompts": selected_prompts,
                "dashboard": {
                    "data_path": data_path,
                    "n_samples": n_samples,
                    "auto_format_original": auto_format_original,
                    "low_variance_threshold": st.session_state["low_variance_threshold"],
                    "high_variance_threshold": st.session_state["high_variance_threshold"],
                    "covariance_threshold": st.session_state["covariance_threshold"],
                    "active_metrics": active_metrics,
                    "known_metrics": all_metrics,
                    "active_prompts": active_prompts,
                    "known_prompts": all_prompts,
                    "weights": weights,
                },
            })
            st.success("Settings saved to your user settings file (src/settings/user/settings.json).")

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
            if render in (render_metric_variance_tab, render_metric_covariance_tab):
                render(df, active_metrics, metric_names, metric_descriptions, n_samples, auto_format_original)
            elif render is render_prompt_comparison_tab:
                prompt_stats = render(df, active_metrics, metric_names, n_samples, auto_format_original)
                st.session_state["prompt_stats"] = prompt_stats
            elif render is render_verify_best_prompt_tab:
                render(df, prompt_stats, active_metrics, metric_names, n_samples, auto_format_original)
            else:
                render(df, active_metrics, metric_names, n_samples, auto_format_original)


if __name__ == "__main__":
    main()
