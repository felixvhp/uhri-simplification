# Pkg 3: Prompt Improvement Workflow — session notes

This README documents the prompt-improvement workflow/GUI work in this
directory specifically (`prompt_improvement_workflow.py`,
`prompt_improvement_app.py`, `evaluation.py`, `prompts.py`,
`definitions_store.py`, `metrics.json`, `prompts.json`). It's a companion
to the top-level `README.md` and `CLAUDE.md` — read those first for the
project as a whole; this one goes into more depth on just this corner of
the codebase and the decisions made while building it.

## Running the GUI

```bash
cd code
source ../venv/bin/activate                        # see docs/skills/activate-venv.md
pip install -r requirements.txt                    # first time only
streamlit run src/prompt_improvement_app.py
```

This opens the app at `http://localhost:8501` (Streamlit prints the exact
URL, and picks another port if 8501 is busy). To run it in the background
and check it's up without a browser:

```bash
cd code
streamlit run src/prompt_improvement_app.py --server.headless true --server.port 8501 &
curl -s -o /dev/null -w "%{http_code}\n" http://localhost:8501   # expect 200
```

It reads `data/cleaned.csv` by default — the long-format CSV produced by
unpivoting `simplification_results.csv` (see `analyse_simplification.ipynb`).
The CSV path is editable in the sidebar.

Everything in the GUI is local and instant *except*:

- The "➕ Add a new metric" form (judges every row on one new dimension —
  one Anthropic API call per row).
- The "➕ Add a new prompt" form (simplifies *and* judges every unique text
  on the new prompt — one OpenAI call + one Anthropic call per unique text).

Both need `ANTHROPIC_API_KEY` (and the prompt form also `OPENAI_API_KEY`)
set in your environment or a `.env` file (loaded via `python-dotenv`), and
both cost real API credits — the GUI shows a cost/row-count warning and
requires an explicit confirmation checkbox before enabling the button.

## What's in this directory

**`prompt_improvement_workflow.py`** is the logic layer: four step
functions (`check_metric_variance`, `show_sanity_samples`,
`compare_prompts`, `verify_best_prompt`), each a pure
DataFrame/dict-in-DataFrame/dict-out, plus a `__main__` block that runs
them as a CLI and prints a text report. Every step function takes a
`verbose` flag — `True` (CLI default) prints; `False` returns plain data
instead, which is what the GUI uses.

**`prompt_improvement_app.py`** is a Streamlit GUI over the exact same
step functions (`verbose=False`) — deliberately thin, UI only. Change the
evaluation logic once in `prompt_improvement_workflow.py` and both the CLI
and the GUI pick it up. Neither file is "the old one" — they're a
library/frontend split, both current. See their own module docstrings for
the full design rationale.

**`evaluation.py`** / **`prompts.py`** / **`definitions_store.py`** /
**`metrics.json`** / **`prompts.json`** — the metric and prompt definition
storage. Both `metrics.json` and `prompts.json` hold *all* definitions —
standard and custom — together in one file, told apart by a
`"standard": true/false` flag rather than which file they're in:

```json
{
  "obligations_complete": {
    "name": "obligations_complete",
    "description": "...",
    "standard": true,
    "date_added": null
  },
  "absolute_accessible": {
    "name": "absolute_accessible",
    "description": "...",
    "standard": false,
    "date_added": "2026-09-13"
  }
}
```

`date_added` is `null` for standard entries (they predate the field) and
an ISO date for every custom one. `EVAL_DIMENSIONS` / `PROMPTS` are
derived from the `standard: true` subset at import time — flattened to
`slug -> description` / `slug -> template`, the shape every existing
caller expects. `definitions_store.py`'s `DefinitionsStore` class holds
the one load/save implementation both files share (read-whole-file,
write-whole-file — so saving a new custom entry can never drop the
standard ones).

A custom metric/prompt added via the GUI lives only in
`st.session_state` until you hit "💾 Save": that merges its definition
into the JSON file (git-tracked, shared with anyone who pulls the repo)
and writes its computed values into the CSV in place (`data/` is
gitignored, so that part stays local).

## What changed in this session

Starting point: a working but duplicated/dead-code-carrying GUI + CLI
pair, plus the user's request to (1) suggest ways to verify the code and
(2) actually simplify it — remove redundant functions, merge things where
possible — with a backup taken first.

1. **Backup checkpoint** before any cleanup — full session state
   committed to `checkpoint/pre-simplification-cleanup` before touching
   anything, so every step below is independently revertible.
2. **Removed dead code**: `semantic_similarity()`/`_embedder()` in
   `evaluation.py` (zero live call sites) and the `sentence-transformers`
   dependency it pulled in.
3. **Unified duplicate FK-normalization math**: deleted
   `add_fk_normalized_score()` (a list-of-dicts reimplementation of
   `normalise_fk_grade()`'s DataFrame min-max logic); the CLI pipeline now
   builds the DataFrame earlier and reuses the one implementation.
4. **Extracted `DefinitionsStore`**: `evaluation.py` and `prompts.py` had
   hand-rolled, near-identical load-all/load-custom/save-custom trios —
   now both delegate to one shared class in `definitions_store.py`, with
   thin wrapper functions preserving the exact original public API.
5. **Merged GUI helper pairs**: `_select_active_metrics`/
   `_select_active_prompts` -> `_select_active_items()`;
   `_render_metric_descriptions`/`_render_prompt_templates` ->
   `_render_definition_viewer()`.
6. **Resample on Verify Best Prompt**: `verify_best_prompt()` pinned
   `random_state=7`, so its one sample never changed across reruns —
   unlike the Sanity Check tab, which redraws on every rerun. Dropped the
   fixed seed and added a matching "🔀 Resample" button so both tabs let
   you draw fresh samples on demand.
7. **Merged in `origin/main`'s judge refactor**: main had independently
   replaced the 5 standard judge dimensions with 9 more fine-grained ones
   (`obligations_complete`, `actors_exact`, `deontic_force`,
   `terms_of_art`, `references_precise`, `scope_and_qualifiers`,
   `additions_explanatory_only`, `plain_vocabulary`,
   `standalone_comprehension`), hardcoded directly in `evaluation.py`.
   Reconciled that with this branch's unified-JSON architecture by
   porting the 9 new dimensions into `metrics.json` as the `standard:
   true` entries (rather than reintroducing a hardcoded dict), leaving
   the 2 existing custom metrics untouched. Also caught and fixed a
   latent bug this surfaced: main's new CLI weights used
   `"<dim>__score"` (double underscore), but `_build_row()` flattens
   `score_simplification()`'s `"__"` to `"_"` — so those weights would
   have silently matched no column and done nothing.
8. **Restored the 5 retired dimensions into `metrics.json` instead of
   leaving them deleted, and made `metrics.json` the GUI's ground truth**:
   after the 5->9 switch above, `metrics.json` only had the 9 new entries —
   the original 5 (`content_preserved`, `actors_preserved`,
   `no_meaning_change`, `no_hallucination`, `accessible_and_simple`) were
   gone outright. Separately, the GUI's metric list was discovered purely
   by sniffing which `*_score` columns a loaded CSV happened to have — so a
   metric could exist in `metrics.json` with zero connection to what the
   GUI actually showed for any given file (the 9 new dimensions were
   invisible on any CSV not yet re-judged; the 5 retired ones, if a column
   of theirs no longer matched anything current, had nowhere to pull a
   description from). Fixed as one piece:
   - Added `"active": true/false` to every `metrics.json`/`prompts.json`
     entry (`DefinitionsStore.set_active()`). An entry is never deleted to
     retire it anymore — `active: false` keeps its description and
     history, and it can be reactivated later. Restored the 5 dimensions
     (recovered via git history) as `active: false`. `EVAL_DIMENSIONS`/
     `PROMPTS` filter on `standard AND active` (inline — there's no
     `load_active()`; "active" means something slightly different to each
     caller, so each filters it themselves), so a retired dimension stops
     being judged on any *new* simplification pass but keeps its
     definition and old scores.
   - `prompt_improvement_app.py`'s `main()` now seeds an empty (NaN) column
     for every *active* `metrics.json` entry a loaded CSV doesn't already
     have — before computing `all_metrics` via the same
     `get_score_columns(df)` as always. That one seeding step is what makes
     `metrics.json` the actual source of truth: every active metric is
     guaranteed a slot (new, not-yet-judged ones included) rather than
     depending on some CSV happening to already have the column; a retired
     metric is deliberately **not** seeded, so it only appears when a CSV
     already has real historical data for it, never as a phantom empty
     column. A column `metrics.json` doesn't govern at all
     (`fk_normalized_score`) is untouched and still just flows through
     `get_score_columns(df)`.
   - `metric_coverage` (judged-row-count per active metric) drives a
     "⚠️ N/total judged" label in the metrics dropdown
     (`_metric_option_label`, via `st.multiselect`'s `format_func` — the
     selection value itself is unaffected, only the displayed text) and a
     "⚠️ N metric(s) not fully judged" expander
     (`_render_run_missing_metrics_section`) with a cost-gated "🧮 Run
     missing metrics over full dataset" button. It calls the new
     `evaluate_missing_dimensions()` (evaluation.py): one Anthropic call
     per row that still has *any* gap, covering whichever of the
     incomplete metrics that row is missing — not one call per metric — so
     catching a CSV up after a judge-prompt rewrite, or computing a
     metrics.json entry for the first time, costs exactly what's actually
     missing and nothing for rows already judged.
   - `build_review_tool.py` (review-tool calibration axes) reads the same
     `standard AND active` filter directly from `metrics.json`, so it
     can't silently drift from `EVAL_DIMENSIONS` either.
9. **Shrunk `definitions_store.py`**: removed `load_active()` (and the
   `load_active_metric_definitions()`/`load_active_prompt_definitions()`
   wrappers around it) — it was only ever used by the GUI metric-list
   filtering from item 8's first draft, which got reverted. `active`
   filtering now happens inline at each of the 2-3 places that actually
   need it (`EVAL_DIMENSIONS`, `PROMPTS`, the coverage logic above), each
   with its own one-line condition, rather than a shared method guessing a
   meaning of "active" that didn't actually fit every caller the same way.
   Kept `set_active()`/`set_metric_active()`/`set_prompt_active()` (the
   real retire-don't-delete mechanism) and `load_custom()` (backs a real
   GUI feature — the "load a saved definition" dropdown — and a lot of
   existing tests).
10. **Sampling for every live LLM action**: "Run missing metrics", "Add a
    new metric", and "Add a new prompt" all used to unconditionally run
    over the *entire* dataset. Fine for a 43-record file; not once the
    dataset is much larger. Added a shared `_render_sample_controls()` —
    a slider (default: everything, so nothing changes unless you touch
    it) plus a "🎲" button that picks a new random subset — used by all
    three:
    - "Run missing metrics" samples from whichever rows actually have a
      gap (not from the whole dataset, so the slider's range is exactly
      "how many of the *missing* ones to do right now").
    - "Add a new metric" now calls `evaluate_missing_dimensions()` (a
      single-dimension call, as if the metric were a standard one with
      100% of rows missing) instead of the old `evaluate_new_metric()`,
      which is now dead code and was removed along with its tests.
      Judging a sample leaves the rest of the rows `NaN` — exactly the
      partial-coverage shape item 8 already renders as "⚠️ N/total judged"
      and can finish later via "Run missing metrics", so sampling here
      needed no new machinery, just reusing what already existed.
    - "Add a new prompt" samples which *unique texts* to generate+judge;
      a prompt with fewer rows than the others is already a normal state
      (same as deselecting a prompt), so no new machinery was needed there
      either.
    `evaluate_missing_dimensions()` gained a `rows` parameter (defaults to
    every row, so existing callers/tests are unaffected) to make all of
    this possible: restrict which row labels are even considered,
    independent of which ones currently have gaps.

11. **Human-readable metric names everywhere, instead of raw `*_score`
    columns**: the GUI used to show technical column names like
    `obligations_complete_score` verbatim in the metrics dropdown, weight
    sliders, variance chart, sample views, and "Run missing metrics"
    listing. Fixed by actually using `metrics.json`'s existing `"name"`
    field (previously left equal to the slug for every standard entry —
    only a custom metric's typed name ever differed) as the single source
    of display text, via a new `_build_metric_names()` that maps each
    `"_score"` column to its `metrics.json` name (reusing the same labels
    `build_review_tool.py` already used, for consistency — "Obligation
    strength" for `deontic_force`, "Legal terms kept" for `terms_of_art`,
    etc.) or a generic "Underscores -> Title Case" fallback for a column
    `metrics.json` doesn't govern (`fk_normalized_score`). The real column
    name is still what's selected/keyed everywhere downstream — only the
    displayed text changed, via `format_func` on the relevant
    `st.multiselect`/`st.slider`/`st.selectbox` calls.
12. **New "Metric Covariance" tab**: two metrics that correlate almost
    perfectly add little beyond one of them — same idea as low variance,
    but about a *pair*. `check_metric_covariance()`
    (prompt_improvement_workflow.py) computes pairwise correlation
    (`df[score_columns].corr()`, not raw covariance — covariance's
    magnitude depends on each metric's own scale/variance, so it can't be
    compared against one fixed threshold across different pairs the way a
    value in [-1, 1] can) and flags any pair above a tunable threshold
    (`HIGH_COVARIANCE_THRESHOLD = 0.8` by default); the GUI renders the
    full matrix plus a remove-either-one button per flagged pair.
13. **Low-variance warnings show the metric's description, not sample
    rows**: the Metric Variance tab's flagged-metric expanders used to
    repeat the same sample-row view the Sanity Check tab already exists
    for. Now they show the judge question itself (from
    `metric_descriptions`, already built in `main()`) plus a one-line
    pointer to Sanity Check for examples — less redundant, and the actual
    judge question is arguably more useful for deciding whether to remove
    a metric than a couple of sample rows were.

Every step was verified with `python -m py_compile`, a full headless
`AppTest` boot (zero exceptions), targeted functional checks (remove
buttons, date captions, resample buttons, settings save/load), and the
test suite.

## Testing

```bash
cd code
python -m pytest tests/ -q
```

As of this session: 104 passed, 5 pre-existing failures — all
`FileNotFoundError` on local data files that aren't in this checkout
(`data/search-*.xlsx` etc.), confirmed present before any of this
session's changes and unrelated to them. (A teammate's concurrent fix
during this session resolved 3 more pre-existing `compare_prompts`
failures that used to bring this to 8.)

### Architecture change: no more "standard" metrics, no more automatic judging

Two related decisions changed how `metrics.json` and judging work:

1. **`metrics.json` no longer has a `standard` field.** Every metric is
   just a metric — the only thing that distinguishes them is `active`
   (currently judged vs. retired, see item 8 above). `date_added` is also
   gone from both `metrics.json` and `prompts.json` — it only ever drove
   one GUI caption, and git history already answers "when was this added"
   more reliably than a field that has to be kept in sync by hand.
   `prompts.json` is unaffected — `standard` still means something there
   (a prompt is a thing being *tried*, so "was this curated vs. typed into
   the GUI" is still meaningful), so `DefinitionsStore` gained an
   `include_standard` flag: `True` (default) for prompts.json, `False` for
   metrics.json.
2. **Nothing judges automatically anymore.** `evaluate()`/
   `build_eval_prompt()`/`score_simplification()` no longer have an
   implicit "judge everything active" default — `dimensions` is now a
   required argument everywhere in evaluation.py; there is no fallback
   inside the library. `active_metric_definitions()` still exists as a
   plain, explicit way to ask for "every currently-active metric" — but
   calling it is now always something a *caller* opts into, never
   something that happens on its own:
   - The GUI's "Run missing metrics" and "Add a new metric" were already
     explicit (unaffected).
   - The GUI's "Add a new prompt" now builds its `dimensions` from exactly
     what's ticked in the sidebar's "Active metrics" dropdown at the
     moment you press "Generate prompt" — not "every active metric", so
     judging here can never silently cover more than what you're actually
     looking at. The button disables itself if that selection is empty.
   - `simplification.py`'s CLI batch pipeline (`run_experiment()`,
     `generate_new_prompt()` called without a GUI) still defaults to
     `active_metric_definitions()` when no `dimensions` is given, since
     there's no interactive selection to draw from there — but that
     default now lives as an explicit line in `simplification.py`, not
     hidden inside `evaluate()`.
   - `build_review_tool.py` (the separate human-calibration tool) now
     reads metrics.json's `active` flag directly (no more `standard`
     check) — it picked up 2 more axes this change (the former "custom"
     metrics `absolute_accessible`/`actors_with_context`), since there's
     no more "official battery vs. experiment" distinction to exclude
     them by.

For GUI changes, there's no dedicated Streamlit test file yet — verify by
booting headlessly:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("src/prompt_improvement_app.py", default_timeout=30)
at.run()
assert not at.exception
```

## Known gaps / not yet done

- `data/cleaned.csv` (the GUI's default CSV path) still only has the
  *retired* 5 dimensions' score columns — it predates the 9-dimension
  judge and hasn't been re-run against it. This is no longer invisible,
  though: opening the GUI on it now shows all 9 active dimensions flagged
  "⚠️ 0/N judged" with a "🧮 Run missing metrics over full dataset" button
  right there to fix it (costs real API credits — see item 8 above). A
  9-record sample pass already exists at `data/cleaned_sample10.csv` if
  you want to see the new dimensions live without spending on the full
  43-record file.
- `_render_save_metrics_section`/`_render_save_prompts_section` (GUI)
  look like another mergeable pair, but their underlying save-function
  signatures differ more than the other merged pairs did — flagged, not
  merged, since the risk/value tradeoff was judged worse.
- Saving a pending custom prompt appends rows, changing the CSV's row
  count — which retires any *other* still-unsaved custom metric or prompt
  (their stored base row count no longer matches). Saving a metric only
  adds columns, with no such side effect. Save one pending item at a time
  if you have several, to be safe.
- Settings only reliably auto-load for whatever CSV is open when the
  session starts — typing a *different* CSV path mid-session doesn't
  re-seed `active_metrics`/weights/thresholds/etc. from that file's own
  sidecar (pre-existing behavior, not new). Properly fixing it means
  detecting a data_path change and re-running every seeding step; worth
  doing if multi-CSV-per-session becomes a real workflow.
