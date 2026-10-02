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
source ~/Documents/Venvs/un_project/bin/activate   # or your own venv — see pip install below
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

Every step was verified with `python -m py_compile`, a full headless
`AppTest` boot (zero exceptions), targeted functional checks (remove
buttons, date captions, resample buttons, settings save/load), and the
test suite.

## Testing

```bash
cd code
python -m pytest tests/ -q
```

As of this session: 75 passed, 8 pre-existing failures — all
`FileNotFoundError` on local data files that aren't in this checkout
(`data/search-*.xlsx` etc.), confirmed present before any of this
session's changes and unrelated to them.

For GUI changes, there's no dedicated Streamlit test file yet — verify by
booting headlessly:

```python
from streamlit.testing.v1 import AppTest
at = AppTest.from_file("src/prompt_improvement_app.py", default_timeout=30)
at.run()
assert not at.exception
```

## Known gaps / not yet done

- `definitions_store.py` has no test file of its own — it's only
  exercised indirectly through `evaluation.py`'s and `prompts.py`'s
  wrapper-function tests.
- `_render_save_metrics_section`/`_render_save_prompts_section` (GUI)
  look like another mergeable pair, but their underlying save-function
  signatures differ more than the other merged pairs did — flagged, not
  merged, since the risk/value tradeoff was judged worse.
- The top-level `README.md`'s "Running the Prompt Improvement GUI"
  section predates the `metrics.json`/`prompts.json` unification and
  still mentions `code/src/custom_metrics.json`, which no longer exists —
  worth updating there too.
