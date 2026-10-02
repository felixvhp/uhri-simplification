# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UHRI simplification: turns OHCHR (Office of the High Commissioner for Human Rights) legal human rights recommendations into plain English, and provides a dashboard and a feedback form to evaluate the result. See `README.md` for usage.

## Running Code

Activate the project venv first (see `docs/skills/activate-venv.md`). Run everything from the repo root.

```bash
python -m pytest                                              # tests; no API keys needed
python -m src.simplification data/email_recs.xlsx             # simplify a file (live OpenAI calls)
python -m src.simplification IN -o data/cleaned.csv --layout long   # results file for dashboard / feedback form
streamlit run src/dashboard/app.py                            # evaluation dashboard
python -m src.dashboard.workflow                              # same report in the terminal
python -m src.feedback_form.serve                             # feedback form server
```

Simplifying costs one OpenAI call per row and prompt; judging in the dashboard costs one Anthropic call per row. Don't trigger either without being asked.

## Dependencies

Declared in `pyproject.toml`, pinned in `requirements.txt` (Python 3.11+). API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) go in a `.env` at the repo root — copy `.env.example`. Tests mock both clients.

## Architecture

```
src/settings/         shared layer: defaults/ (tracked) + user/ (gitignored) JSON files and their loader
src/simplification/   core.py (interface), io.py (file formats), __main__.py (CLI)
src/dashboard/        app.py (Streamlit GUI), workflow.py (report logic), evaluation.py (LLM judge), text_utils.py
src/feedback_form/    make_data.py, build.py, serve.py, review_template.html
tests/
```

**Hard rule:** `simplification/`, `dashboard/` and `feedback_form/` never import each other. They share only `src/settings` (`metrics.json`, `prompts.json`, `settings.json`). `tests/test_isolation.py` fails on a crossing. If one tool needs to tell another something, add a key to the settings files.

**Settings layering:** reads merge `defaults/` with `user/` (user wins); every save goes to `user/` only. Never write to `defaults/` from code.

**Data flow:**
```
input file (xlsx/csv/tsv/json/jsonl)
    → src.simplification            → same rows + "simplified" column(s)      (wide layout)
                                    → one row per text × prompt               (long layout: original_text, prompt, transcribed_text)
long-layout CSV → src.dashboard     → judge scores written back into the CSV
long-layout CSV → src.feedback_form → review_data.json → page / server
```

Imports inside `src/` are relative, so the code works both as `src.*` from a checkout and as `uhri_simplification.*` when pip-installed. `dashboard/app.py` has a small bootstrap at the top because Streamlit runs it as a script.

## Data Files

`data/` is gitignored — only the download script is in the repo.

- `data/email_recs.xlsx` — 43-record OHCHR export used as test input
- `data/cleaned.csv` — long-layout results; default input of the dashboard and feedback form
- `data/download_public_data.sh` — downloads the full OHCHR export (run from inside `data/`)

## Conventions

**Commit format:** `[pkgN] short description` (e.g., `[pkg3] add runner for different system prompt options`)

**File header docstring:**
```python
"""
Pkg #N: Component name [last updated YYYY-MM-DD]

Usage: XXX

Open Issues:
- ...

Resumption:
- Where the next person should start
"""
```

**Function docstrings:** standard Google-style with Args/Returns sections.
