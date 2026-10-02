# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UHRI (UN Human Rights Initiative) processes OHCHR (Office of the High Commissioner for Human Rights) export data. The system simplifies legal human rights recommendations into plain English and supports semantic search over them.

## Running Code

Activate the project venv first (see `docs/skills/activate-venv.md`).

**Semantic search:**
```bash
cd code
python src/search.py "education in Sub-Saharan Africa" --top-k 25 --show-components
python src/search.py "query" --geo-json ../data/export_full_en.json --allow-geo-candidates
```

**LLM simplification** (live OpenAI + Anthropic calls — costs API credits):
```bash
cd code
python src/simplification.py
```

**Prompt improvement workflow** (reads `data/cleaned.csv`; local, no API calls):
```bash
cd code
python src/prompt_improvement_workflow.py          # CLI report, press Enter between steps
streamlit run src/prompt_improvement_app.py        # GUI
```

**Tests:**
```bash
cd code
python -m pytest tests/
```

**Linting** (ruff is not in `requirements.txt`; install it separately):
```bash
ruff check code/src/
```

## Dependencies

Pinned in `code/requirements.txt` (needs Python 3.11+). Main ones:
- `pandas`, `openpyxl` — XLSX/JSON parsing
- `numpy`, `scikit-learn` — semantic search (TF-IDF, LSI, cosine similarity)
- `openai` — simplification model; `anthropic` — LLM judge
- `streamlit`, `altair` — prompt improvement GUI
- `python-dotenv` — loads API keys from `.env`

API keys (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`) go in a `.env` at the repo root — copy `.env.example`. Tests mock both clients and need no keys.

## Architecture

Three packages, each prefixed in commits as `[pkg1]`, `[pkg2]`, `[pkg3]`:

**Pkg 1 — Semantic Search (`code/src/search.py`)**
Ranks term candidates against a query using a weighted combination of TF-IDF (45%), LSI (24%), stem-Jaccard (21%), character n-grams (7%), and frequency (3%). No LLM required. Reads word/phrase/topic candidates from `.tsv` files and geo terms from OHCHR JSON exports.

**Pkg 3 — Data Ingestion & Simplification (`code/src/`)**
Three tightly coupled modules that must stay in sync:
- `uhri_record.py` — `UHRIRecord` dataclass (5 fields: `annotation_id`, `text`, `country`, `recommending_body`, `type`)
- `Dataloader.py` — parses OHCHR XLSX/JSON exports into `List[UHRIRecord]`; handles the OHCHR typo `"Reccomending Body"`
- `simplification.py` — runs each record through OpenAI with the prompt templates in `prompts.json`, scores each output with the Anthropic judge (`evaluation.py`, metrics in `metrics.json`); outputs long-format CSV

The prompt-improvement workflow, GUI and review tool built on top of that CSV are documented in `code/src/README.md`.

**Data flow:**
```
OHCHR Export (XLSX/JSON)
    → Dataloader.load_records()
    → List[UHRIRecord]
    ├─→ simplify() + judge → data/cleaned.csv            (live API calls)
    └─→ rank_candidates()  → ranked JSON results          (fast, local)
```

**Advanced search prototype** at `docs/prototypes/search.py` combines search with topic filtering from OHCHR JSON exports — more complete than the `code/src/search.py` version.

## Data Files

`data/` is gitignored — only the download script is in the repo; the rest must be copied in or generated.

- `data/email_recs.xlsx` — 43-record test dataset; default input of `simplification.py`
- `data/search-9_3_2026_592.xlsx` — export used by `tests/dataloader_test.py`
- `data/cleaned.csv` — long-format simplification results (one row per text × prompt); written by `simplification.py`, read by the workflow CLI and GUI
- `data/gpt_topics.tsv`, `data/short_words_gpt.tsv`, `data/short_phrases_gpt.tsv` — term candidates required by `search.py`
- `data/export_full_en.json` — full OHCHR export (~370 MB); used for geo terms and production search
- `data/download_public_data.sh` — downloads the full export (run from inside `data/`)

Hardcoded paths in source files are relative to `code/` (e.g., `../data/email_recs.xlsx`).

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
