# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

UHRI (UN Human Rights Initiative) processes OHCHR (Office of the High Commissioner for Human Rights) export data. The system simplifies legal human rights recommendations into plain English and supports semantic search over them.

## Running Code

**Semantic search:**
```bash
cd code
python src/search.py "education in Sub-Saharan Africa" --top-k 25 --show-components
python src/search.py "query" --geo-json data/export_full_en.json --allow-geo-candidates
```

**LLM simplification** (requires Ollama installed and running):
```bash
cd code
python src/llm-text-simplification.py
```

**Tests:**
```bash
cd code
pytest tests/
# or directly:
python tests/dataloader_test.py
```

**Linting** (ruff cache exists, suggesting ad-hoc use):
```bash
ruff check code/src/
```

## Dependencies

No `requirements.txt` exists. Required packages (inferred from imports):
- `pandas` — XLSX/JSON parsing
- `numpy`, `scikit-learn` — semantic search (TF-IDF, LSI, cosine similarity)
- `ollama` — local LLM interface (requires [Ollama](https://ollama.com/) daemon + `llama3:latest` model)

## Architecture

Three packages, each prefixed in commits as `[pkg1]`, `[pkg2]`, `[pkg3]`:

**Pkg 1 — Semantic Search (`code/src/search.py`)**
Ranks term candidates against a query using a weighted combination of TF-IDF (45%), LSI (24%), stem-Jaccard (21%), character n-grams (7%), and frequency (3%). No LLM required. Reads word/phrase/topic candidates from `.tsv` files and geo terms from OHCHR JSON exports.

**Pkg 3 — Data Ingestion & Simplification (`code/src/`)**
Three tightly coupled modules that must stay in sync:
- `uhri_record.py` — `UHRIRecord` dataclass (5 fields: `annotation_id`, `text`, `country`, `recommending_body`, `type`)
- `Dataloader.py` — parses OHCHR XLSX/JSON exports into `List[UHRIRecord]`; handles the OHCHR typo `"Reccomending Body"`
- `llm-text-simplification.py` — runs each record through Ollama with one of 5 prompt templates; outputs CSV

**Data flow:**
```
OHCHR Export (XLSX/JSON)
    → Dataloader.load_records()
    → List[UHRIRecord]
    ├─→ simplify()         → simplification_results.csv  (slow: ~2–5 min/record)
    └─→ rank_candidates()  → ranked JSON results          (fast, local)
```

**Advanced search prototype** at `docs/prototypes/search.py` combines search with topic filtering from OHCHR JSON exports — more complete than the `code/src/search.py` version.

## Data Files

- `data/email_recs.xlsx` — 43-record test dataset used in unit tests and LLM dev
- `data/export_full_en.json` — full OHCHR export (~344 MB); used for geo terms and production search
- `data/download_public_data.sh` — script to download public datasets

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
