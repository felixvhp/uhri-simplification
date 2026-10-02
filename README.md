# UHRI Simplification

Turns the legal language of UN human rights recommendations (OHCHR's Universal
Human Rights Index) into plain English, and provides two tools to check how
good the result is.

## Repository structure

```
src/
├── settings/         the only thing the three tools share
│   ├── defaults/        metrics.json, prompts.json, settings.json  (tracked)
│   └── user/            your own overrides, created on first save  (gitignored)
├── simplification/   input data -> simplified output (library + command line)
├── dashboard/        evaluation dashboard: LLM judge + Streamlit GUI
└── feedback_form/    human review form (static page or small server)
tests/
data/                 gitignored; holds your input and output files
```

The three tools never import each other. What one needs to tell another goes
through the files in `src/settings/` (`tests/test_isolation.py` enforces this),
and results travel between them as data files in `data/`.

## Setup

```bash
python3 -m venv venv            # Python 3.11+
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env            # then fill in OPENAI_API_KEY and ANTHROPIC_API_KEY
```

Run everything from the repo root. Tests need no API keys:

```bash
python -m pytest
```

## Settings

Each of the three files has a tracked default and an optional personal copy:

| File | Holds |
|---|---|
| `metrics.json` | what the LLM judge and human reviewers score |
| `prompts.json` | the simplification prompt templates |
| `settings.json` | models, selected prompt(s), data paths, dashboard preferences |

Reading merges `defaults/` with `user/`, your values winning. Saving — from the
dashboard or with `src.settings.save_*` — only ever writes `user/`, which is
gitignored. To change what everyone gets, edit the file in `defaults/` and commit.

## Simplifying data

```bash
python -m src.simplification data/email_recs.xlsx                     # -> data/email_recs.simplified.xlsx
python -m src.simplification data/email_recs.xlsx -o data/out.json    # formats can differ
python -m src.simplification in.csv --text-column body --prompt summarise
```

From Python:

```python
from src.simplification import simplify, simplify_file

simplify_file("data/email_recs.xlsx", "data/out.csv")
rows = simplify([{"id": 7, "Text": "The Committee recommends ..."}])   # list of dicts
df = simplify(my_dataframe, text_column="recommendation")              # DataFrame
text = simplify("The Committee recommends ...")                        # one string
```

- **Formats:** `.xlsx`, `.csv`, `.tsv`, `.json` (a list of objects), `.jsonl`.
- **Output:** every original column or key is kept, and the simplification is
  added as `simplified`.
- **Prompt:** the one selected in the settings (`selected_prompts`). If several
  are selected you get a warning first, then one column per prompt
  (`simplified_<prompt>`). The command line asks before continuing; `--yes` skips that.
- **Text column:** `Text`, `text` or `original_text` is found automatically;
  otherwise pass `--text-column` / `text_column=`.
- **Cost:** one OpenAI call per row and prompt. Try `--sample 5` first.

Installed with `pip install .`, the same interface is available as
`from uhri_simplification.simplification import simplify`.

## Evaluating prompts (dashboard)

The dashboard compares prompts on a results file with one row per text and
prompt. Produce it with the long layout:

```bash
python -m src.simplification data/email_recs.xlsx -o data/cleaned.csv --layout long
streamlit run src/dashboard/app.py
```

In the dashboard you can:

- score the outputs with the LLM judge ("Run missing metrics" — live Anthropic calls, asks first);
- add a metric, weight metrics, and compare prompts across five report tabs;
- add a prompt and choose which prompt(s) the simplification step uses next;
- save all of that to your user settings.

A new prompt has no outputs until you rerun the simplification command above.
`python -m src.dashboard.workflow` prints the same report in the terminal.

## Collecting human feedback (feedback form)

Reviewers score each simplification on the same metrics the judge uses.

```bash
python -m src.feedback_form.make_data --sample 80   # pick records from data/cleaned.csv
python -m src.feedback_form.serve                   # http://127.0.0.1:8010, stores submissions
python -m src.feedback_form.build                   # or: one self-contained HTML file to send around
```

Submissions land in `src/feedback_form/exports/` (gitignored).

## Conventions

- Commit messages: `[pkg3] short description`.
- Each code file starts with a docstring: purpose and last-updated date, usage,
  open issues, and where the next person should resume.
- Public functions have Google-style docstrings (Args / Returns).
- Update or delete comments when the code they describe changes.
