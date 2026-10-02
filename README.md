# UHRI Project

## Repository Structure

```
uhri-dev/
├── code/                # Source code and implementation
│   ├── scripts/            # helper scripts
│   ├── src/                # source code
│   ├── tests/              # tests
├── data/                # Datasets and data resources
├── docs/                # Documentation
│   ├── prototypes/         # Protoypes for external presentation
│   ├── spec/               # Write-up of requirements, specifications, etc.
└── README.md            
```

## Running Tests

```bash
cd code
python -m pytest tests/
# or run a single file directly:
python tests/dataloader_test.py
python tests/test_evaluations.py
```

## Running the Prompt Improvement GUI

A Streamlit GUI for pkg3's prompt-improvement workflow lives at
`code/src/prompt_improvement_app.py`. It walks through the same four-step
report as `code/src/prompt_improvement_workflow.py` — metric variance,
sanity check, prompt comparison, best-prompt verification — interactively in
the browser, and additionally lets you tweak metric weights, remove metrics
from the analysis, and add new LLM-judged metrics on the fly.

```bash
cd code
pip install -r requirements.txt   # first time only — installs streamlit
streamlit run src/prompt_improvement_app.py
```

This opens the app at `http://localhost:8501` (Streamlit prints the exact
URL, and picks another port if that one's busy). It reads
`data/cleaned.csv` by default — editable in the sidebar to point at a
different long-format CSV.

Everything in the GUI is local and instant *except* the sidebar's "Add a new
metric" form: that calls the Anthropic API live, once per row, so it needs
`ANTHROPIC_API_KEY` set in your environment (or a `.env` file — the code
loads it via `python-dotenv`) and costs real API credits. The GUI shows a
cost/row-count warning and requires an explicit confirmation checkbox before
that button is enabled.

A custom metric added this way lives only in the browser session until you
hit "💾 Save" next to it: that writes its judge question into
`code/src/metrics.json` (tracked in git — shared with anyone who
pulls the repo) and its computed scores into the loaded CSV in place
(`data/` is gitignored, so that part stays local to your machine). After
saving, it's an ordinary column the next time the CSV loads — the "Add a
new metric" form also has a "📂 Load a saved metric definition" dropdown to
reuse a saved judge question without retyping it, e.g. against a new
dataset.

## Documentation Guidelines

### Code Files

- Include a header comment with file purpose and key dependencies
- Document public functions/classes with descriptions, parameters, and return types
- delete/change comments when updating referenced code
- for each sub-system, keep a "open issues" list / "next work item" (i.e. the place where the next person should resume working)

**Example:**

```python
"""
Pkg #1: Retrieval Implementation [last updated xxx]

Usage: XXX

Open Issues:
- ...

Resumption:
- ...

"""

def extract_filter(
    manual_tag: UHRITag,
    query: str,
) -> UHRIFilter:
    """
    Extract context tags and information from a user query.
    
    Parses the input query string and combines it with manual tags
    to identify relevant UHRI tags and constraints.
    
    Args:
        manual_tag: UHRITag from hard-coded extraction
        query: User query string to parse for tags
        
    Returns:
        UHRIFilter object representing extracted tags and information
    """
```

### Commit Messages

- e.g. `[pkg1] added missing acronyms to vectorization`
