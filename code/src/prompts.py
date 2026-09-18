import json
from datetime import date
from pathlib import Path
from typing import Optional

from src.uhri_record import UHRIRecord
"""
Pkg 3: Prompt templates for simplifying UHRI texts [last updated 2026-09-18]

Usage:
    from src.prompts import PROMPTS, DEFAULT_PROMPT, build_prompt

All prompt definitions — the 5 standard templates AND any custom ones added
via the GUI — live together in one file: prompts.json, next to this module
(git-tracked, so it's shared with anyone who pulls the repo). An entry's
`standard: true/false` flag is what tells the two apart, not which file
it's in — mirrors evaluation.py/metrics.json exactly (see that module's
docstring for the full reasoning).

PROMPTS is derived from prompts.json at import time (the `standard: true`
subset, flattened to slug -> template string) — kept in that flat shape
because build_prompt()/simplify() and every existing caller expect it.
`date_added` is null for the 5 standard ones (they predate the field) and
an ISO date for every custom one, same convention as metrics.json.

Resumption:
    load_custom_prompt_definitions()/save_custom_prompt_definition() (added
    2026-09-13, reshaped into one unified prompts.json 2026-09-18) let the
    prompt-improvement Streamlit GUI persist a custom prompt's name/template
    permanently instead of it living only in st.session_state — mirrors
    evaluation.py's load_custom_metric_definitions()/
    save_custom_metric_definition() for custom metrics.
    save_custom_prompt_definition() always reads and rewrites the *whole*
    file, never just the custom subset, so saving a new custom prompt can
    never accidentally drop the standard 5. The generated *rows* for a
    saved prompt go into the data CSV instead (data/ is gitignored) — see
    simplification.save_custom_prompt_rows_to_csv().
"""

# Where ALL prompt definitions live — the 5 standard templates and any
# custom ones, together. Next to this file (not relative to the cwd),
# tracked in git so a saved prompt's template is shared with anyone who
# pulls the repo.
PROMPTS_PATH = Path(__file__).parent / "prompts.json"


def load_all_prompt_definitions(path: Optional[Path] = None) -> dict:
    """Load every prompt definition — standard AND custom, together.

    Args:
        path: JSON file to read. Defaults to PROMPTS_PATH — read at call
            time (not bound as the parameter default) so tests can
            monkeypatch that module constant to a tmp_path file and never
            touch the real one.

    Returns:
        Dict of slug -> {"name", "template", "standard", "date_added"}.
        Empty dict if the file doesn't exist yet.
    """
    path = path or PROMPTS_PATH
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


# The subset of prompts.json marked `standard: true` — flattened to
# slug -> template, which is the shape build_prompt()/simplify() and every
# existing caller/test expect. A custom (standard: false) prompt in the
# same file only gets used when generate_new_prompt() is asked to.
PROMPTS = {
    slug: entry["template"]
    for slug, entry in load_all_prompt_definitions().items()
    if entry.get("standard")
}

DEFAULT_PROMPT = PROMPTS["prompt_felix"]


def build_prompt(record: UHRIRecord, template: str = DEFAULT_PROMPT) -> str:
    """
    Build a prompt from a record and a template.

    Args:
        record: The UHRIRecord to simplify.
        template: Prompt template with {type}, {body}, {country}, {text} slots.
                  Defaults to the module-level SYSTEM_PROMPT.

    Returns:
        The fully-formatted prompt string, ready to send to the model.
    """
    return template.format(
        type="recommendation" if record.is_recommendation else "observation",
        body=record.recommending_body,
        country=record.country,
        text=record.text,
    )


def load_custom_prompt_definitions(path: Optional[Path] = None) -> dict:
    """Load saved CUSTOM prompt definitions only — the standard 5 excluded.

    "Custom" means anything in prompts.json *not* marked `standard: true` —
    i.e. whatever's been added via the GUI's "Add a new prompt" form. Use
    load_all_prompt_definitions() instead if you need the standard ones too.

    Args:
        path: JSON file to read. Defaults to PROMPTS_PATH — read at call
            time, same reasoning as load_all_prompt_definitions().

    Returns:
        Dict of slug -> {"name", "template", "standard": False,
        "date_added"}. Empty dict if none saved (or the file doesn't exist).
    """
    return {
        slug: entry
        for slug, entry in load_all_prompt_definitions(path).items()
        if not entry.get("standard")
    }


def save_custom_prompt_definition(
    slug: str,
    name: str,
    template: str,
    path: Optional[Path] = None,
    date_added: Optional[str] = None,
) -> None:
    """Persist one custom prompt's definition (name + template).

    Reads and rewrites the *entire* prompts.json (via
    load_all_prompt_definitions(), not load_custom_prompt_definitions()) so
    saving a new custom prompt can never accidentally drop the standard 5
    that live in the same file — read-modify-write, keyed by slug (re-saving
    the same slug overwrites just that entry, including its date_added — a
    re-save is treated as "still current", not backdated to when it was
    first added).

    Args:
        slug: The prompt's ``prompt`` column value (see text_utils.slugify).
        name: Human-readable name, as typed in the GUI.
        template: The prompt template string.
        path: JSON file to write. Defaults to PROMPTS_PATH — read at call
            time, same reasoning as load_all_prompt_definitions().
        date_added: ISO date string (YYYY-MM-DD). Defaults to today — the
            normal case, a fresh save from the GUI. Pass this explicitly
            only to backfill an older entry.
    """
    path = path or PROMPTS_PATH
    definitions = load_all_prompt_definitions(path)
    definitions[slug] = {
        "name": name,
        "template": template,
        "standard": False,
        "date_added": date_added or date.today().isoformat(),
    }
    with open(path, "w") as f:
        json.dump(definitions, f, indent=2, sort_keys=True)
        f.write("\n")
