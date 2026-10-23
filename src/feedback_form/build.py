#!/usr/bin/env python3
"""
Pkg 3: Build the Plain-English Translation Review tool (feedback form)
[last updated 2026-10-23]

Injects three things into review_template.html:
  * FORM_CONFIG : which prompt's draft is reviewed (the first of the settings'
                   selected_prompts) and the reviewer-facing name of every
                   prompt (prompts.json "name"), via build_form_config()
  * REVIEW_DATA : the original + AI-draft pairs (all of review_data.jsonl;
                   the baked page samples client-side, so keep it to a few
                   hundred records — the served page scales, this one does not)
  * DIMENSIONS : the active metrics of the shared settings (src/settings,
                   metrics.json) — the same set the dashboard's LLM judge
                   scores, so the human review axes can never drift from it.
                   A retired metric (active: false) is excluded.

Usage:
    python -m src.feedback_form.build        (from the repo root)
Writes src/feedback_form/translation_review.html (self-contained, opens in any browser).

To repoint at a different dataset, regenerate review_data.jsonl with
make_data.py (one {id, original, variants:{<prompt>: <text>}} per line) and rerun.

Resumption:
    A metric's reviewer-facing label is its "name" in metrics.json and its
    group is its optional "group" ("accessibility", else fidelity) — change
    them there, not here.
"""
import json
from pathlib import Path

from ..settings import load_metrics, load_prompts, load_settings
from .records import RecordStore

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "review_template.html"
DATA = HERE / "review_data.jsonl"
OUT = HERE / "translation_review.html"

# Accessibility metrics are judged on the draft alone; fidelity ones against the original.
GROUP_FIDELITY = "Fidelity: checked against the original record"
GROUP_ACCESS = "Accessibility: judging the draft on its own"


def build_axes() -> list[dict]:
    """The review axes: one per active metric, with its label and group.

    Shared by main() (baked into the standalone page) and serve.py's
    /api/axes (served live), so both delivery modes show identical questions.

    Returns:
        List of {"key", "t" (label), "d" (judge question), "group"}.

    Raises:
        SystemExit: If no metric is active.
    """
    axes = [
        {
            "key": slug,
            "t": entry.get("name", slug),
            "d": entry["description"].strip(),
            "group": GROUP_ACCESS if entry.get("group") == "accessibility" else GROUP_FIDELITY,
        }
        for slug, entry in load_metrics().items()
        if entry.get("active", True)
    ]
    if not axes:
        raise SystemExit("No active metrics found in the settings' metrics.json")
    return axes


def build_form_config() -> dict:
    """Which draft the reviewer judges, and how the prompts are labelled.

    Shared by main() (baked) and serve.py's /api/config (live), so the form
    follows the shared settings instead of hard-coding a prompt name.

    Returns:
        {"primary": <slug of the first selected prompt, or "" if none>,
         "labels": {<slug>: <name>} for every prompt defined}.
    """
    selected = load_settings().get("selected_prompts", [])
    return {
        "primary": selected[0] if selected else "",
        "labels": {slug: entry.get("name", slug) for slug, entry in load_prompts().items()},
    }


def main() -> None:
    axes = build_axes()
    config = build_form_config()
    data = RecordStore(DATA).all()
    html = TEMPLATE.read_text()
    html = html.replace("/*__DIMENSIONS__*/[]", json.dumps(axes, ensure_ascii=False, indent=2), 1)
    html = html.replace("/*__REVIEW_DATA__*/[]", json.dumps(data, ensure_ascii=False), 1)
    html = html.replace("/*__FORM_CONFIG__*/{}", json.dumps(config, ensure_ascii=False), 1)
    assert not any(k in html for k in ("__DIMENSIONS__", "__REVIEW_DATA__", "__FORM_CONFIG__")), "injection failed"
    OUT.write_text(html)
    print(f"Built {OUT.name}: {len(axes)} axes, {len(data)} records, primary draft "
          f"'{config['primary']}', {len(html)//1024} KB")


if __name__ == "__main__":
    main()
