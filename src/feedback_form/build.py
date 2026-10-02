#!/usr/bin/env python3
"""
Pkg 3: Build the Plain-English Translation Review tool (feedback form)
[last updated 2026-10-02]

Injects two things into review_template.html:
  * REVIEW_DATA : the original + AI-draft pairs (from review_data.json)
  * DIMENSIONS : the active metrics of the shared settings (src/settings,
                   metrics.json) — the same set the dashboard's LLM judge
                   scores, so the human review axes can never drift from it.
                   A retired metric (active: false) is excluded.

Usage:
    python -m src.feedback_form.build        (from the repo root)
Writes src/feedback_form/translation_review.html (self-contained, opens in any browser).

To repoint at a different dataset, regenerate review_data.json with
make_data.py (a list of {id, original, variants:{<prompt>: <text>}}) and rerun.

Resumption:
    A metric's reviewer-facing label is its "name" in metrics.json and its
    group is its optional "group" ("accessibility", else fidelity) — change
    them there, not here.
"""
import json
from pathlib import Path

from ..settings import load_metrics

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "review_template.html"
DATA = HERE / "review_data.json"
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


def main() -> None:
    axes = build_axes()
    data = json.loads(DATA.read_text())
    html = TEMPLATE.read_text()
    html = html.replace("/*__DIMENSIONS__*/[]", json.dumps(axes, ensure_ascii=False, indent=2), 1)
    html = html.replace("/*__REVIEW_DATA__*/[]", json.dumps(data, ensure_ascii=False), 1)
    assert "__DIMENSIONS__" not in html and "__REVIEW_DATA__" not in html, "injection failed"
    OUT.write_text(html)
    print(f"Built {OUT.name}: {len(axes)} axes, {len(data)} records, {len(html)//1024} KB")


if __name__ == "__main__":
    main()
