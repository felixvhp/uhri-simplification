#!/usr/bin/env python3
"""
Build the Plain-English Translation Review tool.

Injects two things into review_template.html:
  * REVIEW_DATA : the original + AI-draft pairs (from review_data.json)
  * DIMENSIONS : generated VERBATIM from EVAL_DIMENSIONS in src/evaluation.py,
                   so the human review axes can never drift from the judge's.

Usage:
    python review/build_review_tool.py
Writes review/translation_review.html (self-contained, opens in any browser).

To repoint at a different dataset, regenerate review_data.json as a list of {id, original, variants:{<prompt>: <text>}} and rerun.
"""
import ast
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent                       # code/
EVAL_PY = ROOT / "src" / "evaluation.py"
TEMPLATE = HERE / "review_template.html"
DATA = HERE / "review_data.json"
OUT = HERE / "translation_review.html"

# Short, reviewer-facing labels for each judge dimension (key -> label).
LABELS = {
    "obligations_complete": "Obligations complete",
    "actors_exact": "Actors exact",
    "deontic_force": "Obligation strength",
    "terms_of_art": "Legal terms kept",
    "references_precise": "References precise",
    "scope_and_qualifiers": "Scope & qualifiers",
    "additions_explanatory_only": "Additions explanatory only",
    "plain_vocabulary": "Plain vocabulary",
    "standalone_comprehension": "Stands alone",
}
# Which dimensions are accessibility (judged on the draft alone) vs fidelity.
ACCESSIBILITY = {"plain_vocabulary", "standalone_comprehension"}
GROUP_FIDELITY = "Fidelity: checked against the original record"
GROUP_ACCESS = "Accessibility: judging the draft on its own"


def load_eval_dimensions() -> dict:
    """Parse EVAL_DIMENSIONS out of evaluation.py without importing it."""
    tree = ast.parse(EVAL_PY.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            getattr(t, "id", None) == "EVAL_DIMENSIONS" for t in node.targets
        ):
            return ast.literal_eval(node.value)
    raise SystemExit("EVAL_DIMENSIONS not found in evaluation.py")


def main() -> None:
    dims = load_eval_dimensions()
    missing = set(dims) - set(LABELS)
    if missing:
        raise SystemExit(f"Add short labels for new dimensions: {sorted(missing)}")

    axes = [
        {
            "key": k,
            "t": LABELS[k],
            "d": v,
            "group": GROUP_ACCESS if k in ACCESSIBILITY else GROUP_FIDELITY,
        }
        for k, v in dims.items()
    ]

    data = json.loads(DATA.read_text())
    html = TEMPLATE.read_text()
    html = html.replace("/*__DIMENSIONS__*/[]", json.dumps(axes, ensure_ascii=False, indent=2), 1)
    html = html.replace("/*__REVIEW_DATA__*/[]", json.dumps(data, ensure_ascii=False), 1)
    assert "__DIMENSIONS__" not in html and "__REVIEW_DATA__" not in html, "injection failed"
    OUT.write_text(html)
    print(f"Built {OUT.name}: {len(axes)} axes, {len(data)} records, {len(html)//1024} KB")


if __name__ == "__main__":
    main()
