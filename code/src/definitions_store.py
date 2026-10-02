"""
Pkg 3: Shared "definitions store" helper [added 2026-09-18]

Usage:
    from src.definitions_store import DefinitionsStore

Both evaluation.py (metrics.json: the 5 standard EVAL_DIMENSIONS plus any
custom metric added via the GUI) and prompts.py (prompts.json: the 5
standard PROMPTS plus any custom prompt) persist a "standard vs custom"
definitions file with the exact same shape and rules:
    - One JSON file: slug -> {"name", <content field>, "standard", "date_added"}.
    - Read: missing file -> {}.
    - "Custom" means an entry's `standard` isn't True.
    - Saving a custom entry always reads/writes the *whole* file (never
      just the custom subset), so it can never accidentally drop the
      standard ones sharing the file.
    - `date_added` defaults to today; only a backfill passes it explicitly.

Before this, evaluation.py and prompts.py each hand-rolled their own
load_all/load_custom/save_custom trio — identical logic, differing only in
the JSON file path and what the one variable field is called ("description"
for metrics, "template" for prompts). DefinitionsStore(path, content_field)
captures exactly that one difference; both modules now just declare a
module-level instance and thin wrapper functions instead of re-implementing
the same ~40 lines twice.

Resumption:
    A third definitions file (e.g. for something beyond metrics/prompts)
    should use this too rather than writing a fourth copy of the same logic.
"""

import json
from datetime import date
from pathlib import Path
from typing import Optional


class DefinitionsStore:
    """A "standard + custom, together" JSON definitions file.

    Args:
        path: Where the JSON file lives — typically a module-level constant,
            e.g. ``Path(__file__).parent / "metrics.json"``.
        content_field: Name of the one field these definitions carry besides
            name/standard/date_added — "description" for metrics, "template"
            for prompts.
    """

    def __init__(self, path: Path, content_field: str):
        self.default_path = path
        self.content_field = content_field

    def load_all(self, path: Optional[Path] = None) -> dict:
        """Load every definition — standard AND custom, together.

        Args:
            path: JSON file to read. Defaults to this store's path — read at
                call time (not bound as a parameter default) so tests can
                pass a tmp_path file and never touch the real one.

        Returns:
            Dict of slug -> {"name", content_field, "standard", "date_added"}.
            Empty dict if the file doesn't exist yet.
        """
        path = path or self.default_path
        if not Path(path).exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def load_custom(self, path: Optional[Path] = None) -> dict:
        """Load CUSTOM definitions only — anything not marked `standard: true`.

        Use load_all() instead if you need the standard ones too.
        """
        return {
            slug: entry
            for slug, entry in self.load_all(path).items()
            if not entry.get("standard")
        }

    def save_custom(
        self,
        slug: str,
        name: str,
        content: str,
        path: Optional[Path] = None,
        date_added: Optional[str] = None,
    ) -> None:
        """Persist one custom definition (name + content).

        Reads and rewrites the *entire* file (via load_all(), not
        load_custom()) so saving a new custom entry can never accidentally
        drop the standard ones that live in the same file — read-modify-
        write, keyed by slug (re-saving the same slug overwrites just that
        entry, including its date_added — a re-save is "still current", not
        backdated to when it was first added).

        Args:
            slug: The definition's key (see text_utils.slugify).
            name: Human-readable name, as typed in the GUI.
            content: The one variable field's value (a judge question for a
                metric, a template string for a prompt).
            path: JSON file to write. Defaults to this store's path — read
                at call time, same reasoning as load_all().
            date_added: ISO date string (YYYY-MM-DD). Defaults to today —
                the normal case, a fresh save from the GUI. Pass this
                explicitly only to backfill an older entry.
        """
        path = path or self.default_path
        definitions = self.load_all(path)
        definitions[slug] = {
            "name": name,
            self.content_field: content,
            "standard": False,
            "date_added": date_added or date.today().isoformat(),
        }
        with open(path, "w") as f:
            json.dump(definitions, f, indent=2, sort_keys=True)
            f.write("\n")
