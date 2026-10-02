"""
Pkg 3: Shared "definitions store" helper

Usage:
    from src.definitions_store import DefinitionsStore

Both evaluation.py (metrics.json) and prompts.py (prompts.json) persist a
definitions file with the same shape and rules. A module-level
DefinitionsStore(path, content_field) instance plus thin wrapper functions
is all either module needs.

The two files aren't identical, though: prompts.json still distinguishes
"standard" (built-in) templates from custom ones — a prompt is a thing
being *tried*, so it's meaningful to know which ones were curated versus
typed into the GUI. metrics.json dropped that distinction: a metric is
either currently judged or it isn't (see `active`), with no separate
notion of "official" vs "user-added" — judging only ever happens when a
caller explicitly asks for specific dimensions, so there's no "default
battery" for "standard" to gate anymore. `include_standard` controls which
behavior a given store gets.

Shape and rules:
    - One JSON file: slug -> {"name", <content field>, "active", and
      "standard" too if `include_standard`}.
    - Read: missing file -> {}.
    - An entry is never deleted just because it's no longer wanted — e.g. a
      dimension retired in a judge-prompt rewrite, or a custom one that
      didn't pan out. It's marked `"active": false` instead (see
      set_active()), so its definition survives. Callers filter on this
      themselves via `entry.get("active", True)` (missing key => active) —
      there's no load_active() here, since "active" means something
      different to each caller (PROMPTS wants `standard AND active`; the
      GUI's "run missing metrics" wants active regardless of standard).
    - Saving a new entry always reads/writes the *whole* file (never a
      subset), so it can never accidentally drop any other entry sharing
      the file.

Resumption:
    A third definitions file should use this too rather than writing a
    fourth copy of the same logic.
"""

import json
from pathlib import Path
from typing import Optional


class DefinitionsStore:
    """A "standard + custom, together" JSON definitions file.

    Args:
        path: Where the JSON file lives — typically a module-level constant,
            e.g. ``Path(__file__).parent / "metrics.json"``.
        content_field: Name of the one field these definitions carry besides
            name/active (and standard, if used) — "description" for
            metrics, "template" for prompts.
        include_standard: Whether this file distinguishes standard (built-
            in) entries from custom ones at all. prompts.json does
            (True, the default); metrics.json doesn't (False) — see this
            module's docstring for why.
    """

    def __init__(self, path: Path, content_field: str, include_standard: bool = True):
        self.default_path = path
        self.content_field = content_field
        self.include_standard = include_standard

    def load_all(self, path: Optional[Path] = None) -> dict:
        """Load every definition together.

        Args:
            path: JSON file to read. Defaults to this store's path — read at
                call time (not bound as a parameter default) so tests can
                pass a tmp_path file and never touch the real one.

        Returns:
            Dict of slug -> {"name", content_field, "active", ...}. Empty
            dict if the file doesn't exist yet.
        """
        path = path or self.default_path
        if not Path(path).exists():
            return {}
        with open(path) as f:
            return json.load(f)

    def load_custom(self, path: Optional[Path] = None) -> dict:
        """Load CUSTOM definitions only — anything not marked `standard: true`.

        Only meaningful when `include_standard` is True. Use load_all()
        instead if you need the standard ones too.
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
    ) -> None:
        """Persist one definition (name + content), active by default.

        Reads and rewrites the *entire* file (via load_all(), not
        load_custom()) so saving a new entry can never accidentally drop
        any other one that lives in the same file — read-modify-write,
        keyed by slug (re-saving the same slug overwrites just that entry).

        Args:
            slug: The definition's key (see text_utils.slugify).
            name: Human-readable name, as typed in the GUI.
            content: The one variable field's value (a judge question for a
                metric, a template string for a prompt).
            path: JSON file to write. Defaults to this store's path — read
                at call time, same reasoning as load_all().
        """
        path = path or self.default_path
        definitions = self.load_all(path)
        entry = {"name": name, self.content_field: content, "active": True}
        if self.include_standard:
            entry["standard"] = False
        definitions[slug] = entry
        with open(path, "w") as f:
            json.dump(definitions, f, indent=2, sort_keys=True)
            f.write("\n")

    def set_active(self, slug: str, active: bool, path: Optional[Path] = None) -> None:
        """Retire or restore one definition in place, without touching the rest.

        Flips just the `active` flag on an existing entry (standard or
        custom) — its name/content/standard are untouched. Use this instead
        of deleting an entry from the JSON file: a retired standard
        dimension (e.g. one dropped in a judge-prompt rewrite) keeps its
        description and can be reactivated later by calling this again with
        active=True.

        Args:
            slug: The definition's key. Must already exist in the file.
            active: True to (re)activate, False to retire/hide it.
            path: JSON file to write. Defaults to this store's path.

        Raises:
            KeyError: If ``slug`` isn't in the file.
        """
        path = path or self.default_path
        definitions = self.load_all(path)
        if slug not in definitions:
            raise KeyError(f"No such definition: '{slug}'")
        definitions[slug]["active"] = active
        with open(path, "w") as f:
            json.dump(definitions, f, indent=2, sort_keys=True)
            f.write("\n")
