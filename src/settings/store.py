"""
Pkg 3: Layered "defaults + user" JSON files [last updated 2026-10-23]

Usage:
    from src.settings import load_metrics, save_prompt, load_settings

Every shared file (metrics.json, prompts.json, settings.json) exists twice:

    defaults/<name>   git-tracked. What everyone gets out of the box.
    user/<name>       gitignored. Created the first time something is saved.

Reading merges the two, with the user file winning. Writing only ever
touches the user file, so the defaults can never be changed by using the
tools — edit them by hand (and commit) to change what everyone gets.

Open Issues:
    - A user file can change or add a field, but never remove one from the
      defaults (make a definition inactive with "active": false instead).
    - User files saved before 2026-10-23 hold full copies of the entries
      they touched, which keep winning over later default edits. Delete
      user/metrics.json and user/prompts.json once to reset them.
    - Decide whether save_definition() should slugify its key itself
      instead of trusting the caller.

Resumption:
    A fourth shared file should be one more DefinitionsStore/SettingsStore
    instance in __init__.py rather than new load/save code.
"""

import json
import re
from pathlib import Path
from typing import Optional

SETTINGS_DIR = Path(__file__).resolve().parent
DEFAULTS_DIR = SETTINGS_DIR / "defaults"
USER_DIR = SETTINGS_DIR / "user"


def slugify(name: str) -> str:
    """Turn a free-text name into a lowercase, underscore-separated slug.

    This is the key format of metrics.json and prompts.json, so any tool
    that turns a typed name into a definition key uses this — e.g. a new
    metric's column prefix, or a new prompt's ``prompt`` value.

    Args:
        name: Human-entered name, e.g. "Tone: formal enough?".

    Returns:
        Slug, e.g. "tone_formal_enough".

    Raises:
        ValueError: If ``name`` has no alphanumeric characters to slugify.
    """
    slug = re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")
    if not slug:
        raise ValueError(f"'{name}' has no usable characters for a slug")
    return slug


def _read_json(path: Path) -> dict:
    """Read a JSON object from ``path``; a missing file is an empty dict."""
    if not Path(path).exists():
        return {}
    with open(path) as f:
        return json.load(f)


def _write_json(path: Path, content: dict) -> None:
    """Write ``content`` to ``path`` as sorted, indented JSON, creating the folder."""
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(content, f, indent=2, sort_keys=True)
        f.write("\n")


def _deep_merge(base: dict, override: dict) -> dict:
    """Merge ``override`` into ``base`` and return the result as a new dict.

    Keys only in one side are kept. Where both sides have a key, ``override``
    is kept, except when both values are dicts: those are merged recursively.

    ``base`` is not modified, but only its top level is copied: nested dicts
    that ``override`` doesn't touch are shared with ``base``.

    Args:
        base: The starting values, e.g. the defaults.
        override: Values that take precedence, e.g. the user's saved settings.

    Returns:
        A new dict with ``override`` merged over ``base``.
    """
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class _LayeredFile:
    """One file name that lives in both the defaults and the user folder.

    Loading gives the defaults with the user file merged over them.
    Saving writes into the user file only, never touching the defaults.

    Args:
        filename: e.g. "metrics.json".
        defaults_dir: Folder holding the tracked default file.
        user_dir: Folder holding the gitignored user file. Tests pass a
            tmp_path here so they never touch the real one.
    """

    def __init__(self, filename: str, defaults_dir: Optional[Path] = None, user_dir: Optional[Path] = None):
        self.default_path = Path(defaults_dir or DEFAULTS_DIR) / filename
        self.user_path = Path(user_dir or USER_DIR) / filename

    def load_user(self) -> dict:
        """Load the user file only. Empty dict if nothing has been saved yet."""
        return _read_json(self.user_path)

    def load(self) -> dict:
        """Load the defaults with any saved user values merged over them."""
        defaults = _read_json(self.default_path)
        user = self.load_user()
        return _deep_merge(defaults, user)

    def save(self, changes: dict) -> None:
        """Merge ``changes`` into the user file only, never the defaults.

        Nested dicts merge key by key (see _deep_merge()); lists and other
        values are replaced whole, so saving ``selected_prompts`` replaces
        the list rather than adding to it.

        Args:
            changes: Only the keys to change, e.g.
                ``{"selected_prompts": ["prompt_1"]}``.
        """
        merged_update = _deep_merge(self.load_user(), changes)
        _write_json(self.user_path, merged_update)


class DefinitionsStore(_LayeredFile):
    """Named definitions (metrics or prompts): slug -> {"name", "description" | "template", "active", ...}.

    An entry is never deleted to deactivate it — it's marked ``"active": false``
    (see set_active()), so its text survives and it can be brought back.
    """

    def save_definition(self, slug: str, fields: dict) -> None:
        """Save one definition to the user file, active by default.

        Re-saving a slug merges into that entry: fields not passed keep
        their current value. A slug that also exists in the defaults is
        overridden for this user only, field by field.

        Args:
            slug: The definition's key (see slugify()).
            fields: The fields to set on that entry and their values, e.g.
                ``{"name": "Tone", "description": "Is it formal?"}``.
        """
        self.save({slug: {"active": True, **fields}})

    def set_active(self, slug: str, active: bool) -> None:
        """Deactivate or reactivate one definition, for this user only.

        Stores only ``{"active": ...}`` for this slug in the user file; the
        rest of the entry keeps coming from the defaults, so later edits to
        a default's wording still show up.

        Args:
            slug: The definition's key. Must exist in defaults or user file.
            active: True to (re)activate, False to deactivate.

        Raises:
            KeyError: If ``slug`` isn't defined anywhere.
        """
        definitions = self.load()
        if slug not in definitions:
            raise KeyError(f"No such definition: '{slug}'")
        self.save({slug: {"active": active}})


class SettingsStore(_LayeredFile):
    """A plain settings object: user values override defaults, key by key.

    Sections (``simplification``, ``dashboard``, ``feedback_form``) merge
    key by key, so saving one dashboard value keeps the rest; lists such as
    ``selected_prompts`` are replaced whole.
    """
