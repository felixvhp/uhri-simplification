"""
Pkg 3: Layered "defaults + user" JSON files [last updated 2026-10-02]

Usage:
    from src.settings import load_metrics, save_prompt, load_settings

Every shared file (metrics.json, prompts.json, settings.json) exists twice:

    defaults/<name>   git-tracked. What everyone gets out of the box.
    user/<name>       gitignored. Created the first time something is saved.

Reading merges the two, with the user file winning. Writing only ever
touches the user file, so the defaults can never be changed by using the
tools — edit them by hand (and commit) to change what everyone gets.

Open Issues:
    - A user entry replaces the default entry with the same slug wholesale;
      there is no per-field merge inside one metric/prompt.

Resumption:
    A fourth shared file should be one more DefinitionsStore/SettingsStore
    instance in __init__.py rather than new load/save code.
"""

import json
from pathlib import Path
from typing import Optional

SETTINGS_DIR = Path(__file__).resolve().parent
DEFAULTS_DIR = SETTINGS_DIR / "defaults"
USER_DIR = SETTINGS_DIR / "user"


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
    """Return ``base`` updated with ``override``, merging nested dicts key by key."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


class _LayeredFile:
    """One file name that lives in both the defaults and the user folder.

    Args:
        filename: e.g. "metrics.json".
        defaults_dir: Folder holding the tracked default file.
        user_dir: Folder holding the gitignored user file. Tests pass a
            tmp_path here so they never touch the real one.
    """

    def __init__(self, filename: str, defaults_dir: Optional[Path] = None, user_dir: Optional[Path] = None):
        self.default_path = Path(defaults_dir or DEFAULTS_DIR) / filename
        self.user_path = Path(user_dir or USER_DIR) / filename

    def load_defaults(self) -> dict:
        """Load the tracked default file only."""
        return _read_json(self.default_path)

    def load_user(self) -> dict:
        """Load the user file only. Empty dict if nothing has been saved yet."""
        return _read_json(self.user_path)


class DefinitionsStore(_LayeredFile):
    """Named definitions (metrics or prompts): slug -> {"name", <content>, "active"}.

    An entry is never deleted to retire it — it's marked ``"active": false``
    (see set_active()), so its text survives and it can be brought back.

    Args:
        filename: e.g. "metrics.json".
        content_field: The one field besides name/active — "description"
            for a metric, "template" for a prompt.
        defaults_dir: See _LayeredFile.
        user_dir: See _LayeredFile.
    """

    def __init__(
        self,
        filename: str,
        content_field: str,
        defaults_dir: Optional[Path] = None,
        user_dir: Optional[Path] = None,
    ):
        super().__init__(filename, defaults_dir, user_dir)
        self.content_field = content_field

    def load_all(self) -> dict:
        """Load every definition: defaults, with user entries added or replacing them.

        Returns:
            Dict of slug -> entry, active AND retired. Filter on
            ``entry.get("active", True)`` for the ones currently in use.
        """
        return {**self.load_defaults(), **self.load_user()}

    def load_active(self) -> dict:
        """Load the active definitions, flattened to slug -> content."""
        return {
            slug: entry[self.content_field]
            for slug, entry in self.load_all().items()
            if entry.get("active", True)
        }

    def save(self, slug: str, name: str, content: str) -> None:
        """Save one definition to the user file, active by default.

        Re-saving a slug overwrites just that entry; a slug that also exists
        in the defaults is overridden for this user only.

        Args:
            slug: The definition's key (see dashboard.text_utils.slugify).
            name: Human-readable name.
            content: The judge question (metric) or template (prompt).
        """
        user = self.load_user()
        user[slug] = {"name": name, self.content_field: content, "active": True}
        _write_json(self.user_path, user)

    def set_active(self, slug: str, active: bool) -> None:
        """Retire or restore one definition, for this user only.

        Copies the entry into the user file with the flag changed, so the
        tracked defaults stay as they are.

        Args:
            slug: The definition's key. Must exist in defaults or user file.
            active: True to (re)activate, False to retire.

        Raises:
            KeyError: If ``slug`` isn't defined anywhere.
        """
        definitions = self.load_all()
        if slug not in definitions:
            raise KeyError(f"No such definition: '{slug}'")
        user = self.load_user()
        user[slug] = {**definitions[slug], "active": active}
        _write_json(self.user_path, user)


class SettingsStore(_LayeredFile):
    """A plain settings object: user values override defaults, key by key."""

    def load(self) -> dict:
        """Load the defaults with any saved user values merged over them."""
        return _deep_merge(self.load_defaults(), self.load_user())

    def save(self, changes: dict) -> None:
        """Merge ``changes`` into the user file (nested dicts merge key by key).

        Args:
            changes: Only the keys to change, e.g.
                ``{"selected_prompts": ["prompt_felix"]}``.
        """
        _write_json(self.user_path, _deep_merge(self.load_user(), changes))
