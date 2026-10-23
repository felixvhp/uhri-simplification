"""
Pkg 3: Shared settings — the only layer the three tools have in common
[last updated 2026-10-02]

Usage:
    from src.settings import (
        load_metrics, save_metric, set_metric_active,
        load_prompts, selected_prompts, save_prompt, set_prompt_active,
        load_settings, save_settings, slugify,
    )

simplification/, dashboard/ and feedback_form/ never import each other.
Whatever one of them needs to tell another goes through the three files
here (e.g. the dashboard saves a new prompt and selects it; the next
simplification run picks it up):

    metrics.json    what the LLM judge and human reviewers score
    prompts.json    simplification prompt templates
    settings.json   models, selected prompts, paths, dashboard preferences

Each has a tracked default in defaults/ and an optional gitignored override
in user/ — see store.py.

Resumption:
    New shared knob: add it to defaults/settings.json and read it with
    load_settings(). Don't add a direct import between the tools instead.
"""

from .store import DEFAULTS_DIR, USER_DIR, DefinitionsStore, SettingsStore, slugify

_metrics = DefinitionsStore("metrics.json")
_prompts = DefinitionsStore("prompts.json")
_settings = SettingsStore("settings.json")


# ── Metrics ───────────────────────────────────────────────────────────────────


def load_metrics() -> dict:
    """Every metric definition (active and inactive): slug -> {"name", "description", "active", ...}."""
    return _metrics.load()


def save_metric(slug: str, name: str, description: str) -> None:
    """Save one metric (name + judge question) to the user metrics file."""
    _metrics.save_definition(slug, {"name": name, "description": description})


def set_metric_active(slug: str, active: bool) -> None:
    """Deactivate or reactivate one metric for this user (see DefinitionsStore.set_active())."""
    _metrics.set_active(slug, active)


# ── Prompts ───────────────────────────────────────────────────────────────────


def load_prompts() -> dict:
    """Every prompt definition (active and inactive): slug -> {"name", "template", "active"}."""
    return _prompts.load()


def save_prompt(slug: str, name: str, template: str) -> None:
    """Save one prompt (name + template) to the user prompts file."""
    _prompts.save_definition(slug, {"name": name, "template": template})


def set_prompt_active(slug: str, active: bool) -> None:
    """Deactivate or reactivate one prompt for this user (see DefinitionsStore.set_active())."""
    _prompts.set_active(slug, active)


def selected_prompts() -> dict:
    """The prompts the simplification step runs: slug -> template.

    Which ones are selected is ``selected_prompts`` in settings.json.

    Raises:
        KeyError: If a selected slug has no definition in prompts.json.
    """
    definitions = load_prompts()
    selected = load_settings().get("selected_prompts", [])
    missing = [slug for slug in selected if slug not in definitions]
    if missing:
        raise KeyError(f"Selected prompt(s) not defined in prompts.json: {missing}")
    return {slug: definitions[slug]["template"] for slug in selected}


# ── Settings ──────────────────────────────────────────────────────────────────


def load_settings() -> dict:
    """The default settings with any saved user settings merged over them."""
    return _settings.load()


def save_settings(changes: dict) -> None:
    """Merge ``changes`` into the user settings file (the defaults are never written)."""
    _settings.save(changes)
