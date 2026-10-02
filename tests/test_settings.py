"""Tests for src/settings — the layered defaults + user files.

Every test builds its own defaults/ and user/ folders under tmp_path, so the
real settings files are never touched.

Run from the repo root:
    python -m pytest tests/test_settings.py
"""

import json

import pytest

import src.settings as settings
from src.settings.store import DefinitionsStore, SettingsStore


def _dirs(tmp_path, filename, defaults):
    """Create defaults/<filename> with ``defaults`` and return (defaults_dir, user_dir)."""
    defaults_dir, user_dir = tmp_path / "defaults", tmp_path / "user"
    defaults_dir.mkdir()
    (defaults_dir / filename).write_text(json.dumps(defaults))
    return defaults_dir, user_dir


def _definitions(tmp_path, defaults=None):
    defaults_dir, user_dir = _dirs(tmp_path, "defs.json", defaults or {})
    return DefinitionsStore("defs.json", "description", defaults_dir, user_dir)


DEFAULT_DEFS = {
    "kept": {"name": "Kept", "description": "default question", "active": True},
    "retired": {"name": "Retired", "description": "old question", "active": False},
}


# ── DefinitionsStore ──────────────────────────────────────────────────────────


def test_load_all_is_defaults_when_nothing_saved(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    assert store.load_all() == DEFAULT_DEFS
    assert store.load_user() == {}


def test_load_all_empty_when_no_files_exist(tmp_path):
    store = DefinitionsStore("missing.json", "description", tmp_path / "d", tmp_path / "u")
    assert store.load_all() == {}


def test_save_writes_user_file_and_leaves_defaults_untouched(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    store.save("tone", "Tone", "Is it formal?")

    assert store.load_user() == {"tone": {"name": "Tone", "description": "Is it formal?", "active": True}}
    assert store.load_defaults() == DEFAULT_DEFS
    assert set(store.load_all()) == {"kept", "retired", "tone"}


def test_save_creates_the_user_folder(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    assert not store.user_path.parent.exists()
    store.save("tone", "Tone", "Is it formal?")
    assert store.user_path.is_file()


def test_user_entry_overrides_default_with_same_slug(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    store.save("kept", "Kept", "my own question")
    assert store.load_all()["kept"]["description"] == "my own question"
    assert store.load_defaults()["kept"]["description"] == "default question"


def test_save_keeps_other_user_entries(tmp_path):
    store = _definitions(tmp_path)
    store.save("a", "A", "first")
    store.save("b", "B", "second")
    assert set(store.load_user()) == {"a", "b"}


def test_load_active_excludes_retired_and_flattens(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    assert store.load_active() == {"kept": "default question"}


def test_set_active_retires_a_default_for_this_user_only(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    store.set_active("kept", False)

    assert store.load_all()["kept"]["active"] is False
    assert store.load_all()["kept"]["description"] == "default question"  # text survives
    assert store.load_defaults()["kept"]["active"] is True


def test_set_active_can_reactivate(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    store.set_active("retired", True)
    assert "retired" in store.load_active()


def test_set_active_raises_for_unknown_slug(tmp_path):
    store = _definitions(tmp_path, DEFAULT_DEFS)
    with pytest.raises(KeyError):
        store.set_active("nope", False)


# ── SettingsStore ─────────────────────────────────────────────────────────────

DEFAULT_SETTINGS = {
    "selected_prompts": ["a"],
    "dashboard": {"n_samples": 2, "weights": {"x_score": 5.0}},
}


def _settings(tmp_path):
    defaults_dir, user_dir = _dirs(tmp_path, "settings.json", DEFAULT_SETTINGS)
    return SettingsStore("settings.json", defaults_dir, user_dir)


def test_settings_fall_back_to_defaults(tmp_path):
    assert _settings(tmp_path).load() == DEFAULT_SETTINGS


def test_saved_settings_override_only_what_was_saved(tmp_path):
    store = _settings(tmp_path)
    store.save({"dashboard": {"n_samples": 7}})

    loaded = store.load()
    assert loaded["dashboard"]["n_samples"] == 7
    assert loaded["dashboard"]["weights"] == {"x_score": 5.0}  # untouched default survives
    assert loaded["selected_prompts"] == ["a"]


def test_saving_settings_never_writes_the_defaults(tmp_path):
    store = _settings(tmp_path)
    store.save({"selected_prompts": ["b", "c"]})

    assert store.load_defaults() == DEFAULT_SETTINGS
    assert store.load_user() == {"selected_prompts": ["b", "c"]}
    assert store.load()["selected_prompts"] == ["b", "c"]  # lists are replaced, not merged


def test_nested_weights_merge_key_by_key(tmp_path):
    store = _settings(tmp_path)
    store.save({"dashboard": {"weights": {"y_score": 2.0}}})
    assert store.load()["dashboard"]["weights"] == {"x_score": 5.0, "y_score": 2.0}


# ── Module-level API over the real default files ──────────────────────────────


@pytest.fixture
def isolated_user_dir(tmp_path, monkeypatch):
    """Point the module's three stores at an empty user folder; keep the real defaults."""
    for store in (settings._metrics, settings._prompts, settings._settings):
        monkeypatch.setattr(store, "user_path", tmp_path / store.user_path.name)
    return tmp_path


def test_default_files_are_consistent(isolated_user_dir):
    """Every selected prompt is defined, and every definition has its content field."""
    assert settings.selected_prompts()  # at least one, and all resolvable
    assert all("template" in entry for entry in settings.load_prompts().values())
    assert all("description" in entry for entry in settings.load_metrics().values())
    assert settings.active_metrics()


def test_default_prompts_all_have_a_text_slot(isolated_user_dir):
    for slug, entry in settings.load_prompts().items():
        assert "{text}" in entry["template"], slug


def test_saving_a_prompt_and_selecting_it_reaches_selected_prompts(isolated_user_dir):
    settings.save_prompt("bullets", "Bullets", "As bullets: {text}")
    settings.save_settings({"selected_prompts": ["bullets"]})

    assert settings.selected_prompts() == {"bullets": "As bullets: {text}"}
    assert (isolated_user_dir / "prompts.json").is_file()
    assert (isolated_user_dir / "settings.json").is_file()


def test_selected_prompts_raises_for_undefined_slug(isolated_user_dir):
    settings.save_settings({"selected_prompts": ["does_not_exist"]})
    with pytest.raises(KeyError):
        settings.selected_prompts()


def test_save_metric_reaches_active_metrics(isolated_user_dir):
    settings.save_metric("tone", "Tone", "Is it formal?")
    assert settings.active_metrics()["tone"] == "Is it formal?"
    settings.set_metric_active("tone", False)
    assert "tone" not in settings.active_metrics()
