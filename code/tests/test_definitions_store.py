"""Tests for definitions_store.py — the shared "standard + custom" JSON store
used by both evaluation.py (metrics.json) and prompts.py (prompts.json).

Run from code/:
    python -m pytest tests/test_definitions_store.py
"""

import json
from datetime import date

import pytest

from src.definitions_store import DefinitionsStore


def _store(tmp_path, name="defs.json"):
    return DefinitionsStore(tmp_path / name, content_field="description")


def test_load_all_empty_when_file_missing(tmp_path):
    store = _store(tmp_path)
    assert store.load_all() == {}


def test_load_custom_empty_when_file_missing(tmp_path):
    store = _store(tmp_path)
    assert store.load_custom() == {}


def test_load_active_empty_when_file_missing(tmp_path):
    store = _store(tmp_path)
    assert store.load_active() == {}


def test_load_custom_excludes_standard_entries(tmp_path):
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "date_added": None},
        "b": {"name": "b", "description": "d2", "standard": False, "date_added": "2026-09-13"},
    }))
    assert set(store.load_custom()) == {"b"}
    assert set(store.load_all()) == {"a", "b"}


def test_load_active_excludes_retired_entries(tmp_path):
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "active": False, "date_added": None},
        "b": {"name": "b", "description": "d2", "standard": True, "active": True, "date_added": None},
    }))
    assert set(store.load_active()) == {"b"}
    # load_all() is unfiltered — the retired entry survives there.
    assert "a" in store.load_all()


def test_load_active_treats_missing_flag_as_active(tmp_path):
    # Entries written before "active" existed have no such key — they must
    # default to active rather than silently disappearing.
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "date_added": None},
    }))
    assert set(store.load_active()) == {"a"}


def test_save_custom_roundtrips_and_defaults_active_true(tmp_path):
    store = _store(tmp_path)
    store.save_custom("tone", "Tone", "Is it formal?", date_added="2026-09-13")
    assert store.load_custom() == {
        "tone": {
            "name": "Tone",
            "description": "Is it formal?",
            "standard": False,
            "active": True,
            "date_added": "2026-09-13",
        }
    }


def test_save_custom_defaults_date_added_to_today(tmp_path):
    store = _store(tmp_path)
    store.save_custom("tone", "Tone", "Is it formal?")
    assert store.load_custom()["tone"]["date_added"] == date.today().isoformat()


def test_save_custom_merges_without_clobbering(tmp_path):
    store = _store(tmp_path)
    store.save_custom("a", "A", "desc a")
    store.save_custom("b", "B", "desc b")
    assert set(store.load_custom()) == {"a", "b"}


def test_save_custom_overwrites_same_slug(tmp_path):
    store = _store(tmp_path)
    store.save_custom("a", "A", "first")
    store.save_custom("a", "A", "second")
    defs = store.load_custom()
    assert len(defs) == 1
    assert defs["a"]["description"] == "second"


def test_save_custom_preserves_standard_entries(tmp_path):
    # Regression guard: saving a new custom entry must read/write the
    # *whole* file, not just the custom subset, or it would silently drop
    # the standard entries sharing the file.
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "date_added": None},
    }))
    store.save_custom("b", "B", "d2", date_added="2026-09-18")
    all_defs = store.load_all()
    assert "a" in all_defs and all_defs["a"]["standard"] is True
    assert "b" in all_defs and all_defs["b"]["standard"] is False


def test_set_active_retires_without_deleting(tmp_path):
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "active": True, "date_added": None},
    }))
    store.set_active("a", False)
    all_defs = store.load_all()
    assert all_defs["a"]["active"] is False
    assert all_defs["a"]["description"] == "d"  # untouched
    assert "a" not in store.load_active()


def test_set_active_can_reactivate(tmp_path):
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "active": False, "date_added": None},
    }))
    store.set_active("a", True)
    assert "a" in store.load_active()


def test_set_active_raises_for_unknown_slug(tmp_path):
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({}))
    with pytest.raises(KeyError):
        store.set_active("does_not_exist", False)


def test_set_active_does_not_affect_other_entries(tmp_path):
    store = _store(tmp_path)
    store.default_path.write_text(json.dumps({
        "a": {"name": "a", "description": "d", "standard": True, "active": True, "date_added": None},
        "b": {"name": "b", "description": "d2", "standard": True, "active": True, "date_added": None},
    }))
    store.set_active("a", False)
    all_defs = store.load_all()
    assert all_defs["a"]["active"] is False
    assert all_defs["b"]["active"] is True
