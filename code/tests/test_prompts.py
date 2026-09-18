"""Tests for prompts.py's prompt-definition persistence (prompts.json: standard + custom, together).

Run from code/:
    python -m pytest tests/test_prompts.py
"""

import json
from datetime import date

from src.prompts import (
    PROMPTS,
    load_all_prompt_definitions,
    load_custom_prompt_definitions,
    save_custom_prompt_definition,
)


def test_load_all_prompt_definitions_empty_when_file_missing(tmp_path):
    assert load_all_prompt_definitions(tmp_path / "does_not_exist.json") == {}


def test_load_custom_prompt_definitions_empty_when_file_missing(tmp_path):
    assert load_custom_prompt_definitions(tmp_path / "does_not_exist.json") == {}


def test_load_custom_prompt_definitions_excludes_standard_entries(tmp_path):
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({
        "prompt_felix": {"name": "prompt_felix", "template": "t", "standard": True, "date_added": None},
        "bullet_points": {"name": "Bullet Points", "template": "t2", "standard": False, "date_added": "2026-09-13"},
    }))
    custom_only = load_custom_prompt_definitions(path)
    assert set(custom_only) == {"bullet_points"}
    assert set(load_all_prompt_definitions(path)) == {"prompt_felix", "bullet_points"}


def test_save_then_load_custom_prompt_definition_roundtrips(tmp_path):
    path = tmp_path / "prompts.json"
    save_custom_prompt_definition(
        "bullet_points", "Bullet Points", "Rewrite as bullets: {text}", path, date_added="2026-09-13"
    )
    assert load_custom_prompt_definitions(path) == {
        "bullet_points": {
            "name": "Bullet Points",
            "template": "Rewrite as bullets: {text}",
            "standard": False,
            "date_added": "2026-09-13",
        }
    }


def test_save_custom_prompt_definition_defaults_date_added_to_today(tmp_path):
    path = tmp_path / "prompts.json"
    save_custom_prompt_definition("bullet_points", "Bullet Points", "t", path)
    saved = load_custom_prompt_definitions(path)["bullet_points"]
    assert saved["date_added"] == date.today().isoformat()


def test_save_custom_prompt_definition_merges_without_clobbering(tmp_path):
    path = tmp_path / "prompts.json"
    save_custom_prompt_definition("a", "A", "template a", path)
    save_custom_prompt_definition("b", "B", "template b", path)
    assert set(load_custom_prompt_definitions(path)) == {"a", "b"}


def test_save_custom_prompt_definition_overwrites_same_slug(tmp_path):
    path = tmp_path / "prompts.json"
    save_custom_prompt_definition("a", "A", "first version", path)
    save_custom_prompt_definition("a", "A", "second version", path)
    defs = load_custom_prompt_definitions(path)
    assert len(defs) == 1
    assert defs["a"]["template"] == "second version"


def test_save_custom_prompt_definition_preserves_standard_entries(tmp_path):
    # Regression guard: saving a new custom prompt must read/write the
    # *whole* file, not just the custom subset, or it would silently drop
    # the standard 5 that live in the same prompts.json.
    path = tmp_path / "prompts.json"
    path.write_text(json.dumps({
        "prompt_felix": {"name": "prompt_felix", "template": "t", "standard": True, "date_added": None},
    }))
    save_custom_prompt_definition("bullet_points", "Bullet Points", "t2", path, date_added="2026-09-18")
    all_defs = load_all_prompt_definitions(path)
    assert "prompt_felix" in all_defs and all_defs["prompt_felix"]["standard"] is True
    assert "bullet_points" in all_defs and all_defs["bullet_points"]["standard"] is False


def test_prompts_excludes_custom_prompts():
    # PROMPTS is derived from prompts.json at import time, filtered to
    # standard: true — it must never silently pick up a custom prompt saved
    # to the real file, or the CLI batch pipeline would start using it too.
    assert set(PROMPTS) <= set(load_all_prompt_definitions())
    for slug in PROMPTS:
        assert load_all_prompt_definitions()[slug]["standard"] is True


if __name__ == "__main__":
    tests = [
        test_prompts_excludes_custom_prompts,
        # The tmp_path-based persistence tests below need pytest's fixture
        # injection to run — they'll show FAIL here (no t() args to supply
        # it with), same as the equivalent tests in test_evaluations.py.
        # Use `pytest` to run them.
        test_load_all_prompt_definitions_empty_when_file_missing,
        test_load_custom_prompt_definitions_empty_when_file_missing,
        test_load_custom_prompt_definitions_excludes_standard_entries,
        test_save_then_load_custom_prompt_definition_roundtrips,
        test_save_custom_prompt_definition_defaults_date_added_to_today,
        test_save_custom_prompt_definition_merges_without_clobbering,
        test_save_custom_prompt_definition_overwrites_same_slug,
        test_save_custom_prompt_definition_preserves_standard_entries,
    ]
    for t in tests:
        try:
            t()
            print(f"  PASS  {t.__name__}")
        except Exception as e:
            print(f"  FAIL  {t.__name__}: {e}")
