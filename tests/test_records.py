"""Tests for the feedback form's record store and its /api/records sampling.

Run from the repo root:
    python -m pytest tests/test_records.py
"""

import json
import os
import time
from pathlib import Path
from unittest.mock import patch

import pytest

from src.feedback_form import serve
from src.feedback_form.records import RecordStore


def write_jsonl(path: Path, n: int) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for i in range(n):
            f.write(json.dumps({"id": i, "original": f"orig {i}", "variants": {"p": f"draft {i}"}}) + "\n")
        f.write("\n")   # trailing blank line must be ignored


@pytest.fixture
def store(tmp_path):
    path = tmp_path / "review_data.jsonl"
    write_jsonl(path, 100)
    return RecordStore(path)


def test_count_ignores_blank_lines(store):
    assert store.count() == 100


def test_sample_is_deterministic_and_in_file_order(store):
    a = store.sample(10, seed=7)
    b = store.sample(10, seed=7)
    assert [r["id"] for r in a] == [r["id"] for r in b]
    assert len(a) == 10
    assert [r["id"] for r in a] == sorted(r["id"] for r in a)
    assert a[0]["variants"] == {"p": f"draft {a[0]['id']}"}


def test_different_seeds_differ(store):
    assert store.sample(10, seed=1) != store.sample(10, seed=2)


def test_zero_or_oversized_n_returns_everything(store):
    assert len(store.sample(0)) == 100
    assert len(store.sample(1000, seed=3)) == 100
    assert store.all() == store.sample(0)


def test_index_refreshes_when_the_file_changes(store):
    assert store.count() == 100
    time.sleep(0.01)
    write_jsonl(store.path, 5)
    os.utime(store.path, None)
    assert store.count() == 5
    assert [r["id"] for r in store.sample(0)] == [0, 1, 2, 3, 4]


SETTINGS = {"feedback_form": {"sample_size": 20, "max_sample_size": 30}}


def test_sample_records_defaults_caps_and_echoes_seed(store):
    with patch("src.feedback_form.serve.load_settings", return_value=SETTINGS):
        default = serve.sample_records(store, None, 5)
        assert default["total"] == 100 and default["sample_size"] == 20 and default["seed"] == 5

        capped = serve.sample_records(store, 80, 5)
        assert capped["sample_size"] == 30

        everything = serve.sample_records(store, 0, 5)
        assert everything["sample_size"] == 30   # "all" still respects the cap

        fresh = serve.sample_records(store, 3, None)
        assert isinstance(fresh["seed"], int) and len(fresh["records"]) == 3


def test_sample_records_returns_all_when_file_is_small(tmp_path):
    path = tmp_path / "small.jsonl"
    write_jsonl(path, 4)
    with patch("src.feedback_form.serve.load_settings", return_value=SETTINGS):
        out = serve.sample_records(RecordStore(path), None, 1)
    assert out["sample_size"] == 4 and [r["id"] for r in out["records"]] == [0, 1, 2, 3]


def test_int_param():
    assert serve._int_param({"n": ["12"]}, "n") == 12
    assert serve._int_param({"n": ["x"]}, "n") is None
    assert serve._int_param({}, "n") is None
