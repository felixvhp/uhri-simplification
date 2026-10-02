"""
Pkg 3: Reading and writing record files [last updated 2026-10-02]

Usage:
    from src.simplification.io import read_rows, write_rows
    rows = read_rows("data/email_recs.xlsx")      # list of dicts
    write_rows(rows, "data/out.json")

The format is chosen by file extension: .xlsx, .csv, .tsv, .json (a list of
objects), .jsonl. Tabular files are read as text so IDs and codes come back
exactly as written; JSON values keep their types and nesting.

Open Issues:
    - .json must be a list of objects; an export that wraps the list in an
      outer object needs unwrapping first.

Resumption:
    New format: one entry each in _READERS and _WRITERS.
"""

import json
from pathlib import Path
from typing import Union

import pandas as pd


def _read_table(path: Path, **kwargs) -> list:
    """Read a delimited text file as strings; empty cells become ""."""
    return pd.read_csv(path, dtype=str, keep_default_na=False, **kwargs).to_dict("records")


def _read_xlsx(path: Path) -> list:
    """Read the first sheet of a workbook as strings; empty cells become ""."""
    return pd.read_excel(path, dtype=str).fillna("").to_dict("records")


def _read_json(path: Path) -> list:
    """Read a JSON file holding a list of objects."""
    rows = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(rows, list) or not all(isinstance(row, dict) for row in rows):
        raise ValueError(f"{path.name} must contain a list of objects, one per record.")
    return rows


def _read_jsonl(path: Path) -> list:
    """Read one JSON object per line, skipping blank lines."""
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_json(rows: list, path: Path) -> None:
    path.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _write_jsonl(rows: list, path: Path) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


_READERS = {
    ".xlsx": _read_xlsx,
    ".csv": _read_table,
    ".tsv": lambda path: _read_table(path, sep="\t"),
    ".json": _read_json,
    ".jsonl": _read_jsonl,
}

_WRITERS = {
    ".xlsx": lambda rows, path: pd.DataFrame(rows).to_excel(path, index=False),
    ".csv": lambda rows, path: pd.DataFrame(rows).to_csv(path, index=False),
    ".tsv": lambda rows, path: pd.DataFrame(rows).to_csv(path, index=False, sep="\t"),
    ".json": _write_json,
    ".jsonl": _write_jsonl,
}

SUPPORTED_FORMATS = tuple(_READERS)


def _pick(handlers: dict, path: Path):
    """Look up the handler for ``path``'s extension."""
    suffix = path.suffix.lower()
    if suffix not in handlers:
        raise ValueError(f"Unsupported file type '{suffix}'. Supported: {', '.join(handlers)}")
    return handlers[suffix]


def read_rows(path: Union[str, Path]) -> list:
    """Read a data file into a list of dicts, one per record.

    Args:
        path: File to read. The extension picks the format.

    Returns:
        List of dicts keyed by column name (or JSON key), in file order.

    Raises:
        ValueError: Unsupported extension, or JSON that isn't a list of objects.
    """
    path = Path(path)
    return _pick(_READERS, path)(path)


def write_rows(rows: list, path: Union[str, Path]) -> None:
    """Write a list of dicts to a data file.

    Args:
        rows: One dict per record.
        path: File to write. The extension picks the format; the folder is
            created if needed.

    Raises:
        ValueError: Unsupported extension.
    """
    path = Path(path)
    writer = _pick(_WRITERS, path)
    path.parent.mkdir(parents=True, exist_ok=True)
    writer(rows, path)
