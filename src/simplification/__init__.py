"""
Pkg 3: Simplification — from input data to simplified output

Usage:
    from src.simplification import simplify, simplify_file

See core.py for the interface and `python -m src.simplification --help` for
the command line.
"""

from .core import (
    MultiplePromptsWarning,
    simplify,
    simplify_file,
    simplify_rows,
    simplify_text,
)
from .io import SUPPORTED_FORMATS, read_rows, write_rows

__all__ = [
    "MultiplePromptsWarning",
    "SUPPORTED_FORMATS",
    "read_rows",
    "simplify",
    "simplify_file",
    "simplify_rows",
    "simplify_text",
    "write_rows",
]
