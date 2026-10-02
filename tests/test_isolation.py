"""The three tools may only share code through src/settings.

simplification/, dashboard/ and feedback_form/ must never import each other
— whatever one needs to tell another goes through the settings files. This
reads every module's imports and fails on a crossing.

Run from the repo root:
    python -m pytest tests/test_isolation.py
"""

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
TOOLS = ("simplification", "dashboard", "feedback_form")


def _imported_src_packages(path: Path, own_package: str) -> set:
    """Names of the src/ sub-packages a module imports, absolute or relative."""
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if isinstance(node, ast.Import):
            names = [alias.name for alias in node.names]
        elif isinstance(node, ast.ImportFrom):
            if node.level == 1:      # from . / from .x  -> own package
                names = [f"src.{own_package}"]
            elif node.level == 2:    # from ..x          -> sibling package x
                names = [f"src.{node.module or ''}"]
            else:
                names = [node.module or ""]
        else:
            continue
        for name in names:
            parts = name.split(".")
            if parts[0] == "src" and len(parts) > 1:
                found.add(parts[1])
    return found


@pytest.mark.parametrize("tool", TOOLS)
def test_tool_only_imports_itself_and_settings(tool):
    for path in (SRC / tool).rglob("*.py"):
        crossing = _imported_src_packages(path, tool) - {tool, "settings"}
        assert not crossing, f"{path.relative_to(SRC)} imports {sorted(crossing)}"


def test_settings_imports_none_of_the_tools():
    for path in (SRC / "settings").rglob("*.py"):
        assert _imported_src_packages(path, "settings") <= {"settings"}, path.name
