# Skill: Activate Python Virtual Environment

## When to use this
Before running any Python code, tests, or scripts in this project. The system
Python (`python` / `python3`) does not have the project's packages installed.
Always activate the project venv first.

## Setup
The venv lives at `venv/` in the repo root (gitignored). Create it once per
checkout — it needs Python 3.11 or newer (the pinned pandas/numpy require it):

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Commands

**Activate the venv (required before any Python command):**
```bash
source venv/bin/activate
```

**Run tests:**
```bash
source venv/bin/activate && python -m pytest -v
```

**Run a script:**
```bash
source venv/bin/activate && python -m src.simplification --help
```

## Notes
- A venv breaks if its folder (or the repo) is moved: `activate`, `pip` and
  `pytest` keep pointing at the old path. Delete `venv/` and recreate it.
- Run everything from the repo root so that `src.*` imports resolve.
