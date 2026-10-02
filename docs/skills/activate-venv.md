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
pip install -r code/requirements.txt
```

## Commands

**Activate the venv (required before any Python command):**
```bash
source venv/bin/activate
```

**Run tests:**
```bash
source venv/bin/activate && cd code && python -m pytest tests/ -v
```

**Run a script:**
```bash
source venv/bin/activate && cd code && python src/some_script.py
```

## Notes
- A venv breaks if its folder (or the repo) is moved: `activate`, `pip` and
  `pytest` keep pointing at the old path. Delete `venv/` and recreate it.
- Tests must be run from `code/` as the working directory so that `src.*` imports
  resolve correctly.
