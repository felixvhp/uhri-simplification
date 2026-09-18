# Skill: Activate Python Virtual Environment

## When to use this
Before running any Python code, tests, or scripts in this project. The system
Python (`python` / `python3`) does not have the project's packages installed.
Always activate the project venv first.

## Setup
Virtual environments are stored under `~/Documents/Venvs/`. This project uses
the `un_project` environment.

## Commands

**Activate the venv (required before any Python command):**
```bash
source ~/Documents/Venvs/un_project/bin/activate
```

**Run tests:**
```bash
source ~/Documents/Venvs/un_project/bin/activate && python -m pytest code/tests/ -v
```

**Run a script:**
```bash
source ~/Documents/Venvs/un_project/bin/activate && cd code && python src/some_script.py
```

## Notes
- The `venv` shell alias defined in `~/.zshrc` (`venv un_project`) only works in
  interactive zsh sessions. In non-interactive shells or tool-driven environments,
  use the `source` command above.
- The active Python after sourcing is `~/Documents/Venvs/un_project/bin/python`
  (Python 3.9.6).
- Tests must be run from `code/` as the working directory so that `src.*` imports
  resolve correctly.
