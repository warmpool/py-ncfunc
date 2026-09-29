# AGENTS.md

## Project Overview

`ncfunc` is a Python library for reading/writing NetCDF climate data files. Built on `netCDF4` and `numpy`.

## Environment

- Python 3.14 (required by `pyproject.toml`)
- Package manager: `uv`
- External dependency: `datenum` from local path `/home/lkkbox/projects/py-datenum` (editable install)

## Commands

```bash
# Run tests
uv run pytest

# Run single test
uv run pytest tests/test_core.py::test_var_names

# Lint
uv run ruff check src/ tests/

# Format check
uv run ruff format --check src/ tests/
```

## Code Structure

```
src/ncfunc/
  __init__.py   # Public API exports
  core.py       # Core functionality: DatasetMeta, readers, writers
  read_mf.py    # Multi-file reader: read_mf_within
tests/
  test_core.py    # pytest tests
  test_read_mf.py # pytest tests for read_mf_within
  create_toync.py # Helper that generates dataa*.nc fixture files
  ersst_2022-2024.nc  # Test fixture NetCDF file
```

## Key Facts

- `core.py` contains both metadata classes (`DatasetMeta`, `VariableMeta`) and all public functions
- `DatasetMeta` caches file metadata in `cached_registry` class variable, invalidated by `mtime`
- Time decoding uses `datenum` library, handles special calendars (365_day, 360_day)
- `tests/nctools.py` provides reference implementations for test comparisons
- Legacy files `tests/nctools.py`, `tests/timetools.py`, `tests/multiNcRead.py` are gitignored

## Agent skills

### Issue tracker

Issues live as local markdown files under `.scratch/<feature>/`. See `docs/agents/issue-tracker.md`.

### Triage labels

Default labels: `needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context. See `docs/agents/domain.md`.
