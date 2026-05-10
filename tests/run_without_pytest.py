"""
Fallback runner for environments where pytest isn't installed.

Discovers test_*.py files in this directory, imports each module, and runs
every callable named `test_*`. Doesn't support fixtures or parametrization
— for that, install pytest and run `pytest tests/`. This script exists so
the regression suite is at least executable in a fresh environment without
extra installs.

Usage:
    python tests/run_without_pytest.py
"""
from __future__ import annotations

import importlib
import inspect
import sys
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(ROOT))


def _load_fixture_data():
    """Pre-load stored_records / production_records so tests that take them
    as positional args can be called directly."""
    from tests.conftest import _load_records
    return {
        "stored_records":     _load_records(ROOT / "project" / "output" / "test_runs"),
        "production_records": _load_records(ROOT / "project" / "output" / "topics"),
    }


def main() -> int:
    fixtures = _load_fixture_data()
    passed = 0
    failed = 0
    skipped = 0
    failures: list[str] = []

    for path in sorted(HERE.glob("test_*.py")):
        mod_name = f"tests.{path.stem}"
        try:
            mod = importlib.import_module(mod_name)
        except Exception:
            failed += 1
            failures.append(f"  IMPORT {mod_name}\n{traceback.format_exc()}")
            continue

        for name, fn in inspect.getmembers(mod, inspect.isfunction):
            if not name.startswith("test_"):
                continue
            sig = inspect.signature(fn)
            # Skip tests that require pytest-only features (tmp_path, monkeypatch)
            if any(p.name in ("tmp_path", "monkeypatch") for p in sig.parameters.values()):
                skipped += 1
                continue
            kwargs = {}
            for p in sig.parameters.values():
                if p.name in fixtures:
                    kwargs[p.name] = fixtures[p.name]
            try:
                fn(**kwargs)
                passed += 1
            except Exception:
                failed += 1
                failures.append(f"  FAIL {mod_name}::{name}\n{traceback.format_exc()}")

    print(f"\n{passed} passed, {failed} failed, {skipped} skipped (pytest-only fixtures)")
    if failures:
        print("\n--- failures ---")
        for f in failures:
            print(f)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
