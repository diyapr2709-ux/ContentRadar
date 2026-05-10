"""
Pytest configuration + shared fixtures.

Adds the repo root to sys.path so tests can `from project...` like the
pipeline does. Loads the most recent test_runs snapshot once per session
for invariant tests that iterate over stored records.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def _load_records(root: Path) -> list[dict]:
    out: list[dict] = []
    if not root.exists():
        return out
    for topic_dir in root.iterdir():
        if not topic_dir.is_dir() or topic_dir.name.startswith("_"):
            continue
        for fname in ("blogs.json", "youtube.json", "pubmed.json"):
            p = topic_dir / fname
            if p.exists():
                try:
                    out.extend(json.loads(p.read_text()))
                except Exception:
                    continue
    return out


try:
    import pytest

    @pytest.fixture(scope="session")
    def stored_records() -> list[dict]:
        """Every record currently in project/output/test_runs/<topic>/{blogs,youtube,pubmed}.json"""
        return _load_records(ROOT / "project" / "output" / "test_runs")

    @pytest.fixture(scope="session")
    def production_records() -> list[dict]:
        """Records under project/output/topics/ if available."""
        return _load_records(ROOT / "project" / "output" / "topics")
except ImportError:
    # Allow bare-python execution. The fixtures become module-level callables.
    def stored_records() -> list[dict]:
        return _load_records(ROOT / "project" / "output" / "test_runs")
    def production_records() -> list[dict]:
        return _load_records(ROOT / "project" / "output" / "topics")
