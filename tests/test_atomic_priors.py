"""
Atomic-write contract for `.priors_cache.json`. A crash mid-write or a second
concurrent CLI run must not leave the file truncated/corrupt.
"""
from __future__ import annotations

import json
from pathlib import Path

import project.scoring.trust_score as ts


def test_priors_atomic_write_round_trip(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_PRIORS_FILE", tmp_path / "p.json")
    monkeypatch.setattr(ts, "_priors_data", {"example.com": {"n": 3, "mean_score": 0.7}})
    monkeypatch.setattr(ts, "_priors_loaded", True)

    ts.persist_priors()
    on_disk = json.loads((tmp_path / "p.json").read_text())
    assert on_disk == {"example.com": {"n": 3, "mean_score": 0.7}}


def test_priors_temp_file_cleaned_up(tmp_path, monkeypatch):
    """No `.tmp` file should be left over after successful persist."""
    monkeypatch.setattr(ts, "_PRIORS_FILE", tmp_path / "p.json")
    monkeypatch.setattr(ts, "_priors_data", {"a.com": {"n": 1, "mean_score": 0.5}})
    monkeypatch.setattr(ts, "_priors_loaded", True)

    ts.persist_priors()
    tmps = list(tmp_path.glob("*.tmp"))
    assert not tmps, f"leftover tmp files: {tmps}"


def test_update_prior_then_get_after_threshold(tmp_path, monkeypatch):
    monkeypatch.setattr(ts, "_PRIORS_FILE", tmp_path / "p.json")
    monkeypatch.setattr(ts, "_priors_data", {})
    monkeypatch.setattr(ts, "_priors_loaded", True)

    # Add enough records to exceed prior_min_n (default 3)
    for s in (0.5, 0.6, 0.7, 0.8):
        ts.update_prior("https://example.com/article", s)

    prior = ts._get_prior("https://example.com/article")
    assert prior is not None
    assert 0.4 <= prior <= 0.9, f"prior {prior} out of expected band"
