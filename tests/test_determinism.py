"""
CONTENTRADAR_AS_OF env var must freeze the scoring clock so same input → same
output. Before this fix, recency drift produced ~10pp shifts on stored records.
"""
from __future__ import annotations

import os
from datetime import datetime, timezone

from project.scoring.config import scoring_now
from project.scoring.trust_score import _recency_signal


def test_default_scoring_now_uses_wall_clock():
    """When env var unset, scoring_now() returns real now within a few seconds."""
    os.environ.pop("CONTENTRADAR_AS_OF", None)
    now_real = datetime.now(timezone.utc).replace(tzinfo=None)
    now_fn   = scoring_now()
    delta = abs((now_real - now_fn).total_seconds())
    assert delta < 60, f"scoring_now drifted from wall clock by {delta}s"


def test_as_of_date_freezes_clock():
    os.environ["CONTENTRADAR_AS_OF"] = "2025-01-15"
    try:
        t = scoring_now()
        assert t.year == 2025 and t.month == 1 and t.day == 15
    finally:
        os.environ.pop("CONTENTRADAR_AS_OF", None)


def test_as_of_iso_z_format():
    os.environ["CONTENTRADAR_AS_OF"] = "2025-06-01T12:00:00Z"
    try:
        t = scoring_now()
        assert t.year == 2025 and t.month == 6 and t.day == 1 and t.hour == 12
    finally:
        os.environ.pop("CONTENTRADAR_AS_OF", None)


def test_recency_signal_reproducible_under_frozen_clock():
    """The headline use case: same record → same recency score across runs."""
    os.environ["CONTENTRADAR_AS_OF"] = "2026-01-01"
    try:
        sig_a = _recency_signal("2025-06-15", "blog")
        sig_b = _recency_signal("2025-06-15", "blog")
        assert sig_a.score == sig_b.score
        assert sig_a.confidence == sig_b.confidence
        assert sig_a.evidence == sig_b.evidence
    finally:
        os.environ.pop("CONTENTRADAR_AS_OF", None)


def test_recency_drifts_when_clock_moves():
    """Inverse check: changing as_of must change the recency score."""
    os.environ["CONTENTRADAR_AS_OF"] = "2025-01-01"
    sig_early = _recency_signal("2024-06-01", "blog")
    os.environ["CONTENTRADAR_AS_OF"] = "2027-01-01"
    sig_late  = _recency_signal("2024-06-01", "blog")
    os.environ.pop("CONTENTRADAR_AS_OF", None)
    assert sig_late.score < sig_early.score, \
        "older record should score lower recency when viewed from a later date"


def test_invalid_as_of_falls_through_to_wall_clock():
    os.environ["CONTENTRADAR_AS_OF"] = "not a real date"
    try:
        t = scoring_now()
        # falls back to real now → year must be reasonable
        assert t.year >= 2025
    finally:
        os.environ.pop("CONTENTRADAR_AS_OF", None)
