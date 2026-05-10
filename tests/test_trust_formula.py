"""
Synthetic trust-formula tests. Pin invariants that are too easy to break
when refactoring trust_score.calculate().
"""
from __future__ import annotations

from project.scoring.config import CFG, WEIGHTS
from project.scoring.trust_score import calculate


def _base_record():
    return {
        "source_url":     "https://medium.com/@x/example",
        "source_type":    "blog",
        "author":         "Jane Doe",
        "published_date": "2025-01-01",
        "content":        " ".join(["lorem ipsum"] * 500),
        "citations":      None,
        "views":          None,
        "likes":          None,
        "anomaly_flags":  [],
        "record_dict":    {"author": "Jane Doe", "published_date": "2025-01-01",
                           "source_url": "https://medium.com/@x/example"},
    }


def test_calculate_returns_required_keys():
    r = calculate(**_base_record())
    for k in ("trust_score", "sub_scores", "confidence", "evidence",
              "weights", "abuse_penalty", "abuse_reasons", "prior_used"):
        assert k in r, f"missing key {k}"


def test_trust_score_clamped_to_unit_interval():
    r = calculate(**_base_record())
    assert 0.0 <= r["trust_score"] <= 1.0


def test_all_nine_signals_present():
    r = calculate(**_base_record())
    expected = {
        "author_credibility", "citation_count", "domain_authority", "recency",
        "medical_disclaimer_presence", "metadata_completeness", "content_depth",
        "engagement_authenticity", "adversarial_risk",
    }
    assert set(r["sub_scores"].keys()) == expected
    assert set(r["confidence"].keys()) == expected
    assert set(r["weights"].keys())    == expected


def test_unknown_source_type_falls_back_to_blog():
    """Defensive: calculate() shouldn't crash on a typo'd source_type."""
    args = _base_record()
    args["source_type"] = "bogus_type"
    r = calculate(**args)
    assert r["weights"] == WEIGHTS["blog"].as_dict()


def test_abuse_penalty_floored_at_cfg_floor():
    """Stacking 1000 anomaly flags shouldn't drop penalty below CFG.abuse_floor."""
    args = _base_record()
    args["anomaly_flags"] = ["low_entropy"] * 1000
    r = calculate(**args)
    assert r["abuse_penalty"] >= CFG.abuse_floor


def test_pubmed_engagement_signal_has_zero_confidence():
    """For pubmed, engagement_authenticity must be excluded from the weighted
    fusion via confidence=0 (so it does not contribute to numerator or denom)."""
    args = _base_record()
    args["source_type"] = "pubmed"
    args["citations"] = 50
    r = calculate(**args)
    assert r["confidence"]["engagement_authenticity"] == 0.0


def test_no_double_count_keyword_stuffing_in_anomaly_loop():
    """Keyword stuffing fires via direct check in _abuse_penalty AND can also
    be present as an anomaly_flag. The anomaly loop must skip the flag to
    avoid applying the penalty twice. Empirically: penalty for stuffing-only
    record matches CFG.penalty_keyword_stuffing exactly, not its square."""
    args = _base_record()
    args["content"] = " ".join(["spam"] * 200)  # 100% top-word density
    args["anomaly_flags"] = ["keyword_stuffing:spam:1.000"]
    r = calculate(**args)
    # Penalty should be approximately CFG.penalty_keyword_stuffing (0.80),
    # not 0.64 (0.80^2 if double-counted). Allow small slack for stacking
    # with other automatic checks.
    assert r["abuse_penalty"] > 0.70, \
        f"keyword_stuffing appears to be double-counted: penalty={r['abuse_penalty']}"
