"""
Invariant tests on every record currently persisted under project/output.

These guard the contract every downstream consumer relies on:
  - schema validates
  - trust score reconstructs exactly from sub-scores × confidence × weights
  - weight tables sum to 1.0
  - no record below hard_reject_trust_floor
  - trust tier matches the score band

If any of these fail, the pipeline output is corrupt or the formula has drifted.
"""
from __future__ import annotations

from project.scoring.config import CFG, WEIGHTS
from project.utils.schema import validate_record, SchemaError


def _expected_tier(score: float) -> str:
    if score >= CFG.trust_tier_authoritative:
        return "AUTHORITATIVE"
    if score >= CFG.trust_tier_credible:
        return "CREDIBLE"
    if score >= CFG.trust_tier_uncertain:
        return "UNCERTAIN"
    return "UNRELIABLE"


def test_weight_tables_sum_to_one():
    """TrustWeights.__post_init__ should already enforce this, but spot-check."""
    for source_type, w in WEIGHTS.items():
        total = sum(w.as_dict().values())
        assert abs(total - 1.0) < 1e-6, f"{source_type} weights sum to {total}, not 1.0"


def test_all_three_source_types_have_weights():
    for st in ("blog", "youtube", "pubmed"):
        assert st in WEIGHTS, f"missing weight table for {st}"


def test_stored_records_schema_validates(stored_records):
    if not stored_records:
        return  # no fixtures present; skip
    failures = []
    for r in stored_records:
        try:
            validate_record(r)
        except SchemaError as e:
            failures.append((r.get("source_url", "?"), str(e)))
    assert not failures, f"schema violations: {failures[:3]}"


def test_stored_trust_scores_in_unit_interval(stored_records):
    for r in stored_records:
        ts = r["trust_score"]
        assert 0.0 <= ts <= 1.0, f"trust_score {ts} out of [0,1] for {r['source_url']}"


def test_stored_no_hard_reject_leak(stored_records):
    leaks = [r for r in stored_records if r["trust_score"] < CFG.hard_reject_trust_floor]
    assert not leaks, f"{len(leaks)} records below hard_reject_trust_floor"


def test_stored_tier_matches_score(stored_records):
    mismatches = [r for r in stored_records if r["trust_tier"] != _expected_tier(r["trust_score"])]
    assert not mismatches, f"{len(mismatches)} tier/score mismatches"


def test_stored_trust_formula_reconstructs(stored_records):
    """
    Same formula trust_score.calculate() uses. Each stored record's
    trust_score must reconstruct from its own sub_scores/confidence/weights.
    A failure here means the formula was changed without updating the
    stored data — exactly the kind of regression this test exists to catch.
    """
    bad = []
    for r in stored_records:
        num = sum(r["weights"][k] * r["sub_scores"][k] * r["confidence"][k]
                  for k in r["sub_scores"])
        den = sum(r["weights"][k] * r["confidence"][k] for k in r["sub_scores"])
        raw = num / den if den > 1e-9 else 0.5
        penalised = raw * r["abuse_penalty"]
        if r.get("prior_used") is not None:
            final = (1 - CFG.prior_weight) * penalised + CFG.prior_weight * r["prior_used"]
        else:
            final = penalised
        final = round(min(max(final, 0.0), 1.0), 3)
        if abs(final - r["trust_score"]) > 0.005:
            bad.append((r["source_url"], r["trust_score"], final))
    assert not bad, f"{len(bad)} records failed formula reconstruction: {bad[:3]}"


def test_stored_sub_scores_have_all_nine_signals(stored_records):
    expected = {
        "author_credibility", "citation_count", "domain_authority", "recency",
        "medical_disclaimer_presence", "metadata_completeness", "content_depth",
        "engagement_authenticity", "adversarial_risk",
    }
    for r in stored_records:
        keys = set(r.get("sub_scores", {}).keys())
        assert keys == expected, f"missing signals: {expected - keys} extra: {keys - expected}"


def test_abuse_penalty_in_unit_interval(stored_records):
    for r in stored_records:
        ap = r["abuse_penalty"]
        assert CFG.abuse_floor <= ap <= 1.0, f"abuse_penalty {ap} out of [floor, 1.0]"


def test_content_chunks_non_empty(stored_records):
    """validate_record already enforces this, but pin it explicitly."""
    for r in stored_records:
        assert r["content_chunks"], f"empty content_chunks for {r['source_url']}"
