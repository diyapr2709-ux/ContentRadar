"""
project/utils/schema.py - Canonical output schema enforcement.

Every record written to disk passes validate_record() before serialisation.
This catches type mismatches, missing fields, and out-of-range values that
would silently corrupt downstream vector stores or ranking systems.

Schema version history:
  1.0 - initial (task1/task2)
  2.0 - provenance fields (record_id, schema_version, ingested_at)
         full scoring breakdown (sub_scores, confidence, evidence, weights)
         source-type-specific metadata (pmid/doi/journal, view_count/like_count)
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

from project.scoring.config import CFG

SCHEMA_VERSION = "2.0"

VALID_SOURCE_TYPES = frozenset({"blog", "youtube", "pubmed"})
VALID_TRUST_TIERS  = frozenset({"AUTHORITATIVE", "CREDIBLE", "UNCERTAIN", "UNRELIABLE"})

# Ordered tuple for iteration (blogs → youtube → pubmed) and canonical file mapping.
# Single definition imported by main.py, build_index.py, and run_topics.py.
SOURCE_TYPES: tuple[str, ...] = ("blog", "youtube", "pubmed")
TYPE_TO_FILE: dict[str, str]  = {"blog": "blogs", "youtube": "youtube", "pubmed": "pubmed"}

_REQUIRED: dict[str, type] = {
    "record_id":      str,
    "schema_version": str,
    "ingested_at":    str,
    "source_url":     str,
    "source_type":    str,
    "author":         str,
    "published_date": str,
    "language":       str,
    "region":         str,
    "topic_tags":     list,
    "trust_score":    float,
    "trust_tier":     str,
    "content_chunks": list,
    "sub_scores":     dict,
    "confidence":     dict,
    "evidence":       dict,
    "weights":        dict,
    "abuse_penalty":  float,
    "abuse_reasons":  list,
    "anomaly_flags":  list,
}


def make_record_id(source_url: str, source_type: str) -> str:
    """Stable content ID. Deterministic across pipeline runs."""
    key = f"{source_type}:{source_url}".encode()
    return hashlib.sha256(key).hexdigest()[:CFG.schema_id_hex_len]


def now_utc() -> str:
    """ISO 8601 UTC timestamp for ingestion provenance."""
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class SchemaError(ValueError):
    pass


def validate_record(record: dict[str, Any]) -> dict[str, Any]:
    """
    Validate and return the record. Raises SchemaError on any violation.

    Checks performed:
      - All required fields present with correct Python types
      - trust_score ∈ [0.0, 1.0]
      - abuse_penalty ∈ [0.0, 1.0]
      - source_type in VALID_SOURCE_TYPES
      - trust_tier in VALID_TRUST_TIERS
      - content_chunks is a non-empty list of dicts each with a 'text' key
      - source_url is a non-empty string
    """
    missing = [f for f in _REQUIRED if f not in record]
    if missing:
        raise SchemaError(f"Missing required fields: {missing}")

    for field, expected_type in _REQUIRED.items():
        val = record[field]
        if not isinstance(val, expected_type):
            raise SchemaError(
                f"Field '{field}': expected {expected_type.__name__}, "
                f"got {type(val).__name__} = {val!r}"
            )

    ts = record["trust_score"]
    if not (0.0 <= ts <= 1.0):
        raise SchemaError(f"trust_score {ts} out of [0.0, 1.0]")

    ap = record["abuse_penalty"]
    if not (0.0 <= ap <= 1.0):
        raise SchemaError(f"abuse_penalty {ap} out of [0.0, 1.0]")

    st = record["source_type"]
    if st not in VALID_SOURCE_TYPES:
        raise SchemaError(f"Invalid source_type '{st}'")

    tt = record["trust_tier"]
    if tt not in VALID_TRUST_TIERS:
        raise SchemaError(f"Invalid trust_tier '{tt}'")

    if not record["source_url"]:
        raise SchemaError("source_url is empty")

    chunks = record["content_chunks"]
    if not chunks:
        raise SchemaError("content_chunks is empty")
    for i, c in enumerate(chunks):
        if not isinstance(c, dict) or "text" not in c:
            raise SchemaError(f"content_chunks[{i}] missing 'text' key or not a dict")

    # prior_used is optional (None when fewer than prior_min_n domain records exist)
    if "prior_used" in record:
        pv = record["prior_used"]
        if pv is not None and not isinstance(pv, (float, int)):
            raise SchemaError(
                f"Field 'prior_used': expected float or None, got {type(pv).__name__}"
            )

    return record
