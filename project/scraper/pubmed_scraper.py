"""
project/scraper/pubmed_scraper.py - Topic-driven PubMed scraper.

NCBI eUtils pipeline:
  esearch → get PMIDs ranked by relevance
  esummary (batched) → journal name, pub date, author list
  efetch (XML) → full structured abstract (labelled sections)

Ranking before fetch:
  Candidates are pre-ranked by a composite journal impact score before the
  efetch round-trip. This means we always fetch the best candidate first and
  only fall back to lower-ranked papers if the top candidate fails relevance
  or yields an empty abstract. This minimises API quota consumption.

Journal impact scoring:
  Binary whitelist (high-impact journals known a priori) + recency bonus.
  This is a heuristic - we don't have live impact factor data. The whitelist
  covers the top ~20 journals by citation count in health/AI domains.

Rate limiting:
  NCBI allows 3 req/s without an API key and 10 req/s with one.
  We use a conservative delay to stay well inside limits and avoid
  triggering NCBI's automated ban (they do IP-ban abusive clients).
"""
from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from project.utils.http_client    import fetch
from project.utils.sanitizer      import clean_text, detect_language
from project.utils.tagging        import auto_tag
from project.utils.chunking       import rag_chunk
from project.utils.topic_expansion import expand_topic
from project.scraper.blog_scraper  import expand_terms, _is_relevant
from project.scoring.config       import CFG

NCBI_API_KEY    = os.getenv("NCBI_API_KEY", "")
NCBI_CONTACT    = os.getenv("NCBI_CONTACT_EMAIL", "") or CFG.ncbi_contact_fallback
TARGET_COUNT    = CFG.pubmed_target_count
MAX_CANDIDATES  = CFG.pubmed_max_candidates
RATE_LIMIT_SEC  = CFG.delay_ncbi_with_key if NCBI_API_KEY else CFG.delay_ncbi_without_key

_NCBI_HEADERS = {
    "User-Agent": f"ContentRadar/1.0 (academic; contact: {NCBI_CONTACT})",
    "Accept":     "application/json, text/xml",
}

_HIGH_IMPACT_JOURNALS = frozenset(CFG.pubmed_high_impact_journals)


def _ncbi_get(url: str, params: dict) -> dict | ET.Element | None:
    """Fetch NCBI endpoint with API key injection and retry."""
    if NCBI_API_KEY:
        params = {**params, "api_key": NCBI_API_KEY}
    resp = fetch(url, params=params, headers=_NCBI_HEADERS)
    if resp is None:
        return None
    if "json" in resp.headers.get("Content-Type", ""):
        try:
            return resp.json()
        except Exception:
            return None
    try:
        return ET.fromstring(resp.content)
    except ET.ParseError:
        return None


def _esearch(topic: str) -> list[str]:
    """
    Boolean OR query across expanded terms with abstract presence filter.

    hasabstract[Filter] eliminates editorials and case reports with no abstract
    before we waste efetch quota on them. This is the primary precision lever -
    applying it at search time is far cheaper than fetching and discarding.

    AI/ML filter is only applied for AI/tech topics. Adding it to health
    topics (diabetes, cancer, etc.) biases results toward AI-health
    intersection papers and eliminates the majority of clinical literature.
    """
    expansion = expand_topic(topic)
    terms     = expand_terms(topic)
    or_block  = " OR ".join(f'"{t}"' for t in terms)
    if expansion.is_technical and expansion.pubmed_filter:
        query = f"({or_block}) AND ({expansion.pubmed_filter}) AND hasabstract[Filter]"
    else:
        query = f"({or_block}) AND hasabstract[Filter]"
    result   = _ncbi_get(CFG.ncbi_esearch_url, {
        "db": "pubmed", "term": query,
        "retmax": MAX_CANDIDATES, "retmode": "json", "sort": "relevance",
    })
    if result is None or not isinstance(result, dict):
        return []
    raw_ids = result.get("esearchresult", {}).get("idlist", [])
    # NCBI should return unique IDs but deduplicate defensively - duplicate
    # PMIDs would produce duplicate records that inflate downstream rankings.
    seen: set[str] = set()
    return [pid for pid in raw_ids if pid and pid not in seen and not seen.add(pid)]


def _esummary(pmids: list[str]) -> dict[str, dict]:
    """Single batch call for all PMID summaries - respects NCBI quota."""
    if not pmids:
        return {}
    result = _ncbi_get(CFG.ncbi_esummary_url, {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "json",
    })
    if result is None or not isinstance(result, dict):
        return {}
    raw = result.get("result", {})
    # NCBI includes a "uids" key listing result IDs - skip it when building output
    return {pmid: raw[pmid] for pmid in pmids if pmid in raw and isinstance(raw[pmid], dict)}


def _extract_doi(meta: dict) -> str:
    """Extract DOI from NCBI articleids list. Returns empty string if absent."""
    for aid in meta.get("articleids", []):
        if aid.get("idtype") == "doi":
            return clean_text(aid.get("value", ""))
    return ""


def _efetch_abstract(pmid: str) -> str:
    """
    Fetch structured abstract XML. Preserves section labels (BACKGROUND,
    METHODS, RESULTS, CONCLUSIONS) - these structure PubMed abstracts and
    improve chunking quality significantly vs treating the abstract as a blob.
    """
    result = _ncbi_get(CFG.ncbi_efetch_url, {
        "db": "pubmed", "id": pmid, "retmode": "xml",
    })
    if result is None or not isinstance(result, ET.Element):
        return ""

    parts = []
    for el in result.findall(".//AbstractText"):
        label = el.get("Label", "")
        text  = clean_text(el.text or "")
        if text:
            parts.append(f"{label}: {text}" if label else text)
    return "\n\n".join(parts)


def _journal_impact_score(meta: dict) -> float:
    """
    Pre-rank score for candidate selection. Higher = fetch first.
    Not the trust score - that is computed by trust_score.py after ingestion.
    """
    score   = 0.0
    journal = (meta.get("fulljournalname") or meta.get("source", "")).lower()
    if any(j in journal for j in _HIGH_IMPACT_JOURNALS):
        score += CFG.pubmed_journal_impact_base
    try:
        year   = int(str(meta.get("pubdate", ""))[:4])
        score += max(0.0, (year - CFG.pubmed_recency_base_year) * CFG.pubmed_recency_per_year)
    except (ValueError, TypeError):
        pass
    return score


def run(topic: str, output_dir: str) -> list[dict]:
    print(f"  [PubMed] topic='{topic}'")
    pmids = _esearch(topic)
    print(f"        PMIDs found: {pmids}")

    summaries = _esummary(pmids)
    ranked    = sorted(summaries.items(),
                       key=lambda kv: _journal_impact_score(kv[1]), reverse=True)

    results:   list[dict] = []
    seen_dois: set[str]   = set()

    for pmid, meta in ranked:
        if len(results) >= TARGET_COUNT:
            break

        doi_candidate = _extract_doi(meta)
        if doi_candidate and doi_candidate in seen_dois:
            print(f"        ✗ PMID {pmid}  duplicate DOI {doi_candidate[:40]} - skipping")
            continue

        # Rate-limit each efetch call - NCBI enforces 3 req/s without API key,
        # 10 req/s with. fetch() only sleeps 0.28s; we add the remaining gap.
        time.sleep(RATE_LIMIT_SEC)
        abstract = _efetch_abstract(pmid)
        title    = clean_text(meta.get("title", "Unknown"))

        # Skip editorials/comments with no abstract - title-only records have
        # insufficient content for meaningful trust scoring and inflate scores
        # based on journal prestige alone rather than actual content evidence.
        if not abstract or len(abstract.split()) < CFG.pubmed_min_abstract_words:
            print(f"        ✗ PMID {pmid}  thin/no abstract ({len(abstract.split()) if abstract else 0} words) - skipping")
            continue

        # Preserve full NCBI date (e.g. "2024 Jan 15") - trust_score parses it
        date     = clean_text(meta.get("pubdate", "Unknown"))
        journal  = clean_text(meta.get("fulljournalname") or meta.get("source", ""))

        authors_raw = meta.get("authors", [])
        if not isinstance(authors_raw, list):
            authors_raw = []
        authors = clean_text(", ".join(
            a.get("name", "") for a in authors_raw
            if isinstance(a, dict) and a.get("name")
        )) if authors_raw else "Unknown"

        full_text = f"{title}\n\n{abstract}"
        if not _is_relevant(full_text, topic):
            # Was previously a silent `continue`. With multi-word-only expansions
            # (e.g. 'sleep science' → all phrases) this loop could reject all 20
            # candidates without a single log line, leaving an empty pubmed.json
            # with no diagnostic. Now we surface the rejection.
            print(f"        ✗ PMID {pmid}  not relevant to topic '{topic}' - skipping")
            continue

        # Citation proxy: NCBI eSummary doesn't expose citation counts. Use the
        # journal-tier whitelist (already a frozenset for O(1) lookup) as proxy.
        # Recency is intentionally excluded - new papers haven't accumulated cites,
        # so recency bias would inflate the proxy. Recency only ranks candidates.
        journal_lc     = journal.lower()
        is_high_impact = any(j in journal_lc for j in _HIGH_IMPACT_JOURNALS)
        citation_base  = CFG.pubmed_journal_impact_base if is_high_impact else 0.0
        citation_proxy = max(CFG.pubmed_citation_proxy_floor,
                             int(citation_base * CFG.pubmed_citation_proxy_mult))

        doi = doi_candidate
        if doi:
            seen_dois.add(doi)

        pubmed_url = f"{CFG.pubmed_base_url}{pmid}/"
        chunks     = rag_chunk(abstract)
        if not chunks:
            chunks = [{"text": abstract, "chunk_index": 0, "total_chunks": 1,
                       "quality_score": CFG.chunk_fallback_quality_pubmed, "has_overlap": False}]
        results.append({
            "source_url":     pubmed_url,
            "source_type":    "pubmed",
            "author":         authors,
            "published_date": date,
            "language":       detect_language(full_text),
            "region":         "Unknown",
            "journal":        journal,
            "pmid":           pmid,
            "doi":            doi,
            "topic_tags":     auto_tag(full_text),
            "citations":      citation_proxy,
            "content_chunks": chunks,
        })
        print(f"        ✓ PMID {pmid}  "
              f"journal='{journal[:CFG.display_pubmed_journal_chars]}'  "
              f"doi={doi[:CFG.display_pubmed_doi_chars] or 'n/a'}  "
              f"citation_proxy={citation_proxy}")


    out = Path(output_dir) / "pubmed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"        → {out}")
    return results
