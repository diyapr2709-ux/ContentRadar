""""
task1/scraper/pubmed_scraper.py

Topic-driven PubMed scraper. Scrapes exactly 1 article per run.
Uses NCBI eutils 
"""

from __future__ import annotations

import json
import os
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import requests
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task1.utils.sanitizer      import clean_text
from task1.utils.tagging        import auto_tag
from task1.utils.chunking       import chunk_text
from task1.scraper.blog_scraper import expand_topic

load_dotenv(ROOT / ".env")
NCBI_API_KEY = os.getenv("NCBI_API_KEY", "")

TARGET_COUNT   = 1
MAX_CANDIDATES = 5
MIN_RELEVANCE  = 0.003
MAX_BYTES      = 5 * 1024 * 1024
RATE_LIMIT_SEC = 0.11 if NCBI_API_KEY else 0.34

EUTILS_BASE  = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
ESEARCH_URL  = f"{EUTILS_BASE}/esearch.fcgi"
ESUMMARY_URL = f"{EUTILS_BASE}/esummary.fcgi"
EFETCH_URL   = f"{EUTILS_BASE}/efetch.fcgi"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; AcademicBot/1.0)",
    "Accept":     "application/json, text/xml",
}

_HIGH_IMPACT_JOURNALS = frozenset((
    "nature medicine", "the lancet", "new england journal of medicine",
    "jama", "bmj", "nature", "science", "cell",
    "npj digital medicine", "diabetes care", "diabetologia",
    "plos medicine", "annals of internal medicine", "circulation",
    "bioinformatics", "nature communications",
))


def _build_query(topic: str) -> str:
    terms = expand_topic(topic)
    or_terms = " OR ".join('"' + t + '"' for t in terms)
    return f"({or_terms}) AND (machine learning OR deep learning OR AI OR neural network)"


def _is_relevant(text: str, topic: str) -> bool:
    if not text:
        return False
    terms = expand_topic(topic)
    low = text.lower()
    words = low.split()
    if not words:
        return False
    hits = sum(low.count(t.lower()) for t in terms)
    return (hits / len(words)) >= MIN_RELEVANCE


def _detect_lang(text: str) -> str:
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "en"


def _get(url: str, params: dict, retries: int = 3) -> requests.Response | None:
    if NCBI_API_KEY:
        params["api_key"] = NCBI_API_KEY
    for attempt in range(retries):
        try:
            resp = requests.get(url, params=params, headers=HEADERS, timeout=15)
            resp.raise_for_status()
            if len(resp.content) > MAX_BYTES:
                return None
            time.sleep(RATE_LIMIT_SEC)
            return resp
        except requests.HTTPError as e:
            if e.response.status_code in (400, 404):
                return None
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
        except requests.RequestException:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)
    return None


def _esearch(topic: str) -> list[str]:
    resp = _get(ESEARCH_URL, {
        "db": "pubmed", "term": _build_query(topic),
        "retmax": MAX_CANDIDATES, "retmode": "json",
        "sort": "relevance",
    })
    if resp is None:
        return []
    try:
        return resp.json().get("esearchresult", {}).get("idlist", [])
    except Exception:
        return []


def _esummary_batch(pmids: list[str]) -> dict[str, dict]:
    if not pmids:
        return {}
    resp = _get(ESUMMARY_URL, {
        "db": "pubmed", "id": ",".join(pmids), "retmode": "json",
    })
    if resp is None:
        return {}
    try:
        result = resp.json().get("result", {})
        return {pid: result[pid] for pid in pmids if pid in result}
    except Exception:
        return {}


def _efetch_abstract(pmid: str) -> str:
    resp = _get(EFETCH_URL, {
        "db": "pubmed", "id": pmid, "retmode": "xml",
    })
    if resp is None:
        return ""
    try:
        root = ET.fromstring(resp.content)
        parts = []
        for el in root.findall(".//AbstractText"):
            label = el.get("Label", "")
            text = clean_text(el.text or "")
            parts.append(f"{label}: {text}" if label else text)
        return "\n\n".join(parts)
    except Exception:
        return ""


def _rank_score(meta: dict) -> float:
    score = 0.0
    journal = (meta.get("fulljournalname") or meta.get("source", "")).lower()
    if any(j in journal for j in _HIGH_IMPACT_JOURNALS):
        score += 2.0
    pubdate = meta.get("pubdate", "")
    if pubdate:
        try:
            year = int(pubdate[:4])
            score += max(0, (year - 2015) * 0.3)
        except ValueError:
            pass
    return score


def run(topic: str, output_dir: str) -> list[dict]:
    print(f"  [PubMed] topic='{topic}'")
    pmids = _esearch(topic)
    print(f"        PMIDs={pmids}")

    summaries = _esummary_batch(pmids)
    ranked = sorted(summaries.items(),
                    key=lambda kv: _rank_score(kv[1]), reverse=True)

    results: list[dict] = []
    for pmid, meta in ranked:
        if len(results) >= TARGET_COUNT:
            break

        pubmed_url = f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/"
        abstract   = _efetch_abstract(pmid)
        title      = clean_text(meta.get("title", "Unknown"))
        date       = clean_text(meta.get("pubdate", "Unknown"))[:10]

        authors_raw = meta.get("authors", [])
        authors = clean_text(", ".join(
            a.get("name", "") for a in authors_raw if a.get("name")
        )) if authors_raw else "Unknown"

        full_text = f"{title}\n\n{abstract}"
        if not _is_relevant(full_text, topic):
            continue

        results.append({
            "source_url":     pubmed_url,
            "source_type":    "pubmed",
            "author":         authors,
            "published_date": date,
            "language":       _detect_lang(full_text),
            "region":         "Unknown",
            "topic_tags":     auto_tag(full_text),
            "trust_score":    "",
            "content_chunks": chunk_text(abstract),
        })
        print(f"        ✓ PMID {pmid}")

    while len(results) < TARGET_COUNT:
        results.append({
            "source_url":     "https://pubmed.ncbi.nlm.nih.gov/",
            "source_type":    "pubmed",
            "author":         "Unknown",
            "published_date": "Unknown",
            "language":       "unknown",
            "region":         "Unknown",
            "topic_tags":     [],
            "trust_score":    "",
            "content_chunks": [f"[scrape_error] No relevant PubMed results for '{topic}'"],
        })

    out = Path(output_dir) / "pubmed.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"        → {out}")
    return results


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "diabetes",
        "task1/output/scraped_data")