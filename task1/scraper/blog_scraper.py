"""
task1/scraper/blog_scraper.py
Topic-driven blog scraper. Scrapes exactly 3 blog posts per run.

Search strategy
Medium RSS tag feeds — public XML, no API key needed.
Content extracted directly from RSS feed (content:encoded field)
so no article fetch is needed — avoids all 403 blocks entirely.

Optimisations
1. Topic Expansion       — expanded slugs used for RSS tag search
2. Trust-Aware Selection — candidates pre-ranked by domain tier
3. Source Diversity      — unique per Medium author, not just domain
4. Relevance Filtering   — keyword density check on RSS content
5. No hardcoded URLs     — fully dynamic, topic-driven

Output: trust_score = "" — Task 2 fills it.
"""

from __future__ import annotations

import json
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from langdetect import detect, LangDetectException

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task1.utils.sanitizer import clean_text, validate_url
from task1.utils.tagging   import auto_tag
from task1.utils.chunking  import chunk_text

TARGET_COUNT  = 3
MIN_RELEVANCE = 0.001
MAX_BYTES     = 5 * 1024 * 1024

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                       "AppleWebKit/537.36 (KHTML, like Gecko) "
                       "Chrome/124.0.0.0 Safari/537.36",
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

NOISE_TAGS = ("nav", "header", "footer", "aside", "script",
              "style", "form", "button", "noscript", "iframe", "svg")


# Topic expansion 

EXPANSIONS_RAW: dict[str, list[str]] = {
    "diabetes":      ["diabetes", "type 2 diabetes", "insulin resistance",
                      "diabetic", "glucose"],
    "obesity":       ["obesity", "overweight", "BMI", "weight loss",
                      "metabolic syndrome"],
    "cancer":        ["cancer", "oncology", "tumor",
                      "carcinoma", "malignancy"],
    "covid":         ["covid", "coronavirus", "pandemic", "SARS-CoV-2"],
    "alzheimer":     ["alzheimer", "dementia", "cognitive decline"],
    "hypertension":  ["hypertension", "high blood pressure", "cardiovascular"],
    "gut health":    ["gut", "microbiome", "digestive", "intestinal",
                      "probiotic", "prebiotic"],
    "mental health": ["mental health", "anxiety", "depression",
                      "psychology", "wellbeing"],
    "heart disease": ["heart disease", "cardiovascular", "cardiac",
                      "coronary", "heart failure"],
}

EXPANSIONS_SLUG: dict[str, list[str]] = {
    "diabetes":      ["diabetes", "type-2-diabetes", "insulin",
                      "blood-sugar", "glucose"],
    "obesity":       ["obesity", "overweight", "weight-loss", "bmi"],
    "cancer":        ["cancer", "oncology", "tumor", "malignancy"],
    "covid":         ["covid", "coronavirus", "pandemic"],
    "alzheimer":     ["alzheimer", "dementia", "cognitive-decline"],
    "hypertension":  ["hypertension", "blood-pressure", "cardiovascular"],
    "gut health":    ["gut-health", "microbiome", "digestive-health",
                      "gut", "probiotic"],
    "mental health": ["mental-health", "anxiety", "depression",
                      "psychology", "wellbeing"],
    "heart disease": ["heart-disease", "cardiovascular", "cardiac",
                      "heart-failure"],
}


def expand_topic(topic: str) -> list[str]:
    """Return plain-text expanded terms for relevance checking."""
    key = topic.lower().strip()
    for k, terms in EXPANSIONS_RAW.items():
        if k in key or key in k:
            return terms
    return [topic]


def _expand_slug(topic: str) -> list[str]:
    """Return hyphenated slugs for Medium RSS tag URLs."""
    key = topic.lower().strip()
    for k, terms in EXPANSIONS_SLUG.items():
        if k in key or key in k:
            return terms
    return [topic.lower().replace(" ", "-")]


# ── URL cleaning ───────────────────────────────────────────────────────────────

def _clean_url(url: str) -> str:
    """Strip RSS tracking query params — keeps scheme + netloc + path only."""
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return url


# ── Domain pre-scoring ─────────────────────────────────────────────────────────

_HIGH_DOMAINS = frozenset((
    "medium.com", "towardsdatascience.com", "huggingface.co",
    "openai.com", "anthropic.com", "healthline.com",
    "mayoclinic.org", "webmd.com", "hopkinsmedicine.org",
    "cdc.gov", "nih.gov", "who.int", "cancer.gov",
))
_MED_DOMAINS = frozenset((
    "analyticsvidhya.com", "kdnuggets.com",
    "machinelearningmastery.com", "hackernoon.com",
))


def _domain_tier(url: str) -> int:
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return 0
    if any(domain.endswith(d) for d in _HIGH_DOMAINS):
        return 2
    if any(domain.endswith(d) for d in _MED_DOMAINS):
        return 1
    return 0


def _diversity_key(url: str) -> str:
    """
    Compute a diversity key for deduplication.
    For Medium URLs, use netloc + first path segment (author handle)
    so different authors on medium.com count as distinct sources.
    For all other domains, use netloc only.
    """
    try:
        parsed = urlparse(url)
        netloc = parsed.netloc.lower().replace("www.", "")
        if "medium.com" in netloc:
            path_parts = [p for p in parsed.path.split("/") if p]
            return f"{netloc}/{path_parts[0]}" if path_parts else netloc
        return netloc
    except Exception:
        return url


# ── Relevance check 

def _is_relevant(text: str, topic: str) -> bool:
    if not text:
        return False
    terms = expand_topic(topic)
    low   = text.lower()
    words = low.split()
    if not words:
        return False
    hits = sum(low.count(t.lower()) for t in terms)
    return (hits / len(words)) >= MIN_RELEVANCE


#  Language detection 

def _detect_lang(text: str) -> str:
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "en"


# Medium RSS search 

def _search_medium_rss(topic: str) -> list[dict]:
    """
    Fetch Medium RSS for topic slugs.
    Extracts content directly from content:encoded or description
    fields — no article fetch needed, no 403 risk.
    """
    slugs    = _expand_slug(topic)
    seen_urls: set[str]   = set()
    results:   list[dict] = []

    for slug in slugs:
        rss_url = f"https://medium.com/feed/tag/{slug}"
        try:
            resp = requests.get(rss_url, headers=HEADERS, timeout=12)
            if resp.status_code != 200:
                continue

            root = ET.fromstring(resp.content)
            ns   = {
                "dc":      "http://purl.org/dc/elements/1.1/",
                "content": "http://purl.org/rss/1.0/modules/content/",
            }

            for item in root.findall(".//item"):
                raw_link = item.findtext("link", "").strip()
                if not raw_link:
                    continue

                link = _clean_url(raw_link)
                if not validate_url(link) or link in seen_urls:
                    continue
                seen_urls.add(link)

                title  = clean_text(item.findtext("title", ""))
                author = clean_text(
                    item.findtext("dc:creator", "", ns) or "Unknown"
                )
                pubdate = clean_text(item.findtext("pubDate", ""))[:16]

                # Extract full text from RSS without fetching the article
                content_el = item.find("content:encoded", ns)
                if content_el is not None and content_el.text:
                    soup_c = BeautifulSoup(content_el.text, "html.parser")
                    body   = clean_text(
                        soup_c.get_text(separator="\n\n", strip=True)
                    )
                else:
                    raw_desc = item.findtext("description", "")
                    soup_d   = BeautifulSoup(raw_desc, "html.parser")
                    body     = clean_text(
                        soup_d.get_text(separator="\n\n", strip=True)
                    )

                if not body or len(body) < 100:
                    continue

                results.append({
                    "url":    link,
                    "title":  title,
                    "author": author,
                    "date":   pubdate,
                    "body":   body,
                    "tier":   _domain_tier(link),
                })

        except Exception:
            pass
        time.sleep(0.3)

    results.sort(key=lambda x: x["tier"], reverse=True)
    return results


# Runner

def run(topic: str, output_dir: str) -> list[dict]:
    print(f"  [Blog] topic='{topic}' expanded={expand_topic(topic)}")

    candidates = _search_medium_rss(topic)
    print(f"        RSS candidates={len(candidates)}")

    results:      list[dict] = []
    seen_sources: set[str]   = set()

    for cand in candidates:
        if len(results) >= TARGET_COUNT:
            break

        url        = cand["url"]
        source_key = _diversity_key(url)

        if source_key in seen_sources:
            continue

        body      = cand["body"]
        full_text = f"{cand['title']}\n\n{body}"

        if not _is_relevant(full_text, topic):
            continue

        seen_sources.add(source_key)
        results.append({
            "source_url":     url,
            "source_type":    "blog",
            "author":         cand["author"],
            "published_date": cand["date"],
            "language":       _detect_lang(full_text),
            "region":         "Unknown",
            "topic_tags":     auto_tag(full_text),
            "trust_score":    "",
            "content_chunks": chunk_text(body),
        })
        print(f"        ✓ {url}")

    # Pad with structured error records if not enough found
    while len(results) < TARGET_COUNT:
        results.append({
            "source_url":     f"https://no-result-{len(results)+1}.placeholder",
            "source_type":    "blog",
            "author":         "Unknown",
            "published_date": "Unknown",
            "language":       "unknown",
            "region":         "Unknown",
            "topic_tags":     [],
            "trust_score":    "",
            "content_chunks": [f"[scrape_error] No blog results found for '{topic}'"],
        })

    out = Path(output_dir) / "blogs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"        → {out}")
    return results


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "diabetes",
        "task1/output/scraped_data")