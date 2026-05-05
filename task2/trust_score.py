"""
task2/trust_score.py
====================
Trust Score Engine.

Formula
-------
    raw_score   = Σ (weight_i × sub_score_i)    weighted sum  → [0.0, 1.0]
    final_score = raw_score × abuse_penalty      multiplicative → [0.0, 1.0]

Source-type calibrated weights (each set sums to 1.0):
  blog    : domain_authority + recency dominate
  youtube : author credibility + recency dominate
  pubmed  : citation_count weighted highest

Complexity: O(T) for content checks; O(1) elsewhere.
Security  : expects pre-sanitised strings.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from datetime import datetime
from urllib.parse import urlparse

_WEIGHTS: dict[str, dict[str, float]] = {
    "blog": {
        "author_credibility": 0.20,
        "citation_count":     0.10,
        "domain_authority":   0.35,
        "recency":            0.25,
        "medical_disclaimer": 0.10,
    },
    "youtube": {
        "author_credibility": 0.30,
        "citation_count":     0.10,
        "domain_authority":   0.20,
        "recency":            0.30,
        "medical_disclaimer": 0.10,
    },
    "pubmed": {
        "author_credibility": 0.25,
        "citation_count":     0.35,
        "domain_authority":   0.15,
        "recency":            0.15,
        "medical_disclaimer": 0.10,
    },
}

_HIGH_AUTH = frozenset({
    "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "who.int", "cdc.gov",
    "nih.gov", "nature.com", "science.org", "nejm.org", "thelancet.com",
    "jamanetwork.com", "bmj.com", "cell.com", "arxiv.org", "ieee.org",
    "acm.org", "mit.edu", "stanford.edu", "harvard.edu", "cam.ac.uk",
    "ox.ac.uk", "openai.com", "anthropic.com", "deepmind.google",
    "deepmind.com", "research.google", "ai.google", "microsoft.com",
    "mayoclinic.org", "diabetes.org", "hopkinsmedicine.org",
})

_MED_AUTH = frozenset({
    "medium.com", "towardsdatascience.com", "hackernoon.com",
    "techcrunch.com", "wired.com", "theverge.com", "zdnet.com",
    "youtube.com", "youtu.be", "substack.com", "huggingface.co",
    "cloud.google.com", "developers.googleblog.com", "healthline.com",
    "webmd.com",
})

_LOW_AUTH_RE = re.compile(
    r"(\.blogspot\.|\.wordpress\.com|\.weebly\.|clickbait|"
    r"top\d+tricks|youwontbelieve|shocking)",
    re.IGNORECASE,
)

_CREDIBLE_ORG_RE = re.compile(
    r"\b(nih|who|cdc|mayo\s+clinic|harvard|stanford|ieee|acm|"
    r"openai|anthropic|deepmind|google|microsoft|meta|nature|"
    r"elsevier|springer|hugging\s*face|johns\s+hopkins)\b",
    re.IGNORECASE,
)

_HEALTH_RE = re.compile(
    r"\b(disease|cancer|diagnosis|treatment|drug|medication|symptom|"
    r"clinical|patient|therapy|vaccine|health|diabetes|insulin|glucose)\b",
    re.IGNORECASE,
)

_DISCLAIMER_RE = re.compile(
    r"(consult\s+(a\s+)?(doctor|physician|medical\s+professional|"
    r"healthcare\s+provider)|not\s+(medical\s+)?advice|"
    r"informational\s+purposes\s+only|seek\s+the\s+advice\s+of\s+your\s+"
    r"(doctor|physician)|medical\s+disclaimer)",
    re.IGNORECASE,
)

_DATE_FMTS = (
    "%Y-%m-%d", "%Y/%m/%d", "%Y-%m-%dT%H:%M:%S",
    "%B %d, %Y", "%d %B %Y", "%b %d, %Y", "%Y",
)


def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


def _parse_date(s: str) -> datetime | None:
    if not s or s.lower() in ("unknown", "n/a", ""):
        return None
    for fmt in _DATE_FMTS:
        try:
            return datetime.strptime(s.strip()[:20], fmt)
        except ValueError:
            continue
    return None


def _author_score(author: str, url: str) -> float:
    """Multiple authors → average individual scores (assignment edge case)."""
    domain = _netloc(url)
    is_high = any(domain.endswith(d) for d in _HIGH_AUTH)

    if author and "," in author and not _CREDIBLE_ORG_RE.search(author):
        parts = [p.strip() for p in author.split(",") if p.strip()]
        if len(parts) > 1:
            return round(sum(_author_score(p, url) for p in parts) / len(parts), 4)

    no_author = not author or author.lower() in ("unknown", "n/a", "")
    if no_author:
        return 0.30 if is_high else 0.10
    if _CREDIBLE_ORG_RE.search(author):
        return 1.0
    return 0.70 if is_high else 0.50


def _citation_score(citations: int | None, source_type: str,
                    download_count: int | None) -> float:
    if source_type == "pubmed":
        if citations is None:
            return 0.40
        return min(citations / 200.0, 1.0)
    if download_count is not None and download_count > 0:
        return min(math.log10(download_count) / 6.0, 1.0)
    return 0.50


def _domain_score(url: str) -> float:
    domain = _netloc(url)
    if not domain:
        return 0.10
    if any(domain.endswith(d) for d in _HIGH_AUTH):
        return 0.95
    if any(domain.endswith(d) for d in _MED_AUTH):
        return 0.60
    if _LOW_AUTH_RE.search(url):
        return 0.15
    return 0.40


def _recency_score(published_date: str, source_type: str) -> float:
    """Source-type-aware decay. PubMed papers age slower than blogs."""
    dt = _parse_date(published_date)
    if dt is None:
        return 0.30
    age_days = (datetime.now() - dt).days

    if source_type == "pubmed":
        if age_days < 365:   return 1.00
        if age_days < 730:   return 0.85
        if age_days < 1825:  return 0.65
        if age_days < 3650:  return 0.40
        return 0.20
    if source_type == "youtube":
        if age_days < 180:   return 1.00
        if age_days < 365:   return 0.85
        if age_days < 730:   return 0.65
        if age_days < 1825:  return 0.40
        return 0.20
    # blog
    if age_days < 90:    return 1.00
    if age_days < 180:   return 0.85
    if age_days < 365:   return 0.70
    if age_days < 730:   return 0.50
    if age_days < 1825:  return 0.30
    return 0.10


def _disclaimer_score(content: str, source_type: str) -> float:
    """PubMed → 0.5 neutral. Blog/YouTube → 1.0 if disclaimer, 0.0 if missing on health."""
    if source_type == "pubmed":
        return 0.50
    if not _HEALTH_RE.search(content):
        return 0.50
    return 1.0 if _DISCLAIMER_RE.search(content) else 0.0


def _abuse_penalty(author: str, url: str, content: str,
                   published_date: str, source_type: str) -> float:
    m = 1.0
    is_health = bool(_HEALTH_RE.search(content))

    # 1. Anonymous author on health content
    if source_type != "pubmed" and is_health:
        if not author or author.lower() in ("unknown", "n/a", ""):
            m *= 0.70

    # 2. SEO spam URL patterns
    spam_hits = len(re.findall(
        r"(top\d+|best\d+|\d+tips|click|free.?download|secret)",
        url, re.IGNORECASE,
    ))
    if spam_hits >= 2:
        m *= 0.75

    # 3. Keyword stuffing
    words = content.lower().split()
    if len(words) > 50:
        top_freq = Counter(words).most_common(1)[0][1]
        if top_freq / len(words) > 0.05:
            m *= 0.80

    # 4. Outdated health content
    if source_type != "pubmed" and is_health:
        dt = _parse_date(published_date)
        if dt and (datetime.now() - dt).days > 730:
            m *= 0.80

    # 5. Low-authority domain + health content
    if is_health and _domain_score(url) < 0.30:
        m *= 0.75

    return max(round(m, 4), 0.05)


def calculate(
    source_url:     str,
    source_type:    str,
    author:         str,
    published_date: str,
    content:        str,
    citations:      int | None = None,
    download_count: int | None = None,
) -> dict:
    """Returns {trust_score, sub_scores, weights, abuse_penalty}."""
    weights = _WEIGHTS.get(source_type, _WEIGHTS["blog"])
    sub = {
        "author_credibility": _author_score(author, source_url),
        "citation_count":     _citation_score(citations, source_type, download_count),
        "domain_authority":   _domain_score(source_url),
        "recency":            _recency_score(published_date, source_type),
        "medical_disclaimer": _disclaimer_score(content, source_type),
    }
    raw   = sum(weights[k] * sub[k] for k in weights)
    abuse = _abuse_penalty(author, source_url, content, published_date, source_type)
    final = round(min(max(raw * abuse, 0.0), 1.0), 3)
    return {
        "trust_score":   final,
        "sub_scores":    {k: round(v, 3) for k, v in sub.items()},
        "weights":       weights,
        "abuse_penalty": abuse,
    }