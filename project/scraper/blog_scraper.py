"""
project/scraper/blog_scraper.py - Topic-driven blog scraper.

Ingestion strategy (fallback chain):
  1. Medium RSS content:encoded → full article HTML in feed (when available)
  2. Article fetch fallback    → direct GET with anti-bot headers + BS4 extraction
                                  triggered when RSS body < MIN_BODY_WORDS words
  3. RSS description fallback  → truncated preview (last resort)

Why a fallback chain vs single strategy:
  Medium RSS gives 26-30 word truncated previews ~90% of the time. Keyword
  stuffing, content quality, and anomaly detection all degrade badly on thin
  content. Fetching the full article recovers the real text for proper scoring
  without requiring a JS renderer (Medium serves static HTML to crawlers).

Extraction cascade for article fetches:
  article tag → main tag → pw-post-body-paragraph class → all <p> tags
  Each strategy is tried in priority order; first result ≥ MIN_BODY_WORDS wins.

Additional resilience:
  Circuit breaker (shared via http_client), domain-tier pre-sort, per-author
  diversity gate, positional-TF relevance filter.
"""
from __future__ import annotations

import json
import re
import sys
import time
import xml.etree.ElementTree as ET
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from bs4 import BeautifulSoup

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from project.utils.http_client   import fetch, try_extractors
from project.utils.sanitizer     import clean_text, validate_url, detect_language
from project.utils.tagging       import auto_tag
from project.utils.chunking      import rag_chunk
from project.utils.topic_expansion import expand_topic
from project.utils.semantic      import semantic_score as _semantic_score
from project.scoring.config      import CFG
from project.scoring.trust_score  import _HIGH_AUTHORITY, _MED_AUTHORITY, _domain_matches

TARGET_COUNT      = CFG.blog_target_count
MIN_BODY_WORDS    = CFG.blog_min_body_words
MIN_CONTENT_WORDS = CFG.blog_min_content_words
MIN_RELEVANCE     = CFG.blog_min_relevance

_NS = {
    "dc":      "http://purl.org/dc/elements/1.1/",
    "content": "http://purl.org/rss/1.0/modules/content/",
    "atom":    "http://www.w3.org/2005/Atom",
}

_NOISE_TAGS = ("nav", "header", "footer", "aside", "script", "style",
               "button", "form", "noscript", "iframe", "svg", "figure", "figcaption")


def expand_terms(topic: str) -> list[str]:
    """Return search terms for a topic via Claude API expansion (cached)."""
    return expand_topic(topic).terms


def _expand_slugs(topic: str) -> list[str]:
    """Return URL slugs for a topic via Claude API expansion (cached)."""
    return expand_topic(topic).slugs


from functools import lru_cache as _lru_cache


def _anchor_count_pattern(anchor: str) -> str:
    """
    Single regex pattern that matches all accepted forms of the anchor.

    Three forms unified into one alternation:
      • exact whole-word match
      • trailing-s stripped (plural → singular: 'parkinsons' → 'parkinson')
      • stem-prefix for words ≥7 chars, accepting derivational forms.
        'inflammation' (12c) → stem 'inflammat' + \\w* matches
        'inflammation', 'inflammatory', 'inflammable'.

    Used by BOTH the anchor gate AND the density counter so the two stay
    consistent. Earlier bug: anchor gate matched 'inflammatory' via stem
    while density counter only counted exact 'inflammation' — articles
    passed the gate but failed total ≥ 2 silently.
    """
    if len(anchor) >= 7:
        stem_len = max(5, len(anchor) - 3)
        return r"\b" + re.escape(anchor[:stem_len]) + r"\w*\b"
    if len(anchor) > 4 and anchor.endswith("s") and not anchor.endswith("ss"):
        return r"\b(?:" + re.escape(anchor) + r"|" + re.escape(anchor[:-1]) + r")\b"
    return r"\b" + re.escape(anchor) + r"\b"


def _anchor_match_patterns(anchor: str) -> tuple[str, ...]:
    """Backward-compat wrapper for the gate; returns a one-element tuple."""
    return (_anchor_count_pattern(anchor),)


@_lru_cache(maxsize=256)
def _topic_singletons(topic: str) -> tuple[str, ...]:
    """
    Single-word content tokens of the raw topic. Used both as a relevance
    fallback (when expansion returns only multi-word phrases like 'sleep
    physiology' but the doc just says 'sleep') AND as a strict gate (every
    singleton must appear, so the topic 'AI safety' cannot match a balcony-
    solar article that happens to mention 'safety' once).

    len >= 2 keeps important short tokens like 'ai', 'ml', 'ux'. Stop words
    and pure punctuation are filtered out.

    Result is cached per unique topic — `_is_relevant` runs this on every
    candidate, but a pipeline run only sees a handful of distinct topics.
    Returns a tuple (immutable) so the cache key stays hashable.
    """
    from project.utils.sanitizer import STOP_WORDS as _SW
    return tuple(w for w in re.split(r"[^a-z0-9]+", topic.lower())
                 if w and w not in _SW and len(w) >= 2)


def _is_relevant(text: str, topic: str) -> bool:
    """
    Two-gate relevance check:
      1. Density gate    — combined topic-term hits / word count >= MIN_RELEVANCE
      2. Absolute floor  — total topic-term hits >= 2

    The 5-topic audit found two off-topic records that passed density alone:
      • 'AI safety' admitted a balcony-solar MIT-Tech-Review article because
        'safety' appeared once in 200+ words (density 0.005, just above 0.003).
      • 'kubernetes' admitted a HuggingFace UI-model post via single match.
    A 1-hit-in-200-words match is statistical noise, not topical relevance.
    Requiring >= 2 hits eliminates drive-by matches without false-negatives
    on legitimately on-topic content (which always mentions topic terms 2+).

    Strict-AND on every singleton was tried first but rejected sleep abstracts
    that say 'sleep' 4× but never 'science' — generic descriptor singletons
    ('science', 'engineering', 'system') aren't always echoed in body text.
    """
    if not text:
        return False
    terms      = expand_terms(topic)
    singletons = _topic_singletons(topic)
    low        = text.lower()
    words      = low.split()
    if not words:
        return False

    # Anchor gate: the FIRST content token of the raw topic must appear at
    # least once as a whole word. Multi-word topics have a "subject" word
    # (climate, parkinsons, quantum) and a "descriptor" (change, disease,
    # computing). Descriptors are too common in English/medical/CS prose to
    # gate on alone — the 4-topic audit found cannabis-rescheduling articles
    # passing 'climate change' because 'change' appeared 5+ times. The
    # subject word, by contrast, is reliably distinctive.
    #
    # See _anchor_match_patterns for the three accepted forms:
    # exact word, plural→singular stem, and ≥7-char prefix-stem for
    # derivational forms (inflammation ↔ inflammatory).
    if singletons:
        patterns = _anchor_match_patterns(singletons[0])
        if not any(re.search(p, low) for p in patterns):
            return False

    total = 0
    # Multi-word expansion phrases: substring count (false matches rare)
    for t in terms:
        t = t.lower()
        if " " in t:
            total += low.count(t)
    # Single-word singletons: stem-aware regex count, same pattern the anchor
    # gate uses. Critical for words like 'inflammation' where body text says
    # 'inflammatory' — a count via words.count("inflammation") returns 0 even
    # though the article is clearly inflammation-related. The unified pattern
    # catches 'inflammation', 'inflammatory', 'inflammable' as one count.
    seen_singletons: set[str] = set()
    for s in singletons:
        if s in seen_singletons:
            continue
        seen_singletons.add(s)
        total += len(re.findall(_anchor_count_pattern(s), low))
    # Also count expansion terms that are single words (e.g. 'cytokines')
    for t in terms:
        t = t.lower()
        if " " not in t and t not in seen_singletons:
            seen_singletons.add(t)
            total += len(re.findall(_anchor_count_pattern(t), low))

    if total < 2:
        return False
    if total / len(words) < MIN_RELEVANCE:
        return False

    # Optional third gate: semantic similarity. Active only when
    # sentence-transformers is installed. Catches the failure mode keyword
    # gates can't — articles that legitimately mention the topic terms but
    # cover a tangential subject (e.g. a HuggingFace blog about a UI model
    # that mentions 'kubernetes' twice in passing). Falls back silently when
    # the package isn't available.
    sim = _semantic_score(topic, text)
    if sim is not None and sim < CFG.semantic_min_similarity:
        return False
    return True


# ── URL utilities ──────────────────────────────────────────────────────────────

def _clean_url(url: str) -> str:
    try:
        p = urlparse(url)
        return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))
    except Exception:
        return url


def _diversity_key(url: str) -> str:
    """Per-author key for Medium; per-domain for everything else."""
    try:
        p = urlparse(url)
        netloc = p.netloc.lower().replace("www.", "")
        if "medium.com" in netloc:
            parts = [s for s in p.path.split("/") if s]
            return f"{netloc}/{parts[0]}" if parts else netloc
        return netloc
    except Exception:
        return url


def _domain_tier(url: str) -> int:
    """Pre-sort tier for RSS candidates. Uses shared authority sets from trust_score."""
    try:
        d = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return 0
    if _domain_matches(d, _HIGH_AUTHORITY):
        return 2
    if _domain_matches(d, _MED_AUTHORITY):
        return 1
    return 0


# ── article body extraction ────────────────────────────────────────────────────

_MEDIUM_META_RE = re.compile(
    r"^.{0,500}?\b\d+\s+min\s+read\b.{0,150}?--\n+",
    re.DOTALL,
)

# ScienceDaily "Cite This Page" and related-content sections begin with these markers
_SD_CITE_RE    = re.compile(r"\n\s*(?:CITE THIS PAGE|Cite This Page|cite this page)[:\s]", re.IGNORECASE)
_SD_RELATED_RE = re.compile(r"\n\s*(?:RELATED TOPICS|RELATED STORIES|Related Topics|Related Stories)\s*\n", re.IGNORECASE)


def _strip_medium_metadata(text: str) -> str:
    """Remove Medium header block (author · read-time · timestamp --) from article start."""
    return _MEDIUM_META_RE.sub("", text, count=1)


def _extract_medium_body(html: str) -> str:
    """Cascade of Medium-specific extraction strategies."""
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    # Strategy 1: semantic article tag (most reliable)
    article = soup.find("article")
    if article:
        return _strip_medium_metadata(clean_text(article.get_text(separator="\n\n", strip=True)))

    # Strategy 2: Medium's post-body class (structure-aware)
    pw_body = soup.find_all(class_=re.compile(r"pw-post-body", re.I))
    if pw_body:
        return _strip_medium_metadata(clean_text("\n\n".join(el.get_text(strip=True) for el in pw_body)))

    # Strategy 3: main content area
    main = soup.find("main") or soup.find("div", id=re.compile(r"(content|main)", re.I))
    if main:
        return _strip_medium_metadata(clean_text(main.get_text(separator="\n\n", strip=True)))

    # Strategy 4: paragraph harvest (last resort)
    paras = [p.get_text(strip=True) for p in soup.find_all("p")
             if len(p.get_text(strip=True)) > CFG.blog_min_para_chars]
    return clean_text("\n\n".join(paras))


def _extract_sciencedaily_body(html: str) -> str:
    """
    ScienceDaily-specific extractor. Targets div#story-inner to get the article
    text and excludes the 'Cite This Page' and 'Related Stories' sections that
    follow the article and contaminate chunk quality when included.
    """
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(_NOISE_TAGS):
        tag.decompose()

    story = (soup.find("div", id="story-inner") or
             soup.find("div", id="story") or
             soup.find("div", class_=re.compile(r"story-text", re.I)))
    if story:
        text = clean_text(story.get_text(separator="\n\n", strip=True))
        # Truncate at first "Cite This Page" or "Related Stories" marker
        for pattern in (_SD_CITE_RE, _SD_RELATED_RE):
            m = pattern.search(text)
            if m:
                text = text[:m.start()].strip()
        return text
    return ""


# Common HTML markup that publishers use for the canonical published date.
# Healthline's RSS omits date fields entirely; the article HTML carries it.
# Patterns are tried in order — JSON-LD `"datePublished"` and `<meta>` are
# the most reliable; `<time datetime>` is the loosest fallback.
# Healthline specifically uses `<meta name="article:published_time" ...>` —
# note `name=`, not the more standard Open-Graph `property=`. We accept both.
_HTML_DATE_PATTERNS = (
    # JSON-LD structured data (most reliable, machine-readable)
    re.compile(r'"datePublished"\s*:\s*"([^"]+)"', re.IGNORECASE),
    # Open-Graph article meta (property= variant per OG spec)
    re.compile(r'<meta[^>]+property=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+property=["\']article:published_time["\']', re.IGNORECASE),
    # name= variant (Healthline, some WordPress themes)
    re.compile(r'<meta[^>]+name=["\']article:published_time["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+content=["\']([^"\']+)["\'][^>]+name=["\']article:published_time["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+name=["\']pubdate["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+name=["\']publication_date["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<meta[^>]+itemprop=["\']datePublished["\'][^>]+content=["\']([^"\']+)["\']', re.IGNORECASE),
    re.compile(r'<time[^>]+datetime=["\']([^"\']+)["\']', re.IGNORECASE),
)


def _extract_html_date(html: str) -> str:
    """
    Extract the published date from HTML meta tags. Returns '' if none found.

    Used to backfill records whose RSS feed omitted the date field. Healthline's
    health-news RSS feed has no <pubDate>, <dc:date>, or Atom date elements at
    all — the article HTML carries it via <meta property="article:published_time">.
    """
    for pat in _HTML_DATE_PATTERNS:
        m = pat.search(html)
        if m:
            return m.group(1).strip()
    return ""


def _fetch_article_body(url: str) -> tuple[str, str]:
    """
    Attempt to fetch and extract the full article body from the article URL.
    Routes to domain-specific extractor when available; falls back to generic.
    Returns (body, published_date). Either may be "" on failure.
    """
    resp = fetch(url)
    if resp is None:
        return "", ""
    try:
        domain = urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        domain = ""
    if "sciencedaily.com" in domain:
        body = try_extractors(resp.text, [_extract_sciencedaily_body, _extract_medium_body])
    else:
        body = try_extractors(resp.text, [_extract_medium_body])
    date = _extract_html_date(resp.text)
    return body, date


# ── RSS source registry ────────────────────────────────────────────────────────
# Ordered by content quality. Sources with full_content=True serve complete
# article text in RSS (no article-fetch needed). Slug-based sources ({slug})
# are fetched once per topic expansion; static sources are fetched once total.

_RSS_SOURCES: list[dict] = [
    # Healthline general health news - full article body in RSS (1000+ words avg)
    {
        "name": "healthline_news",
        "url_tpl": "https://www.healthline.com/rss/health-news",
        "slug_based": False, "full_content": True, "tier_override": 2,
        "author_default": "Healthline Editorial",
        "categories": {"health"},
    },
    # ScienceDaily - 60-80 word summaries only; fetch full article from URL
    {
        "name": "sciencedaily_health",
        "url_tpl": "https://www.sciencedaily.com/rss/top/health.xml",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "ScienceDaily",
        "categories": {"health"},
    },
    # ScienceDaily - computers/tech for AI topics
    {
        "name": "sciencedaily_tech",
        "url_tpl": "https://www.sciencedaily.com/rss/top/technology.xml",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "ScienceDaily",
        "categories": {"tech"},
    },
    # NIH News in Health - authoritative consumer health content
    {
        "name": "nih_news",
        "url_tpl": "https://newsinhealth.nih.gov/feeds/rss",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "NIH News in Health",
        "categories": {"health"},
    },
    # STAT News - premium health and science journalism
    {
        "name": "statnews",
        "url_tpl": "https://www.statnews.com/feed/",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "STAT News",
        "categories": {"health"},
    },
    # Harvard Health Publishing Blog - physician-authored health content
    {
        "name": "harvard_health",
        "url_tpl": "https://www.health.harvard.edu/blog/feed",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "Harvard Health Publishing",
        "categories": {"health"},
    },
    # Mayo Clinic News Network - institutional clinical news
    {
        "name": "mayo_clinic_news",
        "url_tpl": "https://newsnetwork.mayoclinic.org/feed/",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "Mayo Clinic",
        "categories": {"health"},
    },
    # The Conversation - expert-written articles by academics
    {
        "name": "the_conversation",
        "url_tpl": "https://theconversation.com/articles.atom",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "The Conversation",
        "categories": {"health", "tech"},
    },
    # MIT Technology Review - AI/science research coverage
    {
        "name": "mit_tech_review",
        "url_tpl": "https://www.technologyreview.com/feed/",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "MIT Technology Review",
        "categories": {"tech"},
    },
    # Towards Data Science - best ML/AI blog coverage via Medium
    {
        "name": "towardsdatascience",
        "url_tpl": "https://towardsdatascience.com/feed",
        "slug_based": False, "full_content": False, "tier_override": 1,
        "author_default": "Towards Data Science",
        "categories": {"tech"},
    },
    # The Verge - broad tech/science news coverage
    {
        "name": "the_verge",
        "url_tpl": "https://www.theverge.com/rss/index.xml",
        "slug_based": False, "full_content": False, "tier_override": 1,
        "author_default": "The Verge",
        "categories": {"tech"},
    },
    # HuggingFace Blog - authoritative AI/ML content; RAG, LLMs, embeddings
    {
        "name": "huggingface_blog",
        "url_tpl": "https://huggingface.co/blog/feed.xml",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "HuggingFace",
        "categories": {"tech"},
    },
    # LangChain Blog - directly covers RAG, agents, vector stores, LLM tooling
    {
        "name": "langchain_blog",
        "url_tpl": "https://blog.langchain.dev/rss/",
        "slug_based": False, "full_content": False, "tier_override": 2,
        "author_default": "LangChain",
        "categories": {"tech"},
    },
    # Medium tag RSS - truncated previews; broad coverage; universal fallback
    {
        "name": "medium",
        "url_tpl": "https://medium.com/feed/tag/{slug}",
        "slug_based": True, "full_content": False, "tier_override": None,
        "author_default": "Unknown",
        "categories": None,  # None = universal
    },
]

def _topic_category(topic: str) -> str:
    """Route to tech vs health RSS sources via Claude API expansion (cached)."""
    return "tech" if expand_topic(topic).is_technical else "health"


def _parse_rss_items(content: bytes, source_meta: dict) -> list[dict]:
    """Parse RSS XML bytes into candidate dicts, normalising across feed formats."""
    try:
        root = ET.fromstring(content)
    except ET.ParseError as e:
        print(f"        ⚠ RSS XML parse error [{source_meta.get('name', '?')}]: {e}")
        return []

    items = []
    for item in root.findall(".//item"):
        try:
            raw_link = item.findtext("link", "").strip()
            if not raw_link:
                continue
            link = _clean_url(raw_link)
            if not validate_url(link):
                continue

            title   = clean_text(item.findtext("title", ""))
            author  = clean_text(
                item.findtext("dc:creator", "", _NS)
                or item.findtext("author", "")
                or source_meta["author_default"]
            )
            # Date fallback chain: pubDate (RSS 2.0) → dc:date (Dublin Core) →
            # atom:published / atom:updated (Atom feeds embedded in RSS).
            # Healthline's health-news feed populated only one of these in
            # ~14% of items; previous code grabbed pubDate and gave up,
            # leaving date='Unknown' for high-trust records that should have
            # had a parseable date.
            pubdate = clean_text(
                item.findtext("pubDate", "")
                or item.findtext("dc:date", "", _NS)
                or item.findtext("atom:published", "", _NS)
                or item.findtext("atom:updated", "", _NS)
                or item.findtext("published", "")
                or item.findtext("updated", "")
            ) or "Unknown"

            # Body extraction cascade: content:encoded → description
            ce = item.find("content:encoded", _NS)
            if ce is not None and ce.text:
                body = clean_text(
                    BeautifulSoup(ce.text, "lxml").get_text(separator="\n\n", strip=True)
                )
            else:
                raw_desc = item.findtext("description", "")
                body = clean_text(
                    BeautifulSoup(raw_desc, "html.parser")
                    .get_text(separator="\n\n", strip=True)
                ) if raw_desc else ""

            # Remove trailer text ("Continue reading on Medium »", etc.)
            body = re.sub(r"\s*(Continue reading|Read more).*$", "", body,
                          flags=re.IGNORECASE).strip()

            tier = source_meta["tier_override"] if source_meta["tier_override"] is not None \
                   else _domain_tier(link)

            items.append({
                "url": link, "title": title, "author": author,
                "date": pubdate, "body": body, "tier": tier,
                "source_name": source_meta["name"],
                "full_content": source_meta["full_content"],
            })
        except Exception:
            # One malformed item must not abort the entire feed parse.
            continue
    return items


def _is_good_source_item(item: dict, topic: str) -> bool:
    """An item that, on its own, would already qualify for selection."""
    return (item["full_content"]
            and len(item["body"].split()) >= MIN_BODY_WORDS
            and _is_relevant(f"{item['title']} {item['body']}", topic))


def _fetch_rss(slug: str, topic: str, fetched_static: set[str],
               good_sources: set[str]) -> list[dict]:
    """
    Multi-source RSS fetcher — parallel across sources, sequential across slugs.

    Each RSS source is a different host (Healthline, ScienceDaily, NIH, MIT
    Tech Review, …) so concurrent fetches don't risk per-host rate limiting.
    Wall-clock for a slug = max(per-source fetch time), not sum.

    Static sources (Healthline news, ScienceDaily…) are fetched only ONCE
    across all slug iterations via `fetched_static`. Slug-based sources
    (Medium tag feeds) re-fetch per slug.

    Category filtering ensures health sources don't appear for AI topics
    and vice versa.

    Early-exit at TARGET_COUNT diverse `good_sources` is preserved at the
    slug-loop level in `run()`. We do NOT short-circuit individual sources
    inside one parallel wave — once submitted, we collect all results.
    """
    cat = _topic_category(topic)

    # Build the fetch plan up front so we know which static URLs to claim.
    # Claiming happens BEFORE we submit, so concurrent slug iterations (if any
    # ever exist) won't double-fetch a static source.
    fetch_jobs: list[tuple[str, dict]] = []
    for source in _RSS_SOURCES:
        if source.get("categories") is not None and cat not in source["categories"]:
            continue
        if source["slug_based"]:
            url = source["url_tpl"].format(slug=slug)
        else:
            url = source["url_tpl"]
            if url in fetched_static:
                continue
            fetched_static.add(url)
        fetch_jobs.append((url, source))

    if not fetch_jobs:
        return []

    all_items: list[dict] = []
    with ThreadPoolExecutor(max_workers=min(CFG.blog_rss_max_workers, len(fetch_jobs))) as ex:
        future_to_source = {ex.submit(fetch, url): src for url, src in fetch_jobs}
        for future in as_completed(future_to_source):
            src = future_to_source[future]
            try:
                resp = future.result()
            except Exception:
                continue
            if resp is None:
                continue
            items = _parse_rss_items(resp.content, src)
            all_items.extend(items)
            for item in items:
                if _is_good_source_item(item, topic):
                    good_sources.add(_diversity_key(item["url"]))

    return all_items


# ── runner ─────────────────────────────────────────────────────────────────────

def run(topic: str, output_dir: str) -> list[dict]:
    fetched_static: set[str] = set()   # local per-run; thread-safe (no shared state)
    print(f"  [Blog] topic='{topic}'  terms={expand_terms(topic)}")

    seen_urls:    set[str] = set()
    candidates:   list[dict] = []
    good_sources: set[str] = set()   # shared with _fetch_rss for O(1) early-exit

    for slug in _expand_slugs(topic):
        for item in _fetch_rss(slug, topic, fetched_static, good_sources):
            if item["url"] not in seen_urls:
                seen_urls.add(item["url"])
                candidates.append(item)

        if len(good_sources) >= TARGET_COUNT:
            break

        time.sleep(CFG.delay_blog_rss)

    candidates.sort(key=lambda x: x["tier"], reverse=True)
    print(f"        RSS candidates={len(candidates)}")

    results:      list[dict] = []
    seen_sources: set[str]   = set()
    domain_counts: dict[str, int] = {}   # base domain → count of accepted records

    for cand in candidates:
        if len(results) >= TARGET_COUNT:
            break

        source_key = _diversity_key(cand["url"])
        if source_key in seen_sources:
            continue

        # Domain-level diversity cap: prevent all records from one base domain.
        # Medium per-author dedup still applies above; this adds a ceiling per
        # base domain so e.g. 3 different Medium authors don't fill all slots.
        try:
            base_domain = urlparse(cand["url"]).netloc.lower().replace("www.", "")
        except Exception:
            base_domain = ""
        if base_domain and domain_counts.get(base_domain, 0) >= CFG.blog_max_per_domain:
            continue

        body = cand["body"]

        # Quick relevance pre-filter: if title + RSS preview together don't
        # mention the topic AND the preview has at least 10 words of signal,
        # skip the expensive article fetch - it won't help.
        preview_text = f"{cand['title']} {body}"
        has_preview_signal = len(body.split()) >= 10
        if has_preview_signal and not _is_relevant(preview_text, topic):
            continue

        # Pre-fetch anchor check on the TITLE. Independent of body length.
        # HuggingFace and similar feeds serve titles + ultra-thin RSS bodies
        # (<10 words), so `has_preview_signal` is False for most of them and
        # the check above never fires. Result: we used to fetch every single
        # HF blog post (300+ articles) only to reject post-fetch. The title
        # alone is a strong signal — if the topic's anchor word isn't there,
        # the article is almost never about the topic. Skip the fetch.
        singletons = _topic_singletons(topic)
        if singletons:
            patterns = _anchor_match_patterns(singletons[0])
            title_low = cand["title"].lower()
            if not any(re.search(p, title_low) for p in patterns):
                continue

        # Article fetch fallback for thin RSS previews (e.g. Medium truncates at ~25w).
        # Only attempted when content is below MIN_BODY_WORDS and source doesn't
        # serve full content in-feed.
        cand_date_lc = cand.get("date", "").strip().lower()
        if len(body.split()) < MIN_BODY_WORDS and not cand.get("full_content"):
            fetched, fetched_date = _fetch_article_body(cand["url"])
            if len(fetched.split()) > len(body.split()):
                body = fetched
                print(f"        ↑ fetch: {len(body.split())} words  {cand['url'][-CFG.display_url_fetch_chars:]}")
            # Backfill date from HTML meta tags when RSS gave nothing useful.
            if fetched_date and cand_date_lc in CFG.missing_date_sentinels:
                cand["date"] = fetched_date
        elif cand_date_lc in CFG.missing_date_sentinels and cand.get("full_content"):
            # Body is fine (full-content feed), but date is missing. Healthline
            # health-news RSS omits <pubDate> entirely; the date is in the
            # article HTML's <meta> / JSON-LD. One extra GET buys us a real date
            # instead of 'Unknown'. Cost: 1 request per kept full-content record
            # whose feed lacks dates.
            _, fetched_date = _fetch_article_body(cand["url"])
            if fetched_date:
                cand["date"] = fetched_date

        # Content length cap: truncate before O(n) anomaly detection to prevent
        # runaway processing on scraped pages that include nav/footer boilerplate.
        body_words = body.split()
        if len(body_words) > CFG.blog_max_content_words:
            body = " ".join(body_words[:CFG.blog_max_content_words])

        # Hard content floor: skip records where fetch failed to recover real content.
        # A 20-word snippet is not useful for RAG and would only pollute the output.
        if len(body.split()) < MIN_CONTENT_WORDS:
            continue

        full_text = f"{cand['title']}\n\n{body}"
        if not _is_relevant(full_text, topic):
            continue

        seen_sources.add(source_key)
        domain_counts[base_domain] = domain_counts.get(base_domain, 0) + 1
        chunks = rag_chunk(body)

        # Ensure at least one chunk even for very thin content
        if not chunks:
            chunks = [{"text": body, "chunk_index": 0, "total_chunks": 1,
                       "quality_score": CFG.chunk_fallback_quality_blog, "has_overlap": False}]

        results.append({
            "source_url":     cand["url"],
            "source_type":    "blog",
            "author":         cand["author"],
            "published_date": cand["date"],
            "language":       detect_language(full_text),
            "region":         "Unknown",
            "topic_tags":     auto_tag(full_text),
            "content_chunks": chunks,
        })
        wc = sum(len(c["text"].split()) for c in chunks)
        print(f"        ✓ {cand['url'][-CFG.display_url_accept_chars:]}  words={wc}")

    out = Path(output_dir) / "blogs.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"        → {out}")
    return results
