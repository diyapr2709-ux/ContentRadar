"""
project/scoring/trust_score.py - Trust-aware content reliability engine.

Scoring Formula
───────────────
  Trust Score = f(9 signals, confidence-weighted fusion)

  raw_score   = Σ (weight_i × score_i × confidence_i) / Σ (weight_i × confidence_i)
  final_score = clamp(raw_score × abuse_penalty × prior_pull, 0.0, 1.0)

9 Signals (source-type-calibrated weights in config.py)
────────────────────────────────────────────────────────
  author_credibility           - named/anonymous; credible-org cross-check; domain consistency
  citation_count               - academic citations (PubMed) / view-count log-proxy (blog/YT)
  domain_authority             - tiered whitelist + spam pattern detection
  recency                      - smooth exponential decay (half-life from config)
  medical_disclaimer_presence  - mandatory on health content for blog/youtube
  metadata_completeness        - structural completeness; DOI/PMID format-validated
  content_depth                - word count + type-token ratio + inverted Gini coefficient
  engagement_authenticity      - YouTube like/view ratio vs. log-normal expected (Borghol 2012)
  adversarial_risk             - Unicode bidi attacks, prompt injection, n-gram repetition

Adaptive fusion: confidence is evidence-driven, not fixed. Low-evidence signals
contribute less regardless of nominal weight. No magic numbers in this file -
all thresholds, penalties, and scores come from project/scoring/config.py.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import threading
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from project.scoring.config import CFG, WEIGHTS, scoring_now
from project.utils.sanitizer import STOP_WORDS as _STOP_WORDS
from project.scoring.signals import (
    Signal,
    metadata_completeness_signal,
    content_depth_signal,
    engagement_authenticity_signal,
    adversarial_risk_signal,
)

# ── authority domain lists (frozensets built from CFG for O(1) lookup) ─────────

_HIGH_AUTHORITY = frozenset(CFG.high_authority_domains)
_MED_AUTHORITY  = frozenset(CFG.med_authority_domains)
_LOW_AUTH_RE = re.compile(
    r"(\.blogspot\.|\.wordpress\.com|\.weebly\.|clickbait|"
    r"top\d+tricks|youwontbelieve|shocking|secretcure|"
    r"natural-?cure|miracle|detox\d|slim\d|"
    r"lose-\d+-pounds|weight-loss-secret|doctors-hate|"
    r"one-weird-trick|this-one-thing|you-wont-believe)",
    re.IGNORECASE,
)
_CREDIBLE_ORG_RE = re.compile(
    r"\b(nih|who|cdc|mayo\s+clinic|harvard|stanford|yale|mit|johns\s+hopkins|"
    r"oxford|cambridge|ieee|acm|openai|anthropic|deepmind|"
    r"google\s+(brain|research)|microsoft\s+research|meta\s+ai|"
    r"elsevier|springer|nature|hugging\s*face|"
    r"american\s+(heart|cancer|diabetes|medical)\s+(association|society)|"
    r"national\s+(institutes?\s+of\s+health|cancer\s+institute|heart|"
    r"institute\s+of\s+(mental\s+health|neurological))|"
    r"world\s+health\s+organization|centers\s+for\s+disease\s+control|"
    r"food\s+and\s+drug\s+administration|\bfda\b|\bema\b|"
    r"european\s+medicines\s+agency|"
    r"lancet|jama|\bbmj\b|\bnejm\b|"
    r"university\s+of\s+(california|michigan|washington|toronto|melbourne)|"
    r"(imperial|kings|university)\s+college\s+london)\b",
    re.IGNORECASE,
)

# ── health / disclaimer detection ──────────────────────────────────────────────

_HEALTH_RE = re.compile(
    r"\b(disease|cancer|diagnosis|treatment|drug|medication|symptom|"
    r"clinical|patient|therapy|vaccine|health|diabetes|insulin|glucose|"
    r"cardiovascular|hypertension|oncology|surgery|prescription|"
    r"obesity|dementia|alzheimer|depression|anxiety|mental\s+health|"
    r"genomics|genome|pathology|prognosis|epidemiology|mortality|"
    r"antibiotic|dosage|contraindication|side\s+effect|adverse\s+event|"
    r"microbiome|probiotic|nutrition|dietary|supplement|vitamin)\b",
    re.IGNORECASE,
)
_DISCLAIMER_RE = re.compile(
    r"(consult\s+(a\s+)?(doctor|physician|medical\s+professional|"
    r"healthcare\s+(provider|professional))|"
    r"not\s+(medical\s+)?advice|"
    r"informational\s+purposes\s+only|"
    r"seek\s+the\s+advice\s+of\s+your\s+(doctor|physician)|"
    r"medical\s+disclaimer|always\s+consult|"
    r"professional\s+medical\s+advice|"
    r"(talk|speak|check)\s+with\s+(your\s+|their\s+|a\s+|the\s+)?(doctor|physician|healthcare)|"
    r"work\s+with\s+(your\s+|their\s+|a\s+)?healthcare|"
    r"medically\s+reviewed|"
    r"reviewed\s+by\s+a\s+(doctor|physician|medical|licensed)|"
    r"this\s+(article|content|information)\s+is\s+(not|for)\s+(medical|informational)|"
    r"(see|visit|contact)\s+(your\s+)?(doctor|physician|clinician|specialist)|"
    r"not\s+a\s+substitute\s+for\s+(professional\s+)?(medical|clinical)|"
    r"before\s+(starting|taking|using|changing)\s+(any\s+)?(medication|treatment|supplement)|"
    r"(peer[\s-]reviewed|fact[\s-]checked|editorial[\s-]reviewed))",
    re.IGNORECASE,
)

# ── SEO spam URL pattern ───────────────────────────────────────────────────────

_SPAM_URL_RE = re.compile(
    r"(top\d+|best\d+|\d+tips|click(?:bait)?|free.?download|"
    r"secret|miracle|youwontbelieve)",
    re.IGNORECASE,
)

# ── Conspiracy / misinformation URL-slug pattern ───────────────────────────────
# Matches URL-path framing that is unambiguous misinformation language.
# One hit is sufficient - these phrases never appear in legitimate health content URLs.
# Note: _LOW_AUTH_RE covers pattern fragments (miracle, doctors-hate, one-weird-trick);
# this covers multi-word conspiracy constructs that evade those single-token checks.
_CONSPIRACY_URL_RE = re.compile(
    # Multi-word constructs only - single words like "big-pharma" alone are too broad
    # (legitimate analysis pieces also use that term). Require conspiratorial framing.
    r"((?:big[-_]pharma|they|doctors?)[-_]don[t']?[-_]want[-_]you|"
    r"doesn[t']?[-_]want[-_]you[-_]to[-_]know|"
    r"don[t']?[-_]want[-_]you[-_]to[-_]know|"
    r"big[-_]pharma[-_](?:doesn[t']?|don[t']?|hides?|suppress|hate)|"
    r"suppress(?:ed)?[-_](?:truth|cure[-_]for|cure[-_]they)|"
    r"hidden[-_]cure[-_](?:for|they|that)|"
    r"natural[-_]cancer[-_]cure|"
    r"they[-_]hide[-_](?:this|the|it)|"
    r"secret[-_](?:cure|remedy)[-_](?:for|they|that|doctors?))",
    re.IGNORECASE,
)

# ── date parsing ───────────────────────────────────────────────────────────────

_DATE_FMTS = (
    ("%Y-%m-%d",                 10),
    ("%Y/%m/%d",                 10),
    ("%Y-%m-%dT%H:%M:%S",        19),
    ("%Y-%m-%dT%H:%M:%SZ",       20),
    ("%a, %d %b %Y %H:%M:%S %z", 31),
    ("%a, %d %b %Y %H:%M:%S",    25),
    ("%a, %d %b %Y",             16),
    ("%B %d, %Y",                None),
    ("%d %B %Y",                 None),
    ("%b %d, %Y",                None),
    ("%Y %b %d",                 None),
    ("%Y %b",                    None),
    ("%Y",                        4),
)


def _parse_date(s: str) -> datetime | None:
    if not s or s.lower() in CFG.missing_date_sentinels:
        return None
    s = s.strip()
    for fmt, slice_len in _DATE_FMTS:
        candidate = s[:slice_len] if slice_len else s
        try:
            dt = datetime.strptime(candidate, fmt)
            if dt.tzinfo is not None:
                # Convert tz-aware datetime to UTC before stripping tzinfo so that
                # e.g. "14:50 -0400" is treated as 18:50 UTC, not penalised as future.
                dt = dt.astimezone(timezone.utc).replace(tzinfo=None)
            return dt
        except ValueError:
            continue
    return None


def _netloc(url: str) -> str:
    try:
        return urlparse(url).netloc.lower().replace("www.", "")
    except Exception:
        return ""


# _STOP_WORDS imported from project.utils.sanitizer (canonical shared definition)

# Anomaly flags already handled by direct checks - skip in the anomaly loop
_DIRECT_CHECK_PREFIXES = CFG.anomaly_direct_check_prefixes


# ══════════════════════════════════════════════════════════════════════════════
# Core Signal Computers
# ══════════════════════════════════════════════════════════════════════════════

def _domain_matches(domain: str, authority_set: frozenset) -> bool:
    """
    Exact-match or subdomain-match against an authority set.
    Requires a '.' prefix for subdomain matching so 'fake-nih.gov' does NOT
    match 'nih.gov' (endswith without '.' prefix would accept it).
    """
    return any(domain == d or domain.endswith("." + d) for d in authority_set)


def _author_signal(author: str, url: str) -> Signal:
    """
    Author credibility with domain-consistency cross-check.

    If author claims a credible organisation (NIH, Harvard…) but the hosting
    domain is low-authority, confidence is reduced by CFG.author_domain_mismatch_conf_penalty.
    The score is unchanged (we don't know the claim is false) - only the
    confidence changes (we have less evidence it's true).
    """
    domain   = _netloc(url)
    is_high  = _domain_matches(domain, _HIGH_AUTHORITY)
    is_spam  = bool(_LOW_AUTH_RE.search(url))
    no_author = not author or author.lower() in CFG.missing_author_sentinels

    if no_author:
        score = CFG.author_score_anon_high_domain if is_high else CFG.author_score_anon_low_domain
        return Signal(score, CFG.author_score_anon_conf,
                      {"reason": "no_author", "domain_proxy": is_high})

    if "," in author and not _CREDIBLE_ORG_RE.search(author):
        parts = [p.strip() for p in author.split(",") if p.strip()]
        if len(parts) > 1:
            subs  = [_author_signal(p, url) for p in parts]
            avg_s = sum(s.score      for s in subs) / len(subs)
            avg_c = sum(s.confidence for s in subs) / len(subs)
            return Signal(round(avg_s, 4), round(avg_c, 4),
                          {"reason": "multi_author_average", "count": len(parts)})

    if _CREDIBLE_ORG_RE.search(author):
        conf = CFG.author_score_credible_org_conf
        evid: dict = {"reason": "credible_org_match", "author": author}
        if is_spam:
            conf = round(conf - CFG.author_domain_mismatch_conf_penalty, 4)
            evid["warning"] = "domain_mismatch_credible_org_on_low_auth_domain"
        return Signal(CFG.author_score_credible_org, conf, evid)

    score = CFG.author_score_named_high_domain if is_high else CFG.author_score_named_low_domain
    return Signal(score, CFG.author_score_named_conf,
                  {"reason": "named_author", "high_domain": is_high})


_EVIDENCE_RE = re.compile(
    r"\b(study|studies|research|trial|published\s+in|according\s+to|evidence|"
    r"meta-analysis|systematic\s+review|randomized|cohort|clinical|findings|"
    r"showed|demonstrated|found\s+that|researchers|scientists|participants|"
    r"peer.reviewed|journal|doi|pmid|citation|reference)\b",
    re.IGNORECASE,
)


def _citation_signal(source_type: str, citations: int | None, views: int | None,
                     content: str = "") -> Signal:
    if source_type == "pubmed":
        if citations is None:
            return Signal(CFG.citation_unknown_score, CFG.citation_unknown_conf,
                          {"reason": "citations_unknown"})
        score = min(citations / CFG.pubmed_citation_saturation, 1.0)
        return Signal(round(score, 4), CFG.citation_pubmed_conf, {"citations": citations})
    if views and views > 0:
        score = min(math.log10(views) / CFG.view_log_denominator, 1.0)
        return Signal(round(score, 4), CFG.citation_view_conf, {"views": views})
    # Blog: no views/citations available. Use evidence-phrase density as proxy.
    # Counts academic/evidence language ("study", "published in", "RCT", etc.)
    # to distinguish evidence-backed articles from opinion pieces.
    if content:
        hits  = len(_EVIDENCE_RE.findall(content))
        score = round(min(hits / CFG.citation_blog_evidence_saturation, 1.0), 4)
        return Signal(score, CFG.citation_blog_conf,
                      {"reason": "blog_evidence_proxy", "evidence_phrases": hits})
    return Signal(CFG.engagement_missing_score, CFG.engagement_missing_conf,
                  {"reason": "no_engagement_data"})


def _domain_signal(url: str) -> Signal:
    domain = _netloc(url)
    if not domain:
        return Signal(CFG.domain_score_no_url, CFG.domain_score_no_url_conf,
                      {"reason": "no_domain"})
    if _domain_matches(domain, _HIGH_AUTHORITY):
        return Signal(CFG.domain_score_high, CFG.domain_score_high_conf,
                      {"tier": "high", "domain": domain})
    if _domain_matches(domain, _MED_AUTHORITY):
        return Signal(CFG.domain_score_medium, CFG.domain_score_medium_conf,
                      {"tier": "medium", "domain": domain})
    if _LOW_AUTH_RE.search(url):
        return Signal(CFG.domain_score_spam, CFG.domain_score_spam_conf,
                      {"tier": "spam", "pattern": "low_auth_regex"})
    return Signal(CFG.domain_score_unknown, CFG.domain_score_unknown_conf,
                  {"tier": "unknown", "domain": domain})


def _recency_signal(published_date: str, source_type: str) -> Signal:
    """
    Smooth exponential decay: score = ceiling × 2^(−age_days / halflife).
    All parameters from CFG. No step functions - continuous gradient.

    Future-dated content (age_days < 0) is treated as missing-date: the
    timestamp is almost certainly a metadata error and should not receive
    the maximum recency score that a 0-day age would produce.
    """
    dt = _parse_date(published_date)
    if dt is None:
        return Signal(CFG.recency_missing_score, CFG.recency_missing_conf,
                      {"reason": "missing_date", "raw": published_date})
    age_days = (scoring_now() - dt).days
    if age_days < 0:
        return Signal(CFG.recency_missing_score, CFG.recency_missing_conf,
                      {"reason": "future_date", "raw": published_date, "age_days": age_days})
    halflife  = CFG.halflife(source_type)
    score     = max(CFG.recency_floor,
                    round(CFG.recency_ceiling * math.pow(2.0, -age_days / halflife), 4))
    return Signal(score, CFG.recency_conf,
                  {"age_days": age_days, "halflife": halflife, "published": published_date})


def _disclaimer_signal(content: str, source_type: str, url: str = "") -> Signal:
    if source_type == "pubmed":
        return Signal(CFG.disclaimer_pubmed_score, CFG.disclaimer_pubmed_conf,
                      {"reason": "pubmed_peer_reviewed"})
    is_health = bool(_HEALTH_RE.search(content))
    if not is_health:
        return Signal(CFG.disclaimer_neutral_score, CFG.disclaimer_neutral_conf,
                      {"reason": "non_health_content"})
    has_disclaimer = bool(_DISCLAIMER_RE.search(content))
    if has_disclaimer:
        return Signal(CFG.disclaimer_present_score, CFG.disclaimer_present_conf,
                      {"health_content": True, "disclaimer_found": True})
    # No inline disclaimer found. Check domain editorial policy as implicit proxy.
    # High-auth health publishers (Healthline, WebMD, MedicalNewsToday) have
    # mandatory medical review policies - disclaimer lives in footer/sidebar, not body.
    domain = _netloc(url)
    if domain and _domain_matches(domain, _HIGH_AUTHORITY):
        return Signal(CFG.disclaimer_high_auth_implicit_score,
                      CFG.disclaimer_high_auth_implicit_conf,
                      {"health_content": True, "disclaimer_found": False,
                       "implicit_editorial_review": True, "domain": domain})
    if domain and _domain_matches(domain, _MED_AUTHORITY):
        return Signal(CFG.disclaimer_med_auth_implicit_score,
                      CFG.disclaimer_med_auth_implicit_conf,
                      {"health_content": True, "disclaimer_found": False,
                       "implicit_editorial_review": "partial", "domain": domain})
    return Signal(CFG.disclaimer_absent_score, CFG.disclaimer_absent_conf,
                  {"health_content": True, "disclaimer_found": False})


# ══════════════════════════════════════════════════════════════════════════════
# Abuse Penalty (Zipf-adaptive - all parameters from CFG)
# ══════════════════════════════════════════════════════════════════════════════

def _abuse_penalty(
    author: str, url: str, content: str,
    published_date: str, source_type: str,
    anomaly_flags: list[str],
) -> tuple[float, list[str]]:
    """
    Multiplicative penalty ∈ [CFG.abuse_floor, 1.0].

    Zipf-adaptive stuffing: threshold = max(base, zipf_k / ln(vocab_size)).
    All multipliers from CFG - no magic numbers inline.
    Stacking is intentional: multiple abuse vectors compound via multiplication.
    """
    m:       float     = 1.0
    reasons: list[str] = []

    is_health = bool(_HEALTH_RE.search(content))
    no_author = not author or author.lower() in CFG.missing_author_sentinels

    if source_type != "pubmed" and is_health and no_author:
        m       *= CFG.penalty_anon_health
        reasons.append("anonymous_health_content")

    spam_hits = len(_SPAM_URL_RE.findall(url))
    if spam_hits >= CFG.seo_spam_min_hits:
        m       *= CFG.penalty_seo_spam_url
        reasons.append(f"seo_spam_url:{spam_hits}_patterns")

    # URL-slug conspiracy/misinformation framing. One match is sufficient;
    # phrases like "big-pharma-doesnt-want-you-to-know" in a URL path are
    # unambiguous regardless of body content quality or domain authority.
    conspiracy_hits = len(_CONSPIRACY_URL_RE.findall(url))
    if conspiracy_hits >= 1:
        m       *= CFG.penalty_conspiracy_url
        reasons.append(f"conspiracy_url_slug:{conspiracy_hits}_patterns")

    # Zipf-adaptive keyword stuffing detection.
    # threshold = max(base, zipf_k / ln(vocab_size)) - lenient on short docs,
    # tight on long docs where Zipf predicts lower natural top-word frequency.
    words = content.lower().split()
    if source_type != "pubmed" and len(words) >= CFG.stuffing_min_words:
        content_words = [w for w in words if w.isalpha() and w not in _STOP_WORDS]
        if content_words:
            top_word, top_count = Counter(content_words).most_common(1)[0]
            top_freq = top_count / len(content_words)
            vocab_size = len(set(content_words))
            adaptive_threshold = min(
                CFG.stuffing_max_threshold,
                max(CFG.stuffing_base_threshold,
                    CFG.stuffing_zipf_k / math.log(max(vocab_size, 2))),
            )
            if top_freq > adaptive_threshold:
                m       *= CFG.penalty_keyword_stuffing
                reasons.append(
                    f"keyword_stuffing:{top_word}:{top_freq:.3f}"
                    f":threshold:{adaptive_threshold:.3f}"
                )

    if source_type in ("blog", "youtube") and is_health:
        dt = _parse_date(published_date)
        if dt and (scoring_now() - dt).days > CFG.outdated_health_days:
            m       *= CFG.penalty_outdated_health
            reasons.append("outdated_health_content")

    if is_health and _LOW_AUTH_RE.search(url):
        m       *= CFG.penalty_low_authority_health
        reasons.append("low_authority_health")

    for flag in anomaly_flags:
        if any(flag.startswith(p) for p in _DIRECT_CHECK_PREFIXES):
            continue
        m       *= CFG.penalty_per_anomaly
        reasons.append(f"anomaly:{flag}")

    return round(max(m, CFG.abuse_floor), 4), reasons


# ══════════════════════════════════════════════════════════════════════════════
# Source Reliability Priors
# ══════════════════════════════════════════════════════════════════════════════

_PRIORS_FILE   = Path(__file__).resolve().parents[2] / CFG.priors_cache_filename
_priors_data:  dict[str, dict] = {}
_priors_loaded = False
_priors_lock   = threading.Lock()


def _load_priors() -> None:
    global _priors_data, _priors_loaded
    if _priors_loaded:
        return
    _priors_loaded = True
    try:
        if _PRIORS_FILE.exists():
            _priors_data = json.loads(_PRIORS_FILE.read_text())
    except Exception:
        _priors_data = {}


def _get_prior(url: str) -> float | None:
    with _priors_lock:
        _load_priors()
        domain = _netloc(url)
        entry  = _priors_data.get(domain)
        if entry and entry.get("n", 0) >= CFG.prior_min_n:
            return entry["mean_score"]
    return None


def update_prior(url: str, score: float) -> None:
    with _priors_lock:
        _load_priors()
        domain = _netloc(url)
        if not domain:
            return
        entry = _priors_data.setdefault(domain, {"n": 0, "mean_score": CFG.prior_init_score})
        n     = entry["n"]
        entry["mean_score"] = round((entry["mean_score"] * n + score) / (n + 1), 4)
        entry["n"]          = n + 1


def persist_priors() -> None:
    """Atomic write: stage to .tmp then os.replace, so a crash mid-write or a
    second concurrent CLI run cannot leave the priors file truncated/corrupt."""
    with _priors_lock:
        try:
            tmp = _PRIORS_FILE.with_suffix(_PRIORS_FILE.suffix + ".tmp")
            tmp.write_text(json.dumps(_priors_data, indent=2))
            os.replace(tmp, _PRIORS_FILE)
        except Exception:
            pass


# ══════════════════════════════════════════════════════════════════════════════
# Main Entry Point
# ══════════════════════════════════════════════════════════════════════════════

def calculate(
    source_url:     str,
    source_type:    str,
    author:         str,
    published_date: str,
    content:        str,
    citations:      int | None  = None,
    views:          int | None  = None,
    likes:          int | None  = None,
    anomaly_flags:  list[str] | None = None,
    record_dict:    dict | None = None,
) -> dict:
    """
    Compute trust score for one scraped record.

    Parameters
    ----------
    likes       : YouTube like count (for engagement_authenticity signal)
    record_dict : full record dict (for metadata_completeness signal)

    Returns canonical output schema - all fields present regardless of source type.
    """
    if source_type not in WEIGHTS:
        print(f"  ⚠ trust_score: unknown source_type '{source_type}' - defaulting to 'blog'",
              file=sys.stderr)
    source_type = source_type if source_type in WEIGHTS else "blog"
    w     = WEIGHTS[source_type]
    flags = anomaly_flags or []

    signals: dict[str, Signal] = {
        "author_credibility":          _author_signal(author, source_url),
        "citation_count":              _citation_signal(source_type, citations, views, content),
        "domain_authority":            _domain_signal(source_url),
        "recency":                     _recency_signal(published_date, source_type),
        "medical_disclaimer_presence": _disclaimer_signal(content, source_type, source_url),
        "metadata_completeness":       metadata_completeness_signal(record_dict, source_type),
        "content_depth":               content_depth_signal(content),
        "engagement_authenticity":     engagement_authenticity_signal(views, likes, source_type),
        "adversarial_risk":            adversarial_risk_signal(content),
    }

    weight_dict  = w.as_dict()
    # .get(k, 0.0) is defensive: unknown signal keys contribute zero weight
    # rather than raising KeyError, keeping the pipeline alive for schema drift.
    weighted_num = sum(weight_dict.get(k, 0.0) * sig.score * sig.confidence
                       for k, sig in signals.items())
    weighted_den = sum(weight_dict.get(k, 0.0) * sig.confidence
                       for k, sig in signals.items())
    # Neutral fallback if every signal reports zero confidence (pathological input).
    raw_score = (weighted_num / weighted_den) if weighted_den > CFG.signal_weighted_denom_floor else CFG.signal_zero_denom_fallback

    penalty, reasons = _abuse_penalty(
        author, source_url, content, published_date, source_type, flags
    )
    penalised = raw_score * penalty

    prior      = _get_prior(source_url)
    prior_used = None
    if prior is not None:
        penalised  = (1 - CFG.prior_weight) * penalised + CFG.prior_weight * prior
        prior_used = prior

    final = round(min(max(penalised, 0.0), 1.0), 3)

    return {
        "trust_score":   final,
        "sub_scores":    {k: round(v.score,      3) for k, v in signals.items()},
        "confidence":    {k: round(v.confidence, 3) for k, v in signals.items()},
        "evidence":      {k: v.evidence              for k, v in signals.items()},
        "weights":       weight_dict,
        "abuse_penalty": penalty,
        "abuse_reasons": reasons,
        "prior_used":    prior_used,
    }
