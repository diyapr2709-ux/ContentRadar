"""
project/scoring/signals.py - Extended signal computers for the trust engine.

Provides four new signals beyond the core five in trust_score.py:
  metadata_completeness  - structural completeness of the record
  content_depth          - word count + type-token ratio + Gini diversity
  engagement_authenticity- YouTube like/view ratio vs. log-normal expected
  adversarial_risk       - Unicode attacks, prompt injection, n-gram repetition

All thresholds and parameters come from project/scoring/config.py (CFG).
No magic numbers exist in this file.
"""
from __future__ import annotations

import math
import re
from collections import Counter
from typing import NamedTuple
from urllib.parse import unquote

from project.scoring.config import CFG, ScoringConfig
from project.utils.sanitizer import STOP_WORDS as _STOP_WORDS


class Signal(NamedTuple):
    score:      float
    confidence: float
    evidence:   dict


# ══════════════════════════════════════════════════════════════════════════════
# Shared utility: Gini coefficient
# ══════════════════════════════════════════════════════════════════════════════

def _gini_coefficient(counter: Counter) -> float:
    """
    Lorenz-curve Gini coefficient for a word frequency Counter.

    G = 0.0 → perfect uniformity (all words equally frequent)
    G = 1.0 → maximum inequality (one word has all frequency mass)

    Natural English content words: G ~ 0.60-0.75 (Brown Corpus analysis).
    Keyword-stuffed text: G > 0.85.

    Algorithm (O(n log n)):
      sorted_freqs = sorted(counter.values())  ascending
      G = (2 * Σ (i+1)*f_i) / (n * Σ f_i) − (n+1)/n
    Returns 0.0 for empty or single-element counters.
    """
    if not counter:
        return 0.0
    vals = sorted(counter.values())
    n    = len(vals)
    if n == 1:
        # Single unique element: all mass on one token → maximum concentration.
        # Returning 0.0 here would be mathematically consistent with "no spread
        # among distinct values", but for our use case (lexical diversity) a
        # single unique word repeated N times is maximally concentrated, not
        # maximally uniform. Return 1.0 so gini_score = 1 - 1 = 0.0 (no diversity).
        return 1.0
    total  = sum(vals)
    if total == 0:
        return 0.0
    numer  = sum((i + 1) * v for i, v in enumerate(vals))
    return (2.0 * numer) / (n * total) - (n + 1) / n


# ══════════════════════════════════════════════════════════════════════════════
# Signal: Metadata Completeness
# ══════════════════════════════════════════════════════════════════════════════

_DOI_RE  = re.compile(CFG.doi_pattern)
_PMID_RE = re.compile(CFG.pmid_pattern)


def _field_present(record: dict, field: str) -> bool:
    """
    A field is present if it is non-None, non-empty string, and non-zero int.
    DOI and PMID additionally pass format validation.
    """
    val = record.get(field)
    if val is None:
        return False
    if isinstance(val, str):
        if not val.strip():
            return False
        if field == "doi":
            return bool(_DOI_RE.match(val.strip()))
        if field == "pmid":
            return bool(_PMID_RE.match(val.strip()))
        return True
    if isinstance(val, (int, float)):
        return val != 0
    return bool(val)


def metadata_completeness_signal(
    record: dict | None,
    source_type: str,
    cfg: ScoringConfig = CFG,
) -> Signal:
    """
    Score = fields_present / fields_required for the source type.
    Confidence = 1.0 - deterministic computation, zero uncertainty.

    Edge: record=None → neutral (0.50, 0.30) with low confidence.
    Edge: source_type unknown → falls back to blog fields.
    """
    if record is None:
        return Signal(cfg.signal_neutral_score, cfg.signal_neutral_conf_med, {"reason": "no_record_dict"})

    required = cfg.metadata_fields(source_type)
    present  = sum(1 for f in required if _field_present(record, f))
    score    = round(present / max(len(required), 1), 4)
    missing  = [f for f in required if not _field_present(record, f)]

    return Signal(score, cfg.metadata_missing_conf,
                  {"present": present, "required": len(required),
                   "missing_fields": missing})


# ══════════════════════════════════════════════════════════════════════════════
# Signal: Content Depth
# ══════════════════════════════════════════════════════════════════════════════

# _STOP_WORDS imported from project.utils.sanitizer (canonical shared definition)


def _content_depth_confidence(word_count: int, cfg: ScoringConfig) -> float:
    """
    Confidence scales with word count via smooth linear interpolation between CFG thresholds.

    Avoids the cliff effect where a 99-word doc (conf=0.40) and 100-word doc (conf=0.65)
    get radically different confidence values for a 1-word difference.
    """
    thresholds = cfg.content_depth_conf_thresholds
    for i, (floor, conf) in enumerate(thresholds):
        if word_count >= floor:
            if i > 0:
                prev_floor, prev_conf = thresholds[i - 1]
                t = min(1.0, (word_count - floor) / max(prev_floor - floor, 1))
                return round(conf + t * (prev_conf - conf), 4)
            return conf
    return cfg.signal_depth_min_conf_fallback


def content_depth_signal(
    content: str,
    cfg: ScoringConfig = CFG,
) -> Signal:
    """
    Composite depth score from three orthogonal sub-components:
      word_count_score = min(word_count / saturation, 1.0)
      ttr_score        = unique_words / total_words  (type-token ratio)
      gini_score       = 1.0 - Gini(content_word_frequencies)
                         (inverted: high Gini = low diversity = low score)

    depth = wc_weight × wc_score + ttr_weight × ttr + gini_weight × (1-G)

    Confidence degrades with word count via CFG.content_depth_conf_thresholds.
    Empty / thin content → confidence=0.10 (not crash, not zero contribution).
    """
    if not content:
        return Signal(0.0, cfg.signal_neutral_conf_low, {"reason": "empty_content", "word_count": 0})

    words = content.lower().split()
    wc    = len(words)
    conf  = _content_depth_confidence(wc, cfg)

    if wc < cfg.content_depth_min_words:
        return Signal(0.0, conf, {"reason": "thin_content", "word_count": wc})

    wc_score = min(wc / cfg.content_word_count_saturation, 1.0)
    ttr      = len(set(words)) / wc

    content_words = [w for w in words if w.isalpha() and w not in _STOP_WORDS]
    if content_words:
        freq_counter = Counter(content_words)
        gini         = _gini_coefficient(freq_counter)
        gini_score   = max(0.0, 1.0 - gini)
    else:
        gini      = 0.0
        gini_score = cfg.signal_gini_no_content_words

    depth = (cfg.content_wc_weight  * wc_score +
             cfg.content_ttr_weight * ttr      +
             cfg.content_gini_weight * gini_score)

    return Signal(
        round(min(depth, 1.0), 4), conf,
        {"word_count": wc, "ttr": round(ttr, 3),
         "gini": round(gini, 3), "wc_score": round(wc_score, 3)},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Signal: Engagement Authenticity
# ══════════════════════════════════════════════════════════════════════════════

def engagement_authenticity_signal(
    views:       int | None,
    likes:       int | None,
    source_type: str,
    cfg:         ScoringConfig = CFG,
) -> Signal:
    """
    YouTube-only signal. For other types, returns Signal(0.50, 0.0, {}) -
    zero confidence means zero effective weight in the confidence-weighted sum.

    Like/view ratio r = likes / views.
    Expected: ln(r) ~ Normal(cfg.engagement_like_view_mu, cfg.engagement_like_view_sigma)
    [1] Borghol et al. (2012) - log-normal fits organic YouTube engagement ratios.

    z_score = (ln(r) − mu) / sigma
    score   = max(0.0, 1.0 − |z_score| / outlier_z)  (smoothly degrades)

    Manipulation flags:
      likes=0 with views > min_views → suspicious (zero-like buying)
      |z_score| > outlier_z          → engagement outlier

    Edge cases:
      views=None, likes=None, views < min_views → Signal(0.50, 0.10, {})
      views=0                                   → Signal(0.50, 0.10, {})
    """
    if source_type != "youtube":
        return Signal(cfg.signal_neutral_score, 0.0, {"reason": "not_youtube"})

    if not views or views < cfg.engagement_min_views or likes is None:
        return Signal(cfg.signal_neutral_score, cfg.signal_engagement_low_conf,
                      {"reason": "insufficient_engagement_data", "views": views, "likes": likes})

    ratio = max(likes, cfg.engagement_zero_likes_floor) / views
    ln_r  = math.log(ratio)
    z     = (ln_r - cfg.engagement_like_view_mu) / cfg.engagement_like_view_sigma

    score = max(0.0, 1.0 - abs(z) / cfg.engagement_outlier_z)
    conf  = cfg.signal_engagement_outlier_conf if abs(z) > cfg.engagement_outlier_z else cfg.signal_engagement_normal_conf

    flags = []
    if likes == 0 and views > cfg.engagement_min_views:
        flags.append("zero_likes_high_views")
    if abs(z) > cfg.engagement_outlier_z:
        flags.append(f"engagement_outlier:z={z:.2f}")

    return Signal(
        round(min(score, 1.0), 4), conf,
        {"views": views, "likes": likes,
         "like_view_ratio": round(ratio, 5),
         "z_score": round(z, 3),
         "manipulation_flags": flags},
    )


# ══════════════════════════════════════════════════════════════════════════════
# Signal: Adversarial Risk
# ══════════════════════════════════════════════════════════════════════════════

_INJECTION_RES = [
    re.compile(p, re.IGNORECASE)
    for p in CFG.adversarial_injection_patterns
]
_BIDI_CHARS   = set(CFG.adversarial_bidi_codepoints)
_ZW_CHARS     = set(CFG.adversarial_zwchar_codepoints)
_NGRAM_SIZE   = CFG.adversarial_ngram_size


def _ngram_repetition_ratio(text: str, n: int) -> float:
    """
    Fraction of n-grams that are repeated.
    repetition_ratio = 1 - (unique_ngrams / total_ngrams)
    Returns 0.0 for text too short to form n-grams.
    """
    words  = text.lower().split()
    if len(words) < n:
        return 0.0
    ngrams = [tuple(words[i:i + n]) for i in range(len(words) - n + 1)]
    if not ngrams:
        return 0.0
    return 1.0 - len(set(ngrams)) / len(ngrams)


def adversarial_risk_signal(
    content:     str,
    raw_content: str | None = None,   # pre-sanitization text (catches stripped bidi chars)
    cfg:         ScoringConfig = CFG,
) -> Signal:
    """
    Four-component adversarial risk assessment.

    1. Bidi/zero-width Unicode attacks (CVE-2021-42574):
       Scans raw_content (pre-sanitizer) if available, else content.
       bidi_score = max(0.0, 1.0 - bidi_count / cfg.adversarial_bidi_saturation)

    2. Prompt injection patterns:
       injection_score = max(0.0, 1.0 - hit_count / cfg.adversarial_injection_saturation)

    3. N-gram repetition (spam/spinning):
       repetition_ratio = 1 - unique_ngrams / total_ngrams
       rep_score = 1.0 if below threshold, else max(0.0, 1.0 - repetition_ratio)

    4. Final score = mean(bidi_score, injection_score, rep_score)
       Low score → high adversarial risk → low trust contribution.

    Confidence:
      empty content        → Signal(1.0, 0.0, {})   zero weight in fusion
      thin (<50 chars)     → Signal(0.90, 0.30, {})
      else                 → 0.85
    """
    scan_text = raw_content if raw_content is not None else content

    if not content:
        return Signal(1.0, 0.0, {"reason": "empty_content"})

    if len(content) < cfg.adversarial_thin_content_chars:
        return Signal(cfg.signal_adv_thin_score, cfg.signal_adv_thin_conf, {"reason": "thin_content"})

    # 1. Bidi / zero-width attack chars
    bidi_count = sum(1 for ch in scan_text if ord(ch) in _BIDI_CHARS or ord(ch) in _ZW_CHARS)
    bidi_score = max(0.0, 1.0 - bidi_count / cfg.adversarial_bidi_saturation)

    # 2. Prompt injection - scan decoded + raw content to catch URL-encoded payloads
    # and payloads stripped during sanitization (e.g. <script>ignore...</script>).
    decoded_content  = unquote(content)
    decoded_raw      = unquote(raw_content) if raw_content is not None else decoded_content
    injection_hits   = max(
        sum(1 for rx in _INJECTION_RES if rx.search(decoded_content)),
        sum(1 for rx in _INJECTION_RES if rx.search(decoded_raw)),
    )
    injection_score = max(0.0, 1.0 - injection_hits / cfg.adversarial_injection_saturation)

    # 3. N-gram repetition
    rep_ratio  = _ngram_repetition_ratio(content, cfg.adversarial_ngram_size)
    if rep_ratio > cfg.adversarial_repetition_threshold:
        rep_score = max(0.0, 1.0 - rep_ratio)
    else:
        rep_score = 1.0

    sub_scores = (bidi_score, injection_score, rep_score)
    score = sum(sub_scores) / len(sub_scores)

    return Signal(
        round(min(score, 1.0), 4), cfg.signal_adv_full_conf,
        {"bidi_chars_found":       bidi_count,
         "injection_patterns_hit": injection_hits,
         "repetition_ratio":       round(rep_ratio, 3),
         "ngram_size":             cfg.adversarial_ngram_size},
    )
