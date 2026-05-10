"""
project/utils/anomaly.py - Heuristic content anomaly detector.

Flags suspicious content patterns before trust scoring. Each check returns
a string flag on a positive hit, which the trust engine accumulates into
an abuse penalty multiplier.

Key design invariant: anomaly flags must not duplicate signals that
trust_score._abuse_penalty() already checks directly. Only flag anomalies
that are UNIQUE to this detector (entropy, link density, promotional language,
encoding corruption, templated paragraphs). Keyword stuffing is flagged here
ONLY if it passes the stop-word filter - the trust engine's direct stuffing
check uses the same stop-word logic, so there is no double-counting.

Min-word guard (MIN_WORDS_FOR_STUFFING = 80):
  With short RSS snippets (~26 words), any common word appearing twice hits
  7%+ density - far below the statistical threshold for intentional stuffing.
  Stuffing is only meaningful in longer documents where density is deliberate.
"""
from __future__ import annotations

import math
import re
from collections import Counter

from project.scoring.config import CFG, scoring_now
from project.utils.sanitizer import STOP_WORDS as _STOP_WORDS, detect_language as _detect_language

_PROMO_RE = re.compile(
    r"\b(click here|buy now|subscribe|limited time|act now|order today|"
    r"free download|sign up now|don'?t miss|exclusive offer|"
    r"100\s*%\s*(natural|guaranteed|free)|as seen on tv|"
    r"shocking truth|doctors?\s+hate|miracle cure|secret remedy|"
    r"guaranteed results|risk[\s-]free|money[\s-]back guarantee|"
    r"lose \d+ pounds|detox your|flush your|cleanse your|"
    r"big pharma|they don'?t want you to know|"
    r"one weird trick|this one (thing|hack|secret))\b",
    re.IGNORECASE,
)

_URL_RE     = re.compile(r"https?://\S+")
_SHOUTING_RE = re.compile(r"\b[A-Z]{4,}\b")
_TEMPORAL_RE = re.compile(r"\b(20\d{2})\b")

# _STOP_WORDS imported from project.utils.sanitizer (canonical shared definition)

# All thresholds from CFG - no magic numbers here
MIN_WORDS_FOR_STUFFING  = CFG.stuffing_min_words
MIN_WORDS_FOR_ENTROPY   = CFG.min_words_for_entropy
ENTROPY_LOW_THRESHOLD   = CFG.entropy_low_threshold
PROMO_HIT_THRESHOLD     = CFG.promo_hit_threshold
LINK_DENSITY_THRESHOLD  = CFG.link_density_threshold
CAPS_DENSITY_THRESHOLD  = CFG.caps_density_threshold
TEMPLATE_CV_THRESHOLD   = CFG.template_cv_threshold
STUFFING_DENSITY_FLOOR  = CFG.stuffing_base_threshold   # Zipf-adaptive floor


def _shannon_entropy(text: str) -> float:
    words = text.lower().split()
    if len(words) < MIN_WORDS_FOR_ENTROPY:
        return CFG.anomaly_entropy_skip_value   # sentinel: bypass entropy check on thin content
    freq  = Counter(words)
    total = len(words)
    return -sum((c / total) * math.log2(c / total) for c in freq.values())


def detect(content: str, record: dict) -> list[str]:
    """
    Run all anomaly detectors. Returns a list of flag strings (empty = clean).

    Flags are named consistently so the trust engine can route them correctly
    (e.g. trust_score.py skips keyword_stuffing: flags since it checks
    directly, but processes low_entropy: and promotional_language: itself).
    """
    flags: list[str] = []
    _content = content or ""

    # Lightweight checks applied even on thin content - promotional language
    # and encoding corruption are meaningful signals regardless of length.
    is_thin = not _content or len(_content) < CFG.anomaly_thin_content_chars
    if is_thin:
        flags.append("thin_content")
        promo_hits = len(_PROMO_RE.findall(_content))
        if promo_hits >= PROMO_HIT_THRESHOLD:
            flags.append(f"promotional_language:{promo_hits}_hits")
        corrupt = _content.count("â€") + _content.count("Ã")
        if corrupt > 0:
            flags.append(f"encoding_corruption:{corrupt}_chars")
        return flags

    words = _content.lower().split()
    n     = len(words)

    # 1. Shannon entropy - catches spun / auto-generated articles
    H = _shannon_entropy(_content)
    if H < ENTROPY_LOW_THRESHOLD:
        flags.append(f"low_entropy:{H:.2f}")

    # 2. Keyword stuffing - Zipf-adaptive threshold, consistent with trust engine.
    # threshold = max(base_floor, zipf_k / ln(vocab_size))
    # Lenient on short docs (small vocabulary → naturally high top-word frequency).
    # Skip for pubmed: peer-reviewed abstracts legitimately repeat domain terms.
    is_pubmed = record.get("source_type") == "pubmed"
    if n >= MIN_WORDS_FOR_STUFFING and not is_pubmed:
        content_words = [w for w in words if w.isalpha() and w not in _STOP_WORDS]
        if content_words:
            top_word, top_count = Counter(content_words).most_common(1)[0]
            density    = top_count / len(content_words)
            vocab_size = len(set(content_words))
            adaptive_threshold = min(
                CFG.stuffing_max_threshold,
                max(STUFFING_DENSITY_FLOOR,
                    CFG.stuffing_zipf_k / math.log(max(vocab_size, 2))),
            )
            if density > adaptive_threshold:
                flags.append(f"keyword_stuffing:{top_word}:{density:.3f}")

    # 3. Promotional language - density-adaptive threshold.
    # Long articles need proportionally more hits to flag: max(base, word_count // divisor).
    # Prevents a 5000-word article with 3 footer promo phrases from matching a
    # 100-word clickbait snippet that packs all 3 into its body.
    promo_hits = len(_PROMO_RE.findall(_content))
    promo_threshold = max(PROMO_HIT_THRESHOLD, n // CFG.anomaly_promo_density_divisor)
    if promo_hits >= promo_threshold:
        flags.append(f"promotional_language:{promo_hits}_hits")

    # 4. Link density - content is predominantly URLs (link farm).
    # Threshold is source-aware: YouTube descriptions legitimately carry many
    # URLs (sponsors, socials, chapters, affiliate links), so the 5-topic
    # audit showed 70% false-positive rate against blog 0%. We relax for
    # source_type == "youtube" only.
    url_chars = sum(len(m) for m in _URL_RE.findall(_content))
    link_threshold = (CFG.link_density_threshold_youtube
                      if record.get("source_type") == "youtube"
                      else LINK_DENSITY_THRESHOLD)
    if url_chars / max(len(_content), 1) > link_threshold:
        flags.append("high_link_density")

    # 5. Excessive capitalisation
    shouting = _SHOUTING_RE.findall(_content)
    if len(shouting) / max(n, 1) > CAPS_DENSITY_THRESHOLD:
        flags.append("excessive_caps")

    # 6. Future-year references (temporal inconsistency)
    current_year = scoring_now().year
    future_years = [y for y in _TEMPORAL_RE.findall(_content)
                    if int(y) > current_year]
    if future_years:
        flags.append(f"future_year_reference:{future_years[0]}")

    # 7. Encoding corruption - dual gate: ratio AND absolute floor.
    # The absolute floor catches 100 corrupt chars in a 5000-char article
    # that would slip under the 2% ratio threshold.
    corrupt = _content.count("â€") + _content.count("Ã")
    if (corrupt > CFG.anomaly_corruption_absolute_floor
            or corrupt / max(len(_content), 1) > CFG.anomaly_corruption_ratio):
        flags.append(f"encoding_corruption:{corrupt}_chars")

    # 8. Suspiciously uniform paragraph lengths (templated content)
    paras = [p.strip() for p in _content.split("\n\n") if p.strip()]
    if len(paras) >= CFG.anomaly_template_min_paras:
        lengths = [len(p.split()) for p in paras]
        mean    = sum(lengths) / len(lengths)
        if mean > CFG.anomaly_template_min_mean_words:
            std = math.sqrt(sum((x - mean) ** 2 for x in lengths) / len(lengths))
            if (std / mean) < TEMPLATE_CV_THRESHOLD:
                flags.append("templated_paragraphs")

    return flags


# ── transcript quality detection ───────────────────────────────────────────────

_NOISE_TOKEN_RE = re.compile(r"\[[^\]]{1,30}\]")   # [MUSIC], [APPLAUSE], [INAUDIBLE]…


def detect_transcript_quality(transcript: str) -> list[str]:
    """
    Transcript-specific anomaly detection. Only called when has_transcript=True.
    Returns flag strings (empty = clean). All thresholds from CFG.
    """
    if not transcript or len(transcript) < CFG.anomaly_thin_content_chars:
        return []

    flags: list[str] = []
    words = transcript.split()

    # 1. Noise token ratio - [MUSIC], [APPLAUSE] etc.
    noise_tokens = _NOISE_TOKEN_RE.findall(transcript)
    noise_ratio  = len(noise_tokens) / max(len(words), 1)
    if noise_ratio > CFG.transcript_noise_threshold:
        flags.append(f"transcript_high_noise:{noise_ratio:.2f}")

    # 2. Sentence-level repetition (ASR timeout artifacts)
    sentences = [s.strip().lower() for s in re.split(r"[.!?]+", transcript) if s.strip()]
    if len(sentences) >= CFG.anomaly_min_sentences_for_rep:
        repeated = sum(1 for i in range(1, len(sentences)) if sentences[i] == sentences[i-1])
        rep_frac = repeated / max(len(sentences) - 1, 1)
        if rep_frac > CFG.transcript_repetition_threshold:
            flags.append(f"transcript_repetition:{rep_frac:.2f}")

    # 3. Language consistency via sliding windows
    try:
        size  = CFG.transcript_lang_window_size
        n_win = CFG.transcript_lang_windows
        step  = max(1, (len(transcript) - size) // max(n_win - 1, 1))
        langs = [
            _detect_language(transcript[i * step: i * step + size])
            for i in range(n_win)
            if len(transcript[i * step: i * step + size]) >= CFG.transcript_lang_window_min_chars
        ]
        if langs:
            primary = Counter(langs).most_common(1)[0][0]
            diff    = sum(1 for l in langs if l != primary) / len(langs)
            if diff > CFG.transcript_lang_inconsistency_threshold:
                flags.append(f"transcript_language_inconsistency:{diff:.2f}")
    except Exception:
        pass

    return flags
