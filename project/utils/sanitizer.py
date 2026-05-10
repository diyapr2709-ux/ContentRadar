"""
project/utils/sanitizer.py

Single security/normalisation layer. All scraped strings pass through here
before any downstream module touches them.

Threats neutralised:
  NUL / control chars → str.translate() strips in one O(N) C-level pass
  Unicode bidi spoofing → RLO/LRO/PDF codepoints stripped (CVE-2021-42574)
  Whitespace bombs → collapsed to single space/newline
  Field length bombs → hard truncated at MAX_FIELD_LEN
  URL scheme abuse → only http/https accepted; javascript:/data:/file: rejected
"""
from __future__ import annotations

import re
from urllib.parse import urlparse

from project.scoring.config import CFG

try:
    from langdetect import detect as _ld_detect
    _HAS_LANGDETECT = True
except ImportError:
    _HAS_LANGDETECT = False

_BANNED = (
    list(range(0x00, 0x09)) + [0x0B, 0x0C] + list(range(0x0E, 0x20))
    + [0x200E, 0x200F, 0x202A, 0x202B, 0x202C, 0x202D, 0x202E,
       0x2066, 0x2067, 0x2068, 0x2069]
)
_TABLE = str.maketrans("", "", "".join(chr(c) for c in _BANNED))

_MULTI_SPACE   = re.compile(r"[ \t]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")

MAX_FIELD_LEN = CFG.sanitizer_max_field_len


def clean_text(value: object, max_len: int = MAX_FIELD_LEN) -> str:
    if value is None:
        return ""
    if not isinstance(value, str):
        value = str(value)
    value = value.translate(_TABLE)
    value = _MULTI_SPACE.sub(" ", value)
    value = _MULTI_NEWLINE.sub("\n\n", value)
    return value.strip()[:max_len]


# Canonical English stop words - imported by trust_score, signals, and anomaly modules.
# Superset of all three previous local definitions. Contraction fragments ("s", "t",
# "don", "isn") are included to filter apostrophe-split tokens without affecting
# natural content-word density calculations.
STOP_WORDS: frozenset = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "shall", "can", "not",
    "this", "that", "these", "those", "it", "its", "they", "them", "their",
    "we", "our", "you", "your", "he", "she", "his", "her", "i", "my",
    "no", "nor", "so", "yet", "both", "either", "neither", "about", "above",
    "after", "all", "also", "am", "any", "because", "before", "between",
    "during", "each", "few", "more", "most", "other", "over", "same",
    "than", "then", "there", "through", "under", "until", "up", "very",
    "which", "while", "who", "whom", "whose", "what", "how", "when",
    "where", "if", "into", "out", "such", "only", "own", "too",
    "s", "t", "don", "isn", "wasn", "aren", "won",
})

_TLD_REGION: dict[str, str] = {
    "co.uk": "UK",   "ac.uk": "UK",   "org.uk": "UK",  "gov.uk": "UK",
    "com.au": "AU",  "org.au": "AU",  "net.au": "AU",
    "ca": "CA",      "co.ca": "CA",
    "de": "DE",      "fr": "FR",      "it": "IT",      "es": "ES",
    "nl": "NL",      "be": "BE",      "se": "SE",      "no": "NO",
    "ch": "CH",      "at": "AT",      "dk": "DK",      "fi": "FI",
    "in": "IN",      "co.in": "IN",
    "cn": "CN",      "com.cn": "CN",
    "jp": "JP",      "co.jp": "JP",
    "br": "BR",      "com.br": "BR",
    "gov": "US",     "mil": "US",     "edu": "US",
}

# Pre-sorted longest-first so multi-part TLDs (co.uk) match before their
# single-part suffix (uk). Computed once at import time, not per-call.
_TLD_REGION_SORTED: list[tuple[str, str]] = sorted(
    _TLD_REGION.items(), key=lambda x: len(x[0]), reverse=True
)


def infer_region(url: str) -> str:
    """
    Infer country/region from URL TLD. Returns ISO-3166 alpha-2, 'US' for
    .gov/.mil/.edu, or 'Unknown' for indeterminate TLDs (.com, .org, .net).
    Multi-part TLDs (co.uk) checked before single-part (uk).
    """
    try:
        host = urlparse(url).netloc.lower()
        if host.startswith("www."):
            host = host[len("www."):]
        for tld, region in _TLD_REGION_SORTED:
            if host.endswith("." + tld):
                return region
        return "Unknown"
    except Exception:
        return "Unknown"


def detect_language(text: str) -> str:
    """Detect BCP-47 language code. Falls back to 'en' on any failure."""
    if not _HAS_LANGDETECT or not text or len(text) < CFG.sanitizer_langdetect_min_chars:
        return "en"
    try:
        return _ld_detect(text[:CFG.sanitizer_langdetect_max_chars])
    except Exception:
        return "en"


def validate_url(url: object) -> bool:
    if not isinstance(url, str) or not url.strip():
        return False
    try:
        p = urlparse(url.strip())
        return p.scheme in ("http", "https") and bool(p.netloc) and len(url) < CFG.sanitizer_max_url_len
    except Exception:
        return False
