"""
Sanitizer unit tests. Covers clean_text, validate_url, infer_region, detect_language.
"""
from __future__ import annotations

from project.utils.sanitizer import (
    clean_text, validate_url, infer_region, detect_language, STOP_WORDS,
)


def test_clean_text_strips_control_chars():
    s = "Hello\x00World\x07Tab\x09Newline\nrest"
    out = clean_text(s)
    assert "\x00" not in out
    assert "\x07" not in out
    # Tab (0x09) is in the banned range, should be removed
    assert "\x09" not in out


def test_clean_text_strips_bidi_chars():
    """CVE-2021-42574 mitigation — RLO/LRO/PDF codepoints must not survive."""
    s = "Hello‮World"
    assert "‮" not in clean_text(s)


def test_clean_text_collapses_whitespace():
    assert clean_text("a    b") == "a b"
    assert clean_text("para1\n\n\n\npara2") == "para1\n\npara2"


def test_clean_text_truncates_at_max_len():
    out = clean_text("x" * 100_000, max_len=50)
    assert len(out) == 50


def test_clean_text_handles_none_and_non_string():
    assert clean_text(None) == ""
    assert clean_text(123) == "123"


def test_validate_url_accepts_http_https():
    assert validate_url("https://example.com/path")
    assert validate_url("http://example.com")


def test_validate_url_rejects_dangerous_schemes():
    assert not validate_url("javascript:alert(1)")
    assert not validate_url("data:text/html,<script>")
    assert not validate_url("file:///etc/passwd")


def test_validate_url_rejects_empty_and_non_string():
    assert not validate_url("")
    assert not validate_url(None)
    assert not validate_url(123)


def test_infer_region_tld_basic():
    assert infer_region("https://example.co.uk/path") == "UK"
    assert infer_region("https://example.cdc.gov/page") == "US"


def test_infer_region_unknown_for_generic_tld():
    assert infer_region("https://example.com") == "Unknown"


def test_detect_language_falls_back_to_en_on_short():
    """detect_language must return 'en' for inputs shorter than the min char floor."""
    assert detect_language("hi") == "en"
    assert detect_language("") == "en"


def test_stop_words_includes_common_english():
    for w in ("the", "a", "is", "of", "and"):
        assert w in STOP_WORDS
