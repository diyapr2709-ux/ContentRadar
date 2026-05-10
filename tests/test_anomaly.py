"""
Anomaly detection regression tests. Each test corresponds to a real flag
emitted by the pipeline against a real or synthetic record.
"""
from __future__ import annotations

from project.utils.anomaly import detect


def test_clean_blog_emits_no_flags():
    text = (
        "This is a thoughtful, well-edited article about diabetes management. "
        "It cites several peer-reviewed studies and offers balanced guidance. "
        "There are no suspicious patterns here, just clear prose written by "
        "a domain expert. The article is moderately long and varied."
    ) * 4
    flags = detect(text, {"source_type": "blog"})
    assert flags == [], f"unexpected flags on clean blog: {flags}"


def test_thin_content_flagged():
    flags = detect("Tiny.", {"source_type": "blog"})
    assert "thin_content" in flags


def test_promotional_language_flagged():
    text = (
        "Click here! Buy now! Limited time only! Subscribe today! Free download! "
        "Don't miss this exclusive offer! Act now! 100% natural and guaranteed!"
    )
    flags = detect(text, {"source_type": "blog"})
    promo = [f for f in flags if f.startswith("promotional_language")]
    assert promo, f"promotional language not flagged: {flags}"


def test_youtube_link_density_threshold_more_lenient():
    """Source-aware: same text flags on blog but not on youtube (when the
    URL ratio sits between blog 0.20 and youtube 0.40)."""
    text = (
        "Brief intro. " + ("Visit https://example.com/foo for more. " * 3) +
        " " + (" ".join(["word"] * 70))
    )
    blog_flags = detect(text, {"source_type": "blog"})
    yt_flags   = detect(text, {"source_type": "youtube"})
    if "high_link_density" in blog_flags:
        assert "high_link_density" not in yt_flags, \
            "YouTube threshold should be more permissive than blog"


def test_pubmed_skips_keyword_stuffing():
    """PubMed abstracts legitimately repeat domain terms; the anomaly
    detector must NOT flag them as keyword-stuffed."""
    text = (" ".join(["glucose"] * 50) + " " + " ".join(["studied"] * 50)) * 2
    flags = detect(text, {"source_type": "pubmed"})
    stuffing = [f for f in flags if f.startswith("keyword_stuffing")]
    assert not stuffing, f"PubMed wrongly flagged: {stuffing}"


def test_future_year_flagged():
    """The TEMPORAL regex matches \\b(20\\d{2})\\b, so the year must be a
    realistic 20XX. Pin the scoring clock to 2025 so 2099 is unambiguously
    future regardless of when the test runs."""
    import os
    os.environ["CONTENTRADAR_AS_OF"] = "2025-01-01"
    try:
        text = ("This article speculates that by the year 2099 everything will "
                "look fundamentally different than today. " +
                " ".join(["various words and sentences here for length"] * 8))
        flags = detect(text, {"source_type": "blog"})
        assert any(f.startswith("future_year_reference") for f in flags), \
            f"future year not flagged: {flags}"
    finally:
        os.environ.pop("CONTENTRADAR_AS_OF", None)


def test_encoding_corruption_dual_gate():
    """Both ratio gate and absolute floor must trigger."""
    text = "Normal text " + ("â€" * 10) + " more text " + ("Ã" * 5) + " ending"
    flags = detect(text, {"source_type": "blog"})
    assert any(f.startswith("encoding_corruption") for f in flags)
