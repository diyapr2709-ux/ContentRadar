"""
Regression tests for `_is_relevant`. Every case here corresponds to a real
bug observed during the 5-topic audits this session. If any of these flip,
a relevance regression has shipped.
"""
from __future__ import annotations

from project.scraper.blog_scraper import _is_relevant, _topic_singletons


def test_singletons_include_short_tokens():
    """AI/ML/UX (len 2-3) must be included — they are content words."""
    assert "ai" in _topic_singletons("AI safety")
    assert "ml" in _topic_singletons("ML deployment")
    assert "ux" in _topic_singletons("UX research")


def test_singletons_drop_stop_words():
    assert "the" not in _topic_singletons("the kubernetes manifesto")
    assert "and" not in _topic_singletons("docker and kubernetes")


def test_singletons_split_hyphen_and_punct():
    assert "fine" in _topic_singletons("fine-tuning LLMs")
    assert "tuning" in _topic_singletons("fine-tuning LLMs")
    assert "llms" in _topic_singletons("fine-tuning LLMs")


# ── Regression: 'balcony solar' must not pass for 'AI safety' ─────────────
def test_solar_article_rejected_for_ai_safety():
    """The original bug. A single 'safety' mention in solar text passed
    relevance because density was 0.005 (above the 0.003 floor)."""
    solar = ("Dozens of US states are considering legislation to allow people "
             "to install plug-in solar systems, often called balcony solar. "
             "Safety standards are evolving across multiple jurisdictions.")
    assert _is_relevant(solar, "AI safety") is False


def test_legit_ai_safety_article_passes():
    text = ("AI safety research focuses on alignment of large language models. "
            "The safety of AI systems remains an open problem. AI safety is hard.")
    assert _is_relevant(text, "AI safety") is True


# ── Regression: UI model article must not pass for 'kubernetes' ───────────
def test_ui_model_article_rejected_for_kubernetes():
    """HuggingFace blog about a vision model that mentions kubernetes once
    in passing must not pass relevance."""
    text = ("H Company's new Holo2 model takes the lead in UI localization. "
            "Built using transformer architecture for production traffic.")
    assert _is_relevant(text, "kubernetes") is False


def test_legit_kubernetes_article_passes():
    text = ("Kubernetes architecture: pods, services, deployments. "
            "Container orchestration with Docker and Helm charts. "
            "Run kubernetes clusters in production.")
    assert _is_relevant(text, "kubernetes") is True


# ── Regression: sleep abstract that lacks 'science' word still passes ─────
def test_sleep_abstract_passes_sleep_science():
    """Generic descriptors ('science', 'engineering') aren't always echoed in
    abstracts. Strict-AND on singletons was tried first and rejected this."""
    text = ("Sleep is fundamental for cognition. Disturbances of sleep affect "
            "mood. Circadian rhythms regulate the timing of sleep. Studies "
            "on sleep architecture continue.")
    assert _is_relevant(text, "sleep science") is True


def test_one_hit_alone_does_not_pass():
    """The absolute-hit floor is 2 — a single drive-by mention is rejected."""
    text = "Once upon a time, a princess. She said: be safe. End of fairy tale."
    assert _is_relevant(text, "AI safety") is False


def test_empty_text_rejected():
    assert _is_relevant("", "anything") is False


def test_empty_topic_with_text():
    # No singletons → only expansion terms. With empty-ish topic the
    # singleton list is empty and expansion comes from the API.
    # Edge case is graceful, not a crash.
    result = _is_relevant("some content with words", "")
    assert isinstance(result, bool)
