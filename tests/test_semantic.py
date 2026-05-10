"""
Semantic relevance gate. Only runs when sentence-transformers is installed.
Auto-skips otherwise so CI doesn't have to install the 800MB torch stack.
"""
from __future__ import annotations

from project.utils.semantic import is_available, semantic_score

try:
    import pytest
    _HAS_PYTEST = True
except ImportError:
    _HAS_PYTEST = False


def _skip_if_unavailable():
    if not is_available():
        if _HAS_PYTEST:
            pytest.skip("sentence-transformers not installed")
        else:
            return True
    return False


def test_semantic_score_returns_none_when_unavailable():
    """Contract: when the library is absent, semantic_score returns None and
    the caller falls back to keyword-only. Always-runnable test."""
    if is_available():
        # If installed, this returns a float for valid input
        s = semantic_score("AI safety", "AI alignment research is critical")
        assert isinstance(s, float)
    else:
        # If not installed, this must be None (graceful degradation)
        s = semantic_score("AI safety", "AI alignment research is critical")
        assert s is None


def test_semantic_relevant_pair_scores_higher_than_unrelated():
    if _skip_if_unavailable():
        return
    s_relevant = semantic_score("AI safety",
                                 "AI safety research focuses on alignment of language models")
    s_unrelated = semantic_score("AI safety",
                                  "Balcony solar panels reduce household electricity bills")
    assert s_relevant is not None and s_unrelated is not None
    assert s_relevant > s_unrelated, \
        f"semantic gate not discriminating: relevant={s_relevant:.3f} unrelated={s_unrelated:.3f}"


def test_semantic_score_handles_empty_inputs():
    """Empty topic or text must not crash."""
    assert semantic_score("", "some text") is None
    assert semantic_score("topic", "") is None
