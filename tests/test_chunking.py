"""Tests for the RAG chunker."""
from __future__ import annotations

from project.scoring.config import CFG
from project.utils.chunking import rag_chunk


def test_empty_input_returns_empty():
    assert rag_chunk("") == []


def test_short_input_below_min_words_returns_empty():
    # Below CFG.chunk_min_words → filtered out
    text = "Just a few words here."
    assert len(text.split()) < CFG.chunk_min_words
    assert rag_chunk(text) == []


def test_single_chunk_for_short_paragraph():
    text = " ".join(["lorem"] * 30) + ". " + " ".join(["ipsum"] * 30) + "."
    chunks = rag_chunk(text)
    assert chunks
    assert chunks[0]["total_chunks"] == len(chunks)
    assert "text" in chunks[0]
    assert "quality_score" in chunks[0]


def test_chunks_respect_max_words():
    text = ". ".join(" ".join(["word"] * 30) for _ in range(20))
    chunks = rag_chunk(text)
    for c in chunks:
        wc = len(c["text"].split())
        # overlap can push slightly over, so allow some headroom
        assert wc <= CFG.chunk_max_words + CFG.chunk_overlap_words + 5


def test_overlap_marker_set_on_subsequent_chunks():
    text = ". ".join(" ".join(["word"] * 50) for _ in range(20))
    chunks = rag_chunk(text)
    if len(chunks) > 1:
        # at least one chunk after the first must carry overlap context
        assert any(c.get("has_overlap") for c in chunks[1:])


def test_quality_score_in_unit_interval():
    text = (". ".join(" ".join(["lorem", "ipsum", "dolor", "sit"] * 10)
                       for _ in range(5)))
    chunks = rag_chunk(text)
    for c in chunks:
        assert 0.0 <= c["quality_score"] <= 1.0
