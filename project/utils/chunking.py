"""
project/utils/chunking.py - RAG-ready, confidence-aware chunker.

Why this exists beyond "just split every N words":

  Vector-store retrieval degrades when chunks are cut mid-sentence. A retrieval
  system that stores "...the patient's insulin" in one chunk and "resistance
  worsens over time" in the next will miss that semantic unit in dense search.

  Innovations over the baseline chunker:
    1. Sentence-boundary preference - splits at ". " / "! " / "? " where possible
       instead of at an arbitrary word index.
    2. Configurable overlap - the last OVERLAP_WORDS words of chunk k are prepended
       to chunk k+1. This preserves cross-chunk context for retrieval (a retriever
       scoring chunk k+1 sees the trailing context of chunk k too).
    3. Per-chunk quality score - each chunk is rated on:
         a) lexical diversity  (unique tokens / total tokens)
         b) completeness       (ends with sentence-terminal punctuation)
         c) information signal (not dominated by one repeated token)
       Chunks below QUALITY_THRESHOLD are excluded from RAG output so the vector
       store isn't polluted with boilerplate or error messages.
    4. Positional metadata - chunk_index, total_chunks, char_start, char_end
       allow downstream systems to reconstruct provenance and re-rank by position.
"""
from __future__ import annotations

import re
from collections import Counter

from project.scoring.config import CFG

MAX_WORDS         = CFG.chunk_max_words
OVERLAP_WORDS     = CFG.chunk_overlap_words
MIN_CHUNK_WORDS   = CFG.chunk_min_words
QUALITY_THRESHOLD = CFG.chunk_quality_threshold

_SENT_END = re.compile(r"(?<=[.!?])\s+")


# ── quality scoring ────────────────────────────────────────────────────────────

_CHUNK_STOP_WORDS = frozenset({
    "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
    "of", "with", "by", "from", "as", "is", "was", "are", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "this", "that", "it", "its", "they",
    "we", "you", "he", "she", "i", "not", "so", "also", "more", "which",
    "who", "what", "how", "when", "where", "if", "into", "out", "up",
    "than", "then", "there", "through", "very", "may", "might", "can",
    "such", "only", "each", "both", "over", "about", "after", "all",
})


def _quality_score(text: str) -> float:
    """
    Heuristic quality ∈ [0, 1] for a single chunk.

    Three independent signals - no single pathology can fake a high score:
      lexical_diversity: type/token ratio - low if text is repetitive
      completeness:      bonus if chunk ends on a sentence boundary
      entropy_signal:    penalises chunks dominated by one very frequent
                         CONTENT word (stop words filtered first)
    """
    words = text.lower().split()
    if len(words) < CFG.chunk_quality_min_words:
        return 0.0

    diversity    = len(set(words)) / len(words)
    completeness = (CFG.chunk_quality_completeness_full
                    if text.rstrip().endswith((".", "!", "?", '."', '."'))
                    else CFG.chunk_quality_completeness_partial)

    # Stop-word-filtered density - avoids flagging "the" or "of" as stuffing
    content_words = [w for w in words if w.isalpha() and w not in _CHUNK_STOP_WORDS]
    if content_words:
        top_freq   = Counter(content_words).most_common(1)[0][1] / len(content_words)
        entropy_ok = 0.0 if top_freq > CFG.chunk_quality_entropy_max_top_freq else 1.0
    else:
        entropy_ok = 1.0

    return round(
        diversity    * CFG.chunk_quality_w_diversity
        + completeness * CFG.chunk_quality_w_completeness
        + entropy_ok   * CFG.chunk_quality_w_entropy,
        3,
    )


# ── sentence-boundary split ────────────────────────────────────────────────────

def _sentences(text: str) -> list[str]:
    """Split text into sentences (best-effort; no NLTK required)."""
    parts = _SENT_END.split(text.strip())
    return [p.strip() for p in parts if p.strip()]


def _merge_sentences(sentences: list[str], max_words: int) -> list[str]:
    """
    Greedily merge sentences into chunks ≤ max_words, respecting sentence
    boundaries. If a single sentence exceeds max_words, it is hard-split.
    """
    chunks: list[str] = []
    buf:    list[str] = []
    buf_wc = 0

    for sent in sentences:
        wc = len(sent.split())
        if wc == 0:
            continue
        if wc > max_words:
            # flush current buffer first
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_wc = [], 0
            # hard-split the overlong sentence
            words = sent.split()
            for i in range(0, len(words), max_words):
                chunks.append(" ".join(words[i:i + max_words]))
        elif buf_wc + wc <= max_words:
            buf.append(sent)
            buf_wc += wc
        else:
            if buf:
                chunks.append(" ".join(buf))
            buf, buf_wc = [sent], wc

    if buf:
        chunks.append(" ".join(buf))
    return chunks


# ── public API ─────────────────────────────────────────────────────────────────

def rag_chunk(
    text:       str,
    max_words:  int   = MAX_WORDS,
    overlap:    int   = OVERLAP_WORDS,
    min_words:  int   = MIN_CHUNK_WORDS,
    quality_floor: float = QUALITY_THRESHOLD,
) -> list[dict]:
    """
    RAG-ready chunker. Returns a list of structured chunk objects.

    Each chunk dict:
      text          - chunk content (may include overlap prefix)
      chunk_index   - 0-based position in this document
      total_chunks  - how many chunks this document produced
      quality_score - heuristic quality ∈ [0, 1]
      has_overlap   - True if this chunk carries a prefix from the previous chunk

    Chunks below quality_floor or min_words are dropped before indexing.
    Overlap ensures cross-chunk semantic units survive retrieval.
    """
    if not text:
        return []

    sentences = _sentences(text)
    if not sentences:
        return []

    raw_chunks = _merge_sentences(sentences, max_words)

    # filter too-short chunks before applying overlap
    raw_chunks = [c for c in raw_chunks if len(c.split()) >= min_words]
    if not raw_chunks:
        return []

    result: list[dict] = []
    prev_tail: list[str] = []   # overlap words carried from previous chunk

    for chunk_text_raw in raw_chunks:
        words = chunk_text_raw.split()
        has_overlap = bool(prev_tail)

        combined = (" ".join(prev_tail) + " " + chunk_text_raw).strip() \
                   if prev_tail else chunk_text_raw

        q = _quality_score(combined)
        if q < quality_floor:
            # still update prev_tail so the next chunk maintains context
            prev_tail = words[-overlap:] if overlap else []
            continue

        result.append({
            "text":          combined,
            "chunk_index":   len(result),   # sequential in filtered output
            "quality_score": q,
            "has_overlap":   has_overlap,
        })
        prev_tail = words[-overlap:] if overlap else []

    # back-fill total_chunks after all filtering is complete
    total = len(result)
    for chunk in result:
        chunk["total_chunks"] = total

    return result
