"""
project/utils/semantic.py - Optional semantic relevance via sentence-transformers.

Mirrors the langdetect import pattern: if `sentence_transformers` is installed,
expose `is_semantically_relevant(topic, text)` that encodes both and compares
cosine similarity against `CFG.semantic_min_similarity`. If the package is not
installed, `semantic_score()` returns `None` and the caller falls back to
keyword-only relevance.

Model is lazy-loaded on first call. Default model (`all-MiniLM-L6-v2`) is
~80MB and downloads from HuggingFace on first use. The model object is cached
at module level so subsequent calls are fast.

Why this design vs forced dependency: sentence-transformers + torch is ~800MB.
Forcing it on every user would balloon the install and slow CI runs. Treating
it as an optional capability mirrors how the project handles langdetect.
"""
from __future__ import annotations

from project.scoring.config import CFG

try:
    from sentence_transformers import SentenceTransformer
    _HAS_SBERT = True
except ImportError:
    _HAS_SBERT = False

_model = None   # lazy-loaded on first call


def is_available() -> bool:
    """True iff sentence-transformers is importable. Cheap, no model load."""
    return _HAS_SBERT


def _get_model():
    """Lazy-load the model. Returns None if sentence-transformers absent."""
    global _model
    if not _HAS_SBERT:
        return None
    if _model is None:
        try:
            _model = SentenceTransformer(CFG.semantic_model_name)
        except Exception:
            return None
    return _model


def semantic_score(topic: str, text: str) -> float | None:
    """
    Cosine similarity between `topic` and `text` in shared embedding space.
    Returns None when the package isn't installed or model load fails -
    callers should fall through to keyword-only relevance in that case.

    Text is truncated to `CFG.semantic_truncate_chars` to bound encoding time.
    """
    if not topic or not text:
        return None
    model = _get_model()
    if model is None:
        return None
    try:
        truncated = text[: CFG.semantic_truncate_chars]
        embeddings = model.encode([topic, truncated], convert_to_numpy=True,
                                  show_progress_bar=False)
        topic_vec, text_vec = embeddings[0], embeddings[1]
        # Cosine similarity = dot(a, b) / (||a|| * ||b||)
        denom = (topic_vec @ topic_vec) ** 0.5 * (text_vec @ text_vec) ** 0.5
        if denom == 0:
            return None
        return float(topic_vec @ text_vec / denom)
    except Exception:
        return None
