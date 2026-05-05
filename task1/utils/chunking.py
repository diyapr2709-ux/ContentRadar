"""
task1/utils/chunking.py — paragraph-aware chunker.
O(N) single pass. Handles "Long Articles" assignment edge case.
"""

from __future__ import annotations


def chunk_text(text: str, max_words: int = 150) -> list[str]:
    if not text:
        return []
    paragraphs = [p for p in text.split("\n\n") if p.strip()]
    if not paragraphs:
        return []
    chunks, buf, buf_wc = [], [], 0
    for para in paragraphs:
        words = para.split()
        wc = len(words)
        if wc == 0:
            continue
        if buf_wc + wc <= max_words:
            buf.append(para)
            buf_wc += wc
        else:
            if buf:
                chunks.append(" ".join(buf))
                buf, buf_wc = [], 0
            if wc > max_words:
                for i in range(0, wc, max_words):
                    chunks.append(" ".join(words[i:i + max_words]))
            else:
                buf.append(para)
                buf_wc = wc
    if buf:
        chunks.append(" ".join(buf))
    return chunks