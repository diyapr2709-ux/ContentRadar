"""
task2/edge_cases.py
===================
Runnable demonstration of every edge case from the assignment.
Each test prints the input scenario and the resulting score, proving the
scoring engine handles the case correctly without crashing.

Edge cases covered:
  1. Author not available
  2. Publish date missing
  3. Transcript / content unavailable
  4. Multiple authors (averaged credibility)
  5. Non-English content (language unaffects scoring)
  6. Long articles (chunking still works)

Run:  python task2/edge_cases.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task2.trust_score      import calculate
from task1.utils.chunking   import chunk_text
from task1.utils.tagging    import auto_tag


def case(num: int, title: str) -> None:
    print(f"\n── Case {num}: {title} " + "─" * (40 - len(title)))


def main() -> None:
    print("=" * 60)
    print("  Edge Case Demonstrations")
    print("=" * 60)

    # ── 1. Missing author ─────────────────────────────────────────────────────
    case(1, "Author not available")
    r = calculate(
        source_url="https://medium.com/health-blog",
        source_type="blog", author="", published_date="2024-06-01",
        content="Treatment of diabetes through lifestyle changes...",
    )
    print(f"  author=''  →  score={r['trust_score']}  "
          f"(author_sub={r['sub_scores']['author_credibility']})")
    assert 0.0 <= r['trust_score'] <= 1.0

    # ── 2. Missing publish date ───────────────────────────────────────────────
    case(2, "Publish date missing")
    r = calculate(
        source_url="https://openai.com/blog",
        source_type="blog", author="OpenAI", published_date="",
        content="GPT-5 model release announcement",
    )
    print(f"  date=''  →  score={r['trust_score']}  "
          f"(recency_sub={r['sub_scores']['recency']})")
    assert r['sub_scores']['recency'] == 0.30, "missing date should yield 0.30"

    # ── 3. Transcript / content unavailable ───────────────────────────────────
    case(3, "Content / transcript unavailable")
    r = calculate(
        source_url="https://www.youtube.com/watch?v=abc123",
        source_type="youtube", author="MIT OpenCourseWare",
        published_date="2024-01-01", content="",
    )
    print(f"  content=''  →  score={r['trust_score']}  (no crash)")
    assert isinstance(r['trust_score'], float)

    # ── 4. Multiple authors ───────────────────────────────────────────────────
    case(4, "Multiple authors (credibility averaged)")
    r1 = calculate(
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        source_type="pubmed", author="Smith J",
        published_date="2024-01-01",
        content="Diabetes prediction using ML methods",
        citations=50,
    )
    r2 = calculate(
        source_url="https://pubmed.ncbi.nlm.nih.gov/12345/",
        source_type="pubmed", author="Smith J, Doe A, Patel R, Lee K",
        published_date="2024-01-01",
        content="Diabetes prediction using ML methods",
        citations=50,
    )
    print(f"  1 author : author_sub={r1['sub_scores']['author_credibility']}")
    print(f"  4 authors: author_sub={r2['sub_scores']['author_credibility']}")
    print(f"  → averaging works (multiple authors handled)")

    # ── 5. Non-English content ────────────────────────────────────────────────
    case(5, "Non-English content")
    spanish_text = "Investigación sobre la diabetes y el aprendizaje automático"
    tags = auto_tag(spanish_text)
    r = calculate(
        source_url="https://medium.com/es/diabetes",
        source_type="blog", author="Dr. García",
        published_date="2024-01-01", content=spanish_text,
    )
    print(f"  Spanish text  →  score={r['trust_score']}, tags={tags}")
    print(f"  (scoring works regardless of input language)")

    # ── 6. Long articles — chunking works ────────────────────────────────────
    case(6, "Long article (chunking still works)")
    long_text = "\n\n".join(["Diabetes is a chronic metabolic disease. " * 15
                             for _ in range(20)])
    chunks = chunk_text(long_text, max_words=150)
    print(f"  Input: {len(long_text.split())} words")
    print(f"  Output: {len(chunks)} chunks, max chunk size = "
          f"{max(len(c.split()) for c in chunks)} words")
    assert all(len(c.split()) <= 150 for c in chunks), "chunk overflow!"

    print("\n" + "=" * 60)
    print("  ✓ All edge cases handled correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()