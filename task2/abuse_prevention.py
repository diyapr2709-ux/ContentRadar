"""

Compares a "clean" baseline score against an "abusive" version of the same
input, proving the penalty system reduces the score appropriately.

Abuse vectors covered:
  1. Fake / anonymous authors on health content
  2. SEO spam blogs (low domain authority + spammy URL)
  3. Misleading medical content (no disclaimer)
  4. Outdated information (strong recency penalty on health)
  5. Keyword stuffing (single word > 5% density)

Run:  python3 task2/abuse_prevention.py
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task2.trust_score import calculate


def case(num: int, title: str) -> None:
    print(f"\n── Case {num}: {title} " + "─" * (40 - len(title)))


def main() -> None:
    print("=" * 60)
    print("  Abuse Prevention Demonstrations")
    print("=" * 60)

    #  1. Anonymous health content 
    case(1, "Fake / anonymous author on health content")
    clean = calculate(
        source_url="https://medium.com/health",
        source_type="blog", author="Dr. Sarah Chen",
        published_date="2024-06-01",
        content="Diabetes treatment requires medical supervision.",
    )
    abuse = calculate(
        source_url="https://medium.com/health",
        source_type="blog", author="Unknown",
        published_date="2024-06-01",
        content="Diabetes treatment requires medical supervision.",
    )
    print(f"  Named doctor  : {clean['trust_score']}")
    print(f"  Anonymous     : {abuse['trust_score']}  "
          f"(penalty={abuse['abuse_penalty']})")
    assert abuse['trust_score'] < clean['trust_score'], "Anonymous should score lower"

    # 2. SEO spam blog
    case(2, "SEO spam URL pattern")
    clean = calculate(
        source_url="https://towardsdatascience.com/diabetes-ml",
        source_type="blog", author="John Smith",
        published_date="2024-06-01",
        content="An overview of diabetes prediction models.",
    )
    abuse = calculate(
        source_url="https://top10healthtips.blogspot.com/secret-cure-click",
        source_type="blog", author="John Smith",
        published_date="2024-06-01",
        content="An overview of diabetes prediction models.",
    )
    print(f"  Clean URL : {clean['trust_score']}")
    print(f"  Spam URL  : {abuse['trust_score']}  "
          f"(penalty={abuse['abuse_penalty']})")
    assert abuse['trust_score'] < clean['trust_score'], "Spam URL should score lower"

    # 3. Missing medical disclaimer
    case(3, "Health content without medical disclaimer")
    with_disclaimer = calculate(
        source_url="https://medium.com/health",
        source_type="blog", author="Dr. Lee",
        published_date="2024-06-01",
        content=("This article discusses diabetes treatment options. "
                 "This content is for informational purposes only and is "
                 "not medical advice. Always consult a doctor."),
    )
    no_disclaimer = calculate(
        source_url="https://medium.com/health",
        source_type="blog", author="Dr. Lee",
        published_date="2024-06-01",
        content="This article discusses diabetes treatment options.",
    )
    print(f"  With disclaimer : {with_disclaimer['trust_score']}  "
          f"(disclaimer_sub={with_disclaimer['sub_scores']['medical_disclaimer']})")
    print(f"  Without         : {no_disclaimer['trust_score']}  "
          f"(disclaimer_sub={no_disclaimer['sub_scores']['medical_disclaimer']})")
    assert no_disclaimer['trust_score'] < with_disclaimer['trust_score'], \
        "Missing disclaimer should score lower"

    # ── 4. Outdated health content ────────────────────────────────
    case(4, "Outdated health information")
    recent = calculate(
        source_url="https://medium.com/health",
        source_type="blog", author="Dr. Khan",
        published_date="2024-06-01",
        content="Diabetes treatment guidelines.",
    )
    old = calculate(
        source_url="https://medium.com/health",
        source_type="blog", author="Dr. Khan",
        published_date="2018-01-01",
        content="Diabetes treatment guidelines.",
    )
    print(f"  Recent (2024) : {recent['trust_score']}")
    print(f"  Old (2018)    : {old['trust_score']}  "
          f"(recency_sub={old['sub_scores']['recency']})")
    assert old['trust_score'] < recent['trust_score'], "Older content should score lower"

    # ── 5. Keyword stuffing ───────────────────────────────────────
    case(5, "Keyword stuffing (>5% density)")
    natural = calculate(
        source_url="https://towardsdatascience.com/post",
        source_type="blog", author="Author",
        published_date="2024-06-01",
        content="This article covers software engineering and cloud infrastructure topics. " * 20,
    )
    stuffed = calculate(
        source_url="https://towardsdatascience.com/post",
        source_type="blog", author="Author",
        published_date="2024-06-01",
        content="diabetes diabetes diabetes diabetes diabetes " * 30,
    )
    print(f"  Natural text  : {natural['trust_score']}  "
          f"(penalty={natural['abuse_penalty']})")
    print(f"  Keyword stuff : {stuffed['trust_score']}  "
          f"(penalty={stuffed['abuse_penalty']})")
    assert stuffed['abuse_penalty'] < 1.0, "Stuffed content should be penalised"
    assert stuffed['trust_score'] <= natural['trust_score'], \
        "Stuffed content should score lower or equal"

    print("\n" + "=" * 60)
    print("  ✓ All abuse vectors penalised correctly.")
    print("=" * 60)


if __name__ == "__main__":
    main()