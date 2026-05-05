"""
main.py

Pipeline entry point.
  Task 1 : Scrape 3 blogs + 2 YouTube + 1 PubMed for a given topic.
  Task 2 : Score every scraped record, write to task2/output/scored_data/

Usage:
  python main.py --topic "diabetes"
  python main.py  # prompts for topic
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from task1.scraper.blog_scraper    import run as run_blogs
from task1.scraper.youtube_scraper import run as run_youtube
from task1.scraper.pubmed_scraper  import run as run_pubmed
from task2.apply_scores            import apply_scoring

TASK1_OUT = "task1/output/scraped_data"
TASK2_OUT = "task2/output/scored_data"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Topic-driven multi-source scraper + trust scoring pipeline"
    )
    parser.add_argument(
        "--topic", default=None,
        help="Topic keyword (e.g. diabetes, obesity, cancer). "
             "If omitted, you'll be prompted.",
    )
    args = parser.parse_args()

    # Prompt if topic not provided via CLI
    topic = args.topic
    if not topic or not topic.strip():
        topic = input("Enter topic to scrape: ").strip()

    # Reject empty input
    if not topic:
        print("✗ No topic provided. Exiting.")
        sys.exit(1)

    print("=" * 60)
    print(f"  Pipeline — topic='{topic}'")
    print("=" * 60)

    print("\n[Task 1] Scraping sources...")
    run_blogs(topic, TASK1_OUT)
    run_youtube(topic, TASK1_OUT)
    run_pubmed(topic, TASK1_OUT)

    print("\n[Task 2] Applying trust scores...")
    apply_scoring(TASK1_OUT, TASK2_OUT)

    print("\n" + "=" * 60)
    print("  ✓ Done.")
    print(f"     Task 1 raw    → {TASK1_OUT}/")
    print(f"     Task 2 scored → {TASK2_OUT}/")
    print("=" * 60)


if __name__ == "__main__":
    main()