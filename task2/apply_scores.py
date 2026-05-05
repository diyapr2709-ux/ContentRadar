"""
task2/apply_scores.py
=====================
Reads Task 1 raw JSON output (trust_score = "") and applies the
trust_score.calculate() engine to fill in the score for every record.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from task2.trust_score import calculate

INPUT_FILES = ("blogs.json", "youtube.json", "pubmed.json")


def _score_record(record: dict) -> dict:
    """Apply trust scoring to one scraped record. Returns a NEW dict."""
    content = "\n\n".join(record.get("content_chunks", []))

    result = calculate(
        source_url     = record.get("source_url", ""),
        source_type    = record.get("source_type", "blog"),
        author         = record.get("author", "Unknown"),
        published_date = record.get("published_date", "Unknown"),
        content        = content,
        citations      = None,
        download_count = None,
    )

    scored = dict(record)
    scored["trust_score"] = result["trust_score"]
    return scored


def apply_scoring(input_dir: str, output_dir: str) -> None:
    """Score every record in every Task 1 output file."""
    in_path  = Path(input_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for fname in INPUT_FILES:
        src = in_path / fname
        if not src.exists():
            print(f"  ⚠ {src} not found — skipping")
            continue

        try:
            records = json.loads(src.read_text())
        except json.JSONDecodeError as e:
            print(f"  ✗ {fname} invalid JSON: {e}")
            continue

        scored = [_score_record(r) for r in records]
        dst    = out_path / fname
        dst.write_text(json.dumps(scored, indent=2, ensure_ascii=False))
        print(f"  ✓ {fname}: {len(scored)} records scored → {dst}")


if __name__ == "__main__":
    apply_scoring(
        input_dir  = "task1/output/scraped_data",
        output_dir = "task2/output/scored_data",
    )