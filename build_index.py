"""
build_index.py - Post-run index builder and output auditor.

Scans project/output/topics/, consolidates records by source type,
validates every record against the canonical schema, and writes:

  project/output/index.json              ← master reviewer entry point
  project/output/blogs_all_topics.json   ← all blog records across topics
  project/output/youtube_all_topics.json ← all YouTube records across topics
  project/output/pubmed_all_topics.json  ← all PubMed records across topics

Usage:
  python build_index.py                        # default: project/output
  python build_index.py --output project/output
  python build_index.py --quiet                # suppress per-topic lines
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from project.utils.schema import (
    SCHEMA_VERSION, SOURCE_TYPES, TYPE_TO_FILE, validate_record, SchemaError,
)
from project.scoring.config import CFG


def build_index(output_root: Path, quiet: bool = False) -> dict:
    """
    Scan all topic subdirectories, validate records, write consolidated files.
    Returns the index manifest dict.
    """
    topics_dir = output_root / "topics"
    if not topics_dir.exists():
        raise FileNotFoundError(f"topics directory not found: {topics_dir}")

    buckets:     dict[str, list[dict]] = {st: [] for st in SOURCE_TYPES}
    index_topics: list[dict]           = []
    schema_errors: list[tuple[str, str, str]] = []   # (topic, url, error)

    topic_dirs = sorted(d for d in topics_dir.iterdir() if d.is_dir())
    if not topic_dirs:
        raise FileNotFoundError(f"no topic subdirectories found under {topics_dir}")

    for topic_dir in topic_dirs:
        manifest_path = topic_dir / "scraped_data.json"
        if not manifest_path.exists():
            continue
        try:
            data = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError) as e:
            print(f"  ⚠ Cannot read {manifest_path}: {e} - skipping topic", file=sys.stderr)
            continue
        meta    = data.get("meta", {})
        records = data.get("records", [])

        counts: dict[str, int] = {st: 0 for st in SOURCE_TYPES}
        for rec in records:
            st = rec.get("source_type", "")
            if st not in buckets:
                continue
            # Validate before including in consolidated output
            try:
                validate_record(rec)
            except SchemaError as e:
                schema_errors.append((topic_dir.name, rec.get("source_url", "?"), str(e)))
                continue
            buckets[st].append(rec)
            counts[st] += 1

        valid_total = sum(counts.values())
        index_topics.append({
            "topic":        topic_dir.name,
            "total":        valid_total,
            "avg_trust":    meta.get("avg_trust"),
            "trust_tiers":  meta.get("trust_tiers", {}),
            "sources":      counts,
            "generated_at": meta.get("generated_at"),
            "schema_version": meta.get("schema_version", SCHEMA_VERSION),
            "files": {
                "combined": f"topics/{topic_dir.name}/scraped_data.json",
                "blogs":    f"topics/{topic_dir.name}/blogs.json",
                "youtube":  f"topics/{topic_dir.name}/youtube.json",
                "pubmed":   f"topics/{topic_dir.name}/pubmed.json",
            },
        })

        if not quiet:
            b, y, p = counts["blog"], counts["youtube"], counts["pubmed"]
            total   = b + y + p
            avg     = meta.get("avg_trust", 0)
            print(f"  ✓ {topic_dir.name:<18} blog={b} yt={y} pubmed={p} "
                  f"total={total} avg_trust={avg}")

    # Warn about schema violations but don't abort
    if schema_errors:
        print(f"\n  ⚠ {len(schema_errors)} schema violations found (excluded from consolidated):")
        for topic, url, err in schema_errors:
            print(f"      [{topic}] {url[-CFG.display_url_tail_chars:]}: {err}")

    # Sort each bucket by trust_score desc, then source_url for deterministic tie-breaking
    for st in SOURCE_TYPES:
        buckets[st].sort(
            key=lambda r: (-r.get("trust_score", 0.0), r.get("source_url", ""))
        )

    # Write consolidated files
    totals: dict[str, int] = {}
    for st in SOURCE_TYPES:
        records_out = buckets[st]
        fname       = TYPE_TO_FILE[st] + "_all_topics.json"
        if not records_out:
            totals[st] = 0
            continue

        real_scores = [r["trust_score"] for r in records_out if r.get("trust_score", 0) > 0]
        avg_trust   = round(sum(real_scores) / len(real_scores), 3) if real_scores else 0.0

        tiers: dict[str, int] = {}
        for r in records_out:
            t = r.get("trust_tier", "?")
            tiers[t] = tiers.get(t, 0) + 1

        payload = {
            "meta": {
                "type":           st,
                "schema_version": SCHEMA_VERSION,
                "total_records":  len(records_out),
                "avg_trust":      avg_trust,
                "trust_tiers":    tiers,
                "topics_covered": [t["topic"] for t in index_topics],
                "sorted_by":      "trust_score_desc",
            },
            "records": records_out,
        }
        dst = output_root / fname
        dst.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        totals[st] = len(records_out)
        print(f"  → {fname}  {len(records_out)} records  avg_trust={avg_trust}")

    # Master index
    grand_total = sum(totals.values())
    index = {
        "meta": {
            "pipeline":        "ContentRadar v2",
            "schema_version":  SCHEMA_VERSION,
            "total_topics":    len(index_topics),
            "total_records":   grand_total,
            "breakdown":       totals,
            "consolidated_files": {
                st: f"{TYPE_TO_FILE[st]}_all_topics.json"
                for st in SOURCE_TYPES
            },
            "note": (
                "Start with index.json to navigate. "
                "Each topic folder has scraped_data.json (combined + manifest) "
                "plus blogs.json / youtube.json / pubmed.json. "
                "All records carry record_id (stable hash), ingested_at, schema_version."
            ),
        },
        "topics": index_topics,
    }
    idx_path = output_root / "index.json"
    idx_path.write_text(json.dumps(index, indent=2, ensure_ascii=False))
    print(f"  → index.json  {len(index_topics)} topics  {grand_total} total records")
    return index


def _audit(output_root: Path) -> None:
    """Print a human-readable audit table for reviewer verification."""
    idx_path = output_root / "index.json"
    if not idx_path.exists():
        print("✗ index.json not found - run build_index first", file=sys.stderr)
        return

    idx = json.loads(idx_path.read_text(encoding="utf-8"))
    b   = idx["meta"]["breakdown"]

    print("\n" + "=" * CFG.display_audit_bar_width)
    print("  ContentRadar v2 - Output Audit")
    print("=" * CFG.display_audit_bar_width)
    print(f"  {'Topic':<20} {'Blog':>4} {'YT':>4} {'PubMed':>6} {'Total':>6}  {'AvgTrust':>8}  Tiers")
    print("  " + "-" * CFG.display_audit_inner_width)

    for t in idx["topics"]:
        srcs   = t.get("sources", {})
        bl, yt, pm = srcs.get("blog", 0), srcs.get("youtube", 0), srcs.get("pubmed", 0)
        total  = bl + yt + pm
        avg    = t.get("avg_trust", 0) or 0
        tiers  = t.get("trust_tiers", {})
        tier_s = " ".join(f"{k[0]}={v}" for k, v in sorted(tiers.items()))
        print(f"  ✓ {t['topic']:<20} {bl:>4} {yt:>4} {pm:>6} {total:>6}  {avg:>8.3f}  {tier_s}")

    print("  " + "-" * CFG.display_audit_inner_width)
    print(f"  {'TOTAL':<20} {b.get('blog',0):>4} {b.get('youtube',0):>4} "
          f"{b.get('pubmed',0):>6} {idx['meta']['total_records']:>6}")
    print("=" * CFG.display_audit_bar_width)

    # Check schema_version consistency
    versions = {t.get("schema_version") for t in idx["topics"]}
    if versions == {SCHEMA_VERSION}:
        print(f"\n  schema_version: {SCHEMA_VERSION} - all topics consistent ✓")
    else:
        print(f"\n  ⚠ Mixed schema versions: {versions}")

    print(f"\n  Output layout:")
    print(f"  {output_root}/")
    print(f"  ├── index.json                   ← start here")
    for st in SOURCE_TYPES:
        fname = TYPE_TO_FILE[st] + "_all_topics.json"
        count = b.get(st, 0)
        print(f"  ├── {fname:<30} ← {count} records")
    print(f"  └── topics/")
    print(f"      └── <topic>/")
    print(f"          ├── scraped_data.json      ← combined manifest")
    print(f"          ├── blogs.json")
    print(f"          ├── youtube.json")
    print(f"          └── pubmed.json")
    print()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="ContentRadar - build index and consolidated output files"
    )
    parser.add_argument("--output", default="project/output",
                        help="Root output directory (default: project/output)")
    parser.add_argument("--quiet", action="store_true",
                        help="Suppress per-topic lines")
    args = parser.parse_args()

    output_root = Path(args.output)
    print(f"\nBuilding index from {output_root}/topics/ …\n")
    try:
        build_index(output_root, quiet=args.quiet)
    except FileNotFoundError as e:
        print(f"✗ {e}", file=sys.stderr)
        sys.exit(1)
    _audit(output_root)


if __name__ == "__main__":
    main()
