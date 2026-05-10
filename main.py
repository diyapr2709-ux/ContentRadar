"""
main.py - ContentRadar pipeline entry point.

Architecture
────────────
  INPUT (topic) →
    [concurrent] blog_scraper ─┐
                 youtube_scraper─┼─ asyncio.gather (asyncio.to_thread)
                 pubmed_scraper ─┘
    [sequential per record] anomaly_detect → trust_score → format →
  OUTPUT → project/output/{blogs,youtube,pubmed}.json + scraped_data.json

Concurrency model:
  asyncio.to_thread runs each blocking scraper in the OS thread pool.
  Network I/O releases the GIL, so true parallel execution occurs without
  aiohttp. Wall-clock ingestion time: max(latencies) not sum(latencies).

Output format (per record) - canonical fields:
  source_url, source_type, author, published_date, language, region,
  topic_tags, trust_score, trust_tier, content_chunks

  Scoring breakdown (interpretability layer):
  sub_scores, confidence, evidence, weights,
  abuse_penalty, abuse_reasons, anomaly_flags, prior_used

Usage:
  python main.py                                          # interactive prompt (comma-separate multiple)
  python main.py --topics "diabetes"                      # single topic → topics/diabetes/
  python main.py --topics "diabetes, RAG, climate change" # multiple topics, one folder each
  python main.py --topics "cancer" --sequential           # disable concurrent scraping
  python main.py --topics "rag" --output /tmp/out         # custom base output dir
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Callable

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from project.utils.anomaly       import detect as detect_anomalies, detect_transcript_quality
from project.utils.sanitizer     import infer_region
from project.utils.schema        import (
    SCHEMA_VERSION, SOURCE_TYPES, TYPE_TO_FILE,
    make_record_id, now_utc, validate_record, SchemaError,
)
from project.scoring.trust_score import (
    calculate as trust_calculate,
    update_prior,
    persist_priors,
)
from project.scoring.config import CFG

# TYPE_TO_FILE and SOURCE_TYPES imported from schema - single canonical definition

# Human-readable trust tiers - boundaries from CFG, no inline literals
def _trust_tier(score: float) -> str:
    if score >= CFG.trust_tier_authoritative: return "AUTHORITATIVE"
    if score >= CFG.trust_tier_credible:      return "CREDIBLE"
    if score >= CFG.trust_tier_uncertain:     return "UNCERTAIN"
    return "UNRELIABLE"


# ── async scraper wrapper ──────────────────────────────────────────────────────

async def _run_scraper(fn: Callable, topic: str, raw_dir: str) -> list[dict]:
    try:
        return await asyncio.to_thread(fn, topic, raw_dir)
    except Exception as e:
        print(f"  ⚠ scraper {getattr(fn, '__module__', '?')} failed: {e}")
        return []


# ── helpers ────────────────────────────────────────────────────────────────────


def _assemble_content(record: dict) -> str:
    """
    Join chunk texts for scoring/anomaly analysis.

    Overlap prefixes are stripped from overlapped chunks before joining so that
    keyword-density and entropy calculations see each word exactly once.
    The OVERLAP_WORDS constant defines how many words each overlap prefix contains.
    """
    if "_content" not in record:
        parts = []
        for c in record.get("content_chunks", []):
            if isinstance(c, dict):
                text = c["text"]
                if c.get("has_overlap"):
                    words = text.split()
                    ov = CFG.chunk_overlap_words
                    text = " ".join(words[ov:]) if len(words) > ov else text
            else:
                text = c
            parts.append(text)
        record["_content"] = "\n\n".join(parts)
    return record["_content"]


# ── enrichment ─────────────────────────────────────────────────────────────────

def _enrich(record: dict) -> dict:
    """
    Enrichment stage: anomaly detection + tag refresh + region inference.

    Anomaly flags are attached and forwarded to the trust engine.
    Content is assembled once and cached so _score() reuses it.
    Tag refresh uses full chunk content only (not URL) to avoid slug noise.
    Region is inferred from TLD once here so it's available in output.
    """
    content = _assemble_content(record)
    flags   = detect_anomalies(content, record)
    # For YouTube records with transcripts, run additional transcript quality checks
    if record.get("has_transcript") and record.get("source_type") == "youtube":
        tx_flags = detect_transcript_quality(content)
        flags    = flags + tx_flags
    if flags:
        record["anomaly_flags"] = flags

    # NOTE: tag refresh used to run a second auto_tag(content) here. Removed
    # because the scraper already tags on title+body (richer context than
    # chunks-rejoined-without-title). The duplicate call could only equal or
    # degrade quality while costing ~one regex pass per record per topic.

    # Region inference from source domain TLD
    if record.get("region", "Unknown") == "Unknown":
        record["region"] = infer_region(record.get("source_url", ""))

    return record


# ── scoring ────────────────────────────────────────────────────────────────────

def _score(record: dict) -> dict:
    """Trust engine. Uses cached content assembled in _enrich."""
    content = _assemble_content(record)

    result = trust_calculate(
        source_url     = record.get("source_url", ""),
        source_type    = record.get("source_type", "blog"),
        author         = record.get("author", ""),
        published_date = record.get("published_date", ""),
        content        = content,
        citations      = record.get("citations"),
        views          = record.get("view_count"),
        likes          = record.get("like_count"),
        anomaly_flags  = record.get("anomaly_flags", []),
        record_dict    = record,
    )
    record.update(result)
    update_prior(record.get("source_url", ""), result.get("trust_score", CFG.prior_init_score))
    return record


# ── output formatting ──────────────────────────────────────────────────────────

def _format_output(record: dict, ingested_at: str) -> dict:
    """
    Canonical output schema with provenance fields.

    record_id     - stable SHA-256[:16] of source_type + source_url; deterministic
                    across runs so records can be deduplicated by downstream systems.
    schema_version- bumped when the output schema changes; lets consumers detect stale data.
    ingested_at   - UTC timestamp of this pipeline run (same value for all records in a run).
    trust_tier    - human-readable band alongside the float trust_score.
    word_count    - added per chunk; avoids recomputing in downstream systems.
    _content      - internal cache key; stripped before serialisation.
    """
    raw_chunks = record.get("content_chunks", [])
    norm_chunks = []
    for i, c in enumerate(raw_chunks):
        if isinstance(c, dict):
            chunk = dict(c)
        else:
            chunk = {"text": c, "chunk_index": i, "total_chunks": len(raw_chunks),
                     "quality_score": CFG.chunk_fallback_quality_unknown, "has_overlap": False}
        chunk["word_count"] = len(chunk["text"].split())
        norm_chunks.append(chunk)

    source_url  = record.get("source_url", "")
    source_type = record.get("source_type", "blog")
    ts          = float(record.get("trust_score") or 0.0)

    out = {
        # ── provenance ─────────────────────────────────────────────────────
        "record_id":      make_record_id(source_url, source_type),
        "schema_version": SCHEMA_VERSION,
        "ingested_at":    ingested_at,
        # ── canonical required fields ──────────────────────────────────────
        "source_url":     source_url,
        "source_type":    source_type,
        "author":         record.get("author", "Unknown"),
        "published_date": record.get("published_date", "Unknown"),
        "language":       record.get("language", "unknown"),
        "region":         record.get("region", "Unknown"),
        "topic_tags":     record.get("topic_tags", []),
        "trust_score":    ts,
        "trust_tier":     _trust_tier(ts),
        "content_chunks": norm_chunks,
        # ── scoring breakdown (interpretability) ───────────────────────────
        "sub_scores":     record.get("sub_scores", {}),
        "confidence":     record.get("confidence", {}),
        "evidence":       record.get("evidence", {}),
        "weights":        record.get("weights", {}),
        "abuse_penalty":  float(record.get("abuse_penalty", 1.0)),
        "abuse_reasons":  record.get("abuse_reasons", []),
        "anomaly_flags":  record.get("anomaly_flags", []),
        "prior_used":     record.get("prior_used"),
    }
    # Source-type-specific metadata - preserved from scraper output
    if source_type == "pubmed":
        out["pmid"]    = record.get("pmid", "")
        out["doi"]     = record.get("doi", "")
        out["journal"] = record.get("journal", "")
    elif source_type == "youtube":
        out["view_count"]     = record.get("view_count")
        out["like_count"]     = record.get("like_count")
        out["duration_sec"]   = record.get("duration_sec")
        out["has_transcript"] = record.get("has_transcript", False)
    return out


# ── pipeline ───────────────────────────────────────────────────────────────────

async def _pipeline(topic: str, output_dir: str, concurrent: bool,
                    global_seen_urls: set[str] | None = None) -> list[dict]:
    from project.scraper.blog_scraper    import run as blog_run
    from project.scraper.youtube_scraper import run as yt_run
    from project.scraper.pubmed_scraper  import run as pm_run

    t0  = time.perf_counter()
    bar = "─" * CFG.display_bar_narrow
    print(f"\n{bar}")
    print(f"  ContentRadar  |  topic='{topic}'  |  {datetime.now():%Y-%m-%d %H:%M:%S}")
    print(f"{bar}\n")

    # ── Stage 1: Ingestion ─────────────────────────────────────────────────────
    mode = "concurrent" if concurrent else "sequential"
    print(f"[1/4] Ingestion ({mode}) …")

    if concurrent:
        blog_recs, yt_recs, pm_recs = await asyncio.gather(
            _run_scraper(blog_run, topic, output_dir),
            _run_scraper(yt_run,   topic, output_dir),
            _run_scraper(pm_run,   topic, output_dir),
        )
    else:
        blog_recs = blog_run(topic, output_dir)
        yt_recs   = yt_run(topic,   output_dir)
        pm_recs   = pm_run(topic,   output_dir)

    all_records = blog_recs + yt_recs + pm_recs
    print(f"      {len(all_records)} records ingested  "
          f"(blog={len(blog_recs)} yt={len(yt_recs)} pubmed={len(pm_recs)})  "
          f"elapsed={time.perf_counter()-t0:.2f}s\n")

    if not all_records:
        print(f"  ⚠ Sparse retrieval: all scrapers returned 0 records for topic='{topic}'. "
              f"Check API keys, network, and query relevance.")
        return []

    formatted: list[dict] = []
    try:
        # ── Stage 2: Enrichment ────────────────────────────────────────────────
        print("[2/4] Enrichment (anomaly detection + tag refresh + region) …")
        enriched = [_enrich(r) for r in all_records]
        flagged  = sum(1 for r in enriched if r.get("anomaly_flags"))
        print(f"      {flagged} records flagged with content anomalies\n")

        # ── Stage 3: Trust Scoring ─────────────────────────────────────────────
        print("[3/4] Trust scoring …")
        scored = [_score(r) for r in enriched]

        if scored:
            scores = [r["trust_score"] for r in scored]
            print(f"      n={len(scores)}  "
                  f"avg={sum(scores)/len(scores):.3f}  "
                  f"min={min(scores):.3f}  "
                  f"max={max(scores):.3f}\n")

        # Hard reject: drop records below the trust floor before persist.
        # These are records where evidence of manipulation or quality failure
        # is strong enough that including them would degrade downstream RAG.
        hard_rejected = [r for r in scored if r.get("trust_score", 0) < CFG.hard_reject_trust_floor]
        scored = [r for r in scored if r.get("trust_score", 0) >= CFG.hard_reject_trust_floor]
        if hard_rejected:
            print(f"  ✗ {len(hard_rejected)} records hard-rejected "
                  f"(trust < {CFG.hard_reject_trust_floor}): "
                  + ", ".join(r.get("source_url", "?")[-CFG.display_url_tail_chars:] for r in hard_rejected))

        # ── Stage 4: Persist ───────────────────────────────────────────────────
        print("[4/4] Persisting …")
        out_path = Path(output_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        ingested_at = now_utc()

        # Strip internal cache key before serialisation
        for r in scored:
            r.pop("_content", None)

        formatted_raw = [_format_output(r, ingested_at) for r in scored]

        # Schema validation - reject records that violate the canonical schema.
        # A schema violation here is a pipeline bug, not a data quality issue.
        validation_failures: list[tuple[str, str]] = []
        for rec in formatted_raw:
            try:
                validate_record(rec)
                formatted.append(rec)
            except SchemaError as e:
                validation_failures.append((rec.get("source_url", "?"), str(e)))

        if validation_failures:
            print(f"  ⚠ {len(validation_failures)} records failed schema validation - excluded:")
            for url, err in validation_failures:
                print(f"      ✗ {url[-CFG.display_url_accept_chars:]}: {err}")

        # Final record_id dedup - prevents the same source_url appearing twice if
        # concurrent scrapers or retry logic produced overlapping ingestion results.
        seen_ids: set[str] = set()
        deduped: list[dict] = []
        for rec in formatted:
            rid = rec.get("record_id", "")
            if rid and rid in seen_ids:
                continue
            if rid:
                seen_ids.add(rid)
            deduped.append(rec)
        if len(deduped) < len(formatted):
            print(f"  ⚠ {len(formatted) - len(deduped)} duplicate record_ids removed before persist")
        formatted = deduped

        # Cross-topic dedup: drop any URL already seen in a prior topic's run.
        # Mutates global_seen_urls so subsequent topics skip these sources.
        if global_seen_urls is not None:
            pre_cross = len(formatted)
            formatted = [r for r in formatted if r.get("source_url", "") not in global_seen_urls]
            for r in formatted:
                global_seen_urls.add(r.get("source_url", ""))
            dropped = pre_cross - len(formatted)
            if dropped:
                print(f"  ⚠ {dropped} records dropped (cross-topic duplicates)")

        # Sort by trust_score descending for deterministic, reproducible output ordering.
        # Secondary sort by source_url for stable tie-breaking (Python sort is stable
        # so equal trust_scores preserve relative ingestion order, but explicit
        # secondary key makes behaviour explicit and independent of dict ordering).
        formatted.sort(key=lambda r: (-r.get("trust_score", 0.0), r.get("source_url", "")))

        # Per-source typed files
        by_type: dict[str, list[dict]] = {}
        for rec in formatted:
            by_type.setdefault(rec.get("source_type", "blog"), []).append(rec)

        for st, records in by_type.items():
            fname = TYPE_TO_FILE.get(st, st) + ".json"
            dst   = out_path / fname
            dst.write_text(json.dumps(records, indent=2, ensure_ascii=False))
            real_t = [r["trust_score"] for r in records if r.get("trust_score", 0) > 0]
            avg    = sum(real_t) / len(real_t) if real_t else 0
            tier_dist: dict[str, int] = {}
            for r in records:
                t = r.get("trust_tier", "?")
                tier_dist[t] = tier_dist.get(t, 0) + 1
            print(f"      {fname:<18} {len(records)} records  avg_trust={avg:.3f}  "
                  f"tiers={tier_dist}  → {dst}")

        # Combined scraped_data.json with run manifest
        generated_at = ingested_at  # UTC timestamp shared with ingested_at
        tier_counts: dict[str, int] = {}
        for r in formatted:
            t = r.get("trust_tier", "?")
            tier_counts[t] = tier_counts.get(t, 0) + 1

        manifest = {
            "meta": {
                "topic":          topic,
                "generated_at":   generated_at,
                "ingested_at":    ingested_at,
                "pipeline":       "ContentRadar v2",
                "schema_version": SCHEMA_VERSION,
                "total_records":  len(formatted),
                "avg_trust":      round(sum(r["trust_score"] for r in formatted) / max(len(formatted), 1), 3),
                "trust_tiers":    tier_counts,
                "sources":        {st: len(recs) for st, recs in by_type.items()},
                "sorted_by":      "trust_score_desc",
            },
            "records": formatted,
        }
        combined = out_path / "scraped_data.json"
        combined.write_text(json.dumps(manifest, indent=2, ensure_ascii=False))
        print(f"      scraped_data.json  {len(formatted)} records + manifest  → {combined}")

    finally:
        # Runs even if scoring or serialisation raises, preserving all Bayesian
        # domain mean-updates that were accumulated during the scoring stage.
        try:
            persist_priors()
        except Exception as e:
            print(f"  ⚠ persist_priors failed: {e}")

    elapsed = time.perf_counter() - t0
    bar = "─" * CFG.display_bar_narrow
    print(f"\n{bar}")
    print(f"  Done in {elapsed:.2f}s  |  output -> {out_path}/")
    print(f"{bar}\n")

    return formatted


def _topic_slug(topic: str) -> str:
    """Stable filesystem slug from topic string."""
    return re.sub(r"[^a-z0-9]+", "_", topic.lower().strip()).strip("_")


def _display_results(records: list[dict], topic: str, out_path: str) -> None:
    """Print a human-readable summary of pipeline results."""
    if not records:
        print("  No records found for this topic.")
        return

    bar  = "=" * CFG.display_bar_wide
    bar2 = "-" * CFG.display_bar_wide
    TIER_SYMBOL = {
        "AUTHORITATIVE": "★★★",
        "CREDIBLE":       "★★ ",
        "UNCERTAIN":      "★  ",
        "UNRELIABLE":     "☆  ",
    }

    scores     = [r["trust_score"] for r in records]
    by_type    = {}
    tier_count: dict[str, int] = {}
    for r in records:
        by_type.setdefault(r["source_type"], []).append(r)
        t = r.get("trust_tier", "?")
        tier_count[t] = tier_count.get(t, 0) + 1

    print(f"\n{bar}")
    print(f"  ContentRadar Results  |  topic: {topic!r}")
    print(bar)
    print(f"  {len(records)} records   avg trust {sum(scores)/len(scores):.2f}   "
          f"min {min(scores):.2f}   max {max(scores):.2f}")
    print(f"  Sources: " + "  ".join(f"{k}={len(v)}" for k, v in sorted(by_type.items())))
    tier_str = "  ".join(f"{k}={v}" for k, v in sorted(tier_count.items(), reverse=True))
    print(f"  Tiers:   {tier_str}")
    print(bar2)

    # Show top record per source type
    for src_type, recs in sorted(by_type.items()):
        top = sorted(recs, key=lambda r: -r["trust_score"])
        print(f"\n  {src_type.upper()} ({len(recs)} record{'s' if len(recs)>1 else ''})")
        for r in top:
            sym  = TIER_SYMBOL.get(r.get("trust_tier", ""), "   ")
            tags = ", ".join(r.get("topic_tags", [])[:CFG.display_tags_count])
            url  = r.get("source_url", "")
            author = r.get("author", "Unknown")[:CFG.display_author_max_chars]
            date   = r.get("published_date", "?")[:10]
            words  = sum(c.get("word_count", len(c.get("text","").split()))
                         for c in r.get("content_chunks", []))
            flags  = r.get("anomaly_flags", [])
            abuse  = r.get("abuse_penalty", 1.0)

            print(f"    {sym} {r['trust_score']:.3f}  [{r.get('trust_tier','?')}]")
            print(f"         {url}")
            print(f"         author={author}  date={date}  words={words}")
            print(f"         tags=[{tags}]")
            if flags:
                print(f"         anomaly_flags={flags}")
            if abuse < 1.0:
                print(f"         abuse_penalty={abuse:.3f}  reasons={r.get('abuse_reasons',[])}")
            # Show first chunk preview
            chunks = r.get("content_chunks", [])
            if chunks:
                preview = chunks[0].get("text", "")[:CFG.display_preview_chars].replace("\n", " ")
                print(f"         preview: {preview!r}")
            print()

    print(bar2)
    print(f"  Full output  →  {out_path}/scraped_data.json")
    print(f"{bar}\n")


def _display_multi_results(all_results: list[tuple[str, list[dict], str]]) -> None:
    """Compact summary table when multiple topics were scraped."""
    bar  = "=" * CFG.display_bar_wide
    bar2 = "-" * CFG.display_bar_wide
    print(f"\n{bar}")
    print(f"  ContentRadar  |  {len(all_results)} topics completed")
    print(bar2)
    print(f"  {'Topic':<22}  {'Records':>7}  {'Avg Trust':>9}  {'Tiers'}")
    print(f"  {bar2}")
    total_records = 0
    for topic, records, out_dir in all_results:
        total_records += len(records)
        if not records:
            print(f"  {topic:<22}  {'0':>7}  {'-':>9}  no records")
            continue
        scores     = [r["trust_score"] for r in records]
        avg        = sum(scores) / len(scores)
        tier_count: dict[str, int] = {}
        for r in records:
            t = r.get("trust_tier", "?")
            tier_count[t] = tier_count.get(t, 0) + 1
        tier_str = " ".join(f"{k[0]}:{v}" for k, v in sorted(tier_count.items()))
        print(f"  {topic:<22}  {len(records):>7}  {avg:>9.3f}  {tier_str}")
    print(f"  {bar2}")
    print(f"  {'TOTAL':<22}  {total_records:>7}")
    print(f"\n  Per-topic output -> project/output/topics/<slug>/scraped_data.json")
    print(f"{bar}\n")


# ── entry point ────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="ContentRadar - multi-source scraper + trust scoring"
    )
    parser.add_argument("--topics",     default=None,
                        help="Comma-separated topics, e.g. 'diabetes, RAG, climate change'")
    parser.add_argument("--output",     default=None,
                        help="Base output directory (default: project/output/topics/)")
    parser.add_argument("--sequential", action="store_true",
                        help="Disable concurrent scraping (slower; useful for debugging)")
    args = parser.parse_args()

    # ── topic resolution ───────────────────────────────────────────────────────
    raw = args.topics
    if not raw or not raw.strip():
        raw = input("Enter topic(s), comma-separated for multiple: ").strip()
    if not raw:
        print("✗ No topics provided. Exiting.")
        sys.exit(1)

    # Parse comma-separated, strip whitespace, drop blanks, preserve order
    topics = [t.strip() for t in raw.split(",") if t.strip()]
    if not topics:
        print("✗ No valid topics after parsing. Exiting.")
        sys.exit(1)

    base_out = args.output or str(ROOT / "project" / "output" / "topics")

    # ── per-topic pipeline runs ────────────────────────────────────────────────
    # Each topic runs sequentially here; scrapers within each topic run
    # concurrently via asyncio.gather (the real parallelism).
    # Running topics themselves in parallel would multiply API quota consumption
    # and could exhaust YouTube/NCBI rate limits, which is worse than slower output.
    global_seen_urls: set[str] = set()   # cross-topic URL dedup
    all_results: list[tuple[str, list[dict], str]] = []

    for topic in topics:
        slug    = _topic_slug(topic)
        out_dir = str(Path(base_out) / slug)
        records = asyncio.run(_pipeline(
            topic, out_dir,
            concurrent=not args.sequential,
            global_seen_urls=global_seen_urls,
        ))
        all_results.append((topic, records, out_dir))

    # ── combined results display ───────────────────────────────────────────────
    if len(all_results) == 1:
        topic, records, out_dir = all_results[0]
        _display_results(records, topic, out_dir)
    else:
        _display_multi_results(all_results)


if __name__ == "__main__":
    main()
