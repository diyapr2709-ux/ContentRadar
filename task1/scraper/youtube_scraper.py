"""
task1/scraper/youtube_scraper.py

Topic-driven YouTube scraper. Scrapes exactly 2 videos per run.
Uses YouTube Data API v3 
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv
from langdetect import detect, LangDetectException

try:
    from youtube_transcript_api import (
        YouTubeTranscriptApi, NoTranscriptFound, TranscriptsDisabled,
    )
    _HAS_TRANSCRIPT = True
except ImportError:
    _HAS_TRANSCRIPT = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from task1.utils.sanitizer       import clean_text, validate_url
from task1.utils.tagging         import auto_tag
from task1.utils.chunking        import chunk_text
from task1.scraper.blog_scraper  import expand_topic

load_dotenv(ROOT / ".env")
YOUTUBE_API_KEY = os.getenv("YOUTUBE_API_KEY", "")

TARGET_COUNT   = 2
MAX_CANDIDATES = 10
MIN_RELEVANCE  = 0.003

YT_SEARCH_URL  = "https://www.googleapis.com/youtube/v3/search"
YT_DETAILS_URL = "https://www.googleapis.com/youtube/v3/videos"

HEADERS = {
    "User-Agent":      "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                       "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9",
}


def _detect_lang(text: str) -> str:
    try:
        return detect(text[:2000])
    except LangDetectException:
        return "en"


def _is_relevant(text: str, topic: str) -> bool:
    if not text:
        return False
    terms = expand_topic(topic)
    low = text.lower()
    words = low.split()
    if not words:
        return False
    hits = sum(low.count(t.lower()) for t in terms)
    return (hits / len(words)) >= MIN_RELEVANCE


def _api_search(topic: str) -> list[dict]:
    if not YOUTUBE_API_KEY:
        print("        ⚠ YOUTUBE_API_KEY not set in .env")
        return []

    seen_channels: set[str] = set()
    candidates: list[dict] = []

    for term in expand_topic(topic):
        if len(candidates) >= MAX_CANDIDATES:
            break
        try:
            resp = requests.get(YT_SEARCH_URL, params={
                "part": "snippet", "q": f"{term} explained",
                "type": "video", "maxResults": 5,
                "order": "relevance", "key": YOUTUBE_API_KEY,
            }, headers=HEADERS, timeout=12)
            resp.raise_for_status()
            for item in resp.json().get("items", []):
                vid_id  = item.get("id", {}).get("videoId", "")
                channel = item.get("snippet", {}).get("channelTitle", "")
                if not vid_id or channel in seen_channels:
                    continue
                seen_channels.add(channel)
                candidates.append({
                    "video_id":    vid_id,
                    "channel":     clean_text(channel),
                    "title":       clean_text(item["snippet"].get("title", "")),
                    "description": clean_text(item["snippet"].get("description", "")),
                    "published":   item["snippet"].get("publishedAt", "")[:10],
                    "url":         f"https://www.youtube.com/watch?v={vid_id}",
                })
        except Exception as e:
            print(f"        ⚠ search error: {e}")
        time.sleep(0.3)

    return candidates[:MAX_CANDIDATES]


def _fetch_stats_batch(video_ids: list[str]) -> dict[str, dict]:
    if not YOUTUBE_API_KEY or not video_ids:
        return {}
    try:
        resp = requests.get(YT_DETAILS_URL, params={
            "part": "statistics", "id": ",".join(video_ids),
            "key": YOUTUBE_API_KEY,
        }, headers=HEADERS, timeout=12)
        resp.raise_for_status()
        out: dict[str, dict] = {}
        for item in resp.json().get("items", []):
            s = item.get("statistics", {})
            out[item["id"]] = {
                "view_count": int(s.get("viewCount", 0)),
                "like_count": int(s.get("likeCount", 0)),
            }
        return out
    except Exception:
        return {}


def _fetch_transcript(video_id: str) -> str:
    if not _HAS_TRANSCRIPT:
        return ""
    try:
        listing = YouTubeTranscriptApi.list_transcripts(video_id)
        try:
            t = listing.find_transcript(["en"])
        except NoTranscriptFound:
            t = listing.find_generated_transcript(["en"])
        return clean_text(" ".join(e["text"] for e in t.fetch()))
    except (NoTranscriptFound, TranscriptsDisabled):
        return ""
    except Exception:
        return ""


def run(topic: str, output_dir: str) -> list[dict]:
    print(f"  [YouTube] topic='{topic}' expanded={expand_topic(topic)}")
    candidates = _api_search(topic)
    print(f"        candidates={len(candidates)}")

    stats = _fetch_stats_batch([c["video_id"] for c in candidates])
    candidates.sort(
        key=lambda c: stats.get(c["video_id"], {}).get("view_count", 0),
        reverse=True,
    )

    results: list[dict] = []
    for cand in candidates:
        if len(results) >= TARGET_COUNT:
            break

        vid_id, url = cand["video_id"], cand["url"]
        title       = cand["title"]
        description = cand["description"]

        transcript = _fetch_transcript(vid_id)
        content    = (description + "\n\n" + transcript).strip() \
                     if transcript else description
        full_text  = f"{title}\n\n{content}"

        if not _is_relevant(full_text, topic):
            continue

        results.append({
            "source_url":     url,
            "source_type":    "youtube",
            "author":         cand["channel"],
            "published_date": cand["published"],
            "language":       _detect_lang(full_text),
            "region":         "Unknown",
            "topic_tags":     auto_tag(full_text),
            "trust_score":    "",
            "content_chunks": chunk_text(content),
        })
        print(f"        ✓ {url}")

    while len(results) < TARGET_COUNT:
        results.append({
            "source_url":     f"https://youtube.com/no-result-{len(results)+1}",
            "source_type":    "youtube",
            "author":         "Unknown",
            "published_date": "Unknown",
            "language":       "unknown",
            "region":         "Unknown",
            "topic_tags":     [],
            "trust_score":    "",
            "content_chunks": [f"[scrape_error] Insufficient results for '{topic}'"],
        })

    out = Path(output_dir) / "youtube.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"        → {out}")
    return results


if __name__ == "__main__":
    run(sys.argv[1] if len(sys.argv) > 1 else "diabetes",
        "task1/output/scraped_data")