"""
project/scraper/youtube_scraper.py - Topic-driven YouTube scraper.

Strategy:
  1. YouTube Data API v3 search (term variants, "explained"/"overview" suffixes)
  2. Batch-fetch statistics + contentDetails in one call (views, likes, duration)
  3. Filter out very short videos (< MIN_DURATION_SEC) - trailers, clips, ads
  4. Composite ranking: relevance × log-view-score × transcript-bonus
     - prevents engagement bias from pure view_count sort
  5. Fetch transcript via youtube-transcript-api; fall back to description only
  6. Channel + video-ID diversity gate - one video per channel, no ID dupes

Transcript vs description fallback:
  Transcripts are richer but ~30% of videos have them disabled. Description-
  only records are still scored but receive lower content quality because they
  have lower lexical diversity (descriptions are intentionally short/keyword-rich).
  This is the correct epistemic behaviour - less content = less evidence = less
  trust confidence.

Quota awareness:
  Each Search call costs 100 units (10,000/day free tier → 100 searches/day).
  Quota-exhausted (403 quotaExceeded) is logged separately from auth failures.
"""
from __future__ import annotations

import json
import math
import os
import re
import sys
import time
from pathlib import Path

import requests
from dotenv import load_dotenv

try:
    from youtube_transcript_api import YouTubeTranscriptApi
    _TRANSCRIPT_API = YouTubeTranscriptApi()   # v1.x requires instance, not class methods
    _HAS_TRANSCRIPT = True
except (ImportError, Exception):
    _TRANSCRIPT_API  = None
    _HAS_TRANSCRIPT  = False

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from project.utils.http_client import session as _yt_session
from project.utils.sanitizer   import clean_text, detect_language
from project.utils.tagging     import auto_tag
from project.utils.chunking    import rag_chunk
from project.scraper.blog_scraper import expand_terms, _is_relevant
from project.scoring.config    import CFG

YOUTUBE_API_KEY  = os.getenv("YOUTUBE_API_KEY", "")
TARGET_COUNT     = CFG.youtube_target_count
MAX_CANDIDATES   = CFG.youtube_max_candidates
MIN_DURATION_SEC = CFG.youtube_min_duration_sec

# YouTube-specific JSON headers (different from the HTML Accept used by fetch())
_YT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
}

_QUOTA_RE   = re.compile(r"quotaExceeded|dailyLimitExceeded", re.IGNORECASE)
_ISO8601_RE = re.compile(r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?", re.IGNORECASE)
_YT_RETRY_ON = frozenset(CFG.http_yt_retry_server_codes)


def _yt_get(url: str, params: dict) -> requests.Response | None:
    """
    YouTube API GET using the shared connection-pool session with 5xx retry.

    Uses _yt_session (shared with all scrapers) instead of requests.get() so
    TCP connections to googleapis.com are pooled across search + stats calls.

    YouTube-specific 403 handling is preserved by the caller; this function only
    adds retry for transient server errors. 403 is returned as-is (not None) so
    callers can inspect the body for quotaExceeded vs auth failure.
    """
    timeout = (CFG.http_connect_timeout_sec, CFG.http_read_timeout_sec)
    for attempt in range(CFG.http_max_retries):
        try:
            resp = _yt_session.request("GET", url, params=params,
                                       headers=_YT_HEADERS, timeout=timeout)
            if resp.status_code in _YT_RETRY_ON and attempt < CFG.http_max_retries - 1:
                time.sleep(CFG.http_backoff_base ** attempt)
                continue
            return resp
        except (requests.ConnectionError, requests.Timeout):
            if attempt < CFG.http_max_retries - 1:
                time.sleep(CFG.http_backoff_base ** attempt)
        except Exception:
            break
    return None


def _parse_duration_sec(iso: str) -> int:
    """Convert ISO 8601 duration (PT1H2M3S) to total seconds."""
    m = _ISO8601_RE.match(iso or "")
    if not m:
        return 0
    h, mn, s = (int(x or 0) for x in m.groups())
    return h * 3600 + mn * 60 + s


def _api_search(topic: str) -> list[dict]:
    if not YOUTUBE_API_KEY:
        print("        ⚠ YOUTUBE_API_KEY not set - skipping YouTube")
        return []

    seen_channel_ids: set[str] = set()  # channel ID dedup (title strings are non-unique)
    seen_video_ids:   set[str] = set()  # prevent same video appearing via two queries
    candidates:       list[dict] = []

    terms = expand_terms(topic)
    # First suffix → applied to top-N terms for broad educational recall.
    # Additional suffixes → first term only (targeted complement queries).
    # All suffixes defined in CFG.youtube_query_suffixes.
    _sfx = CFG.youtube_query_suffixes
    queries  = [f"{t} {_sfx[0]}" for t in terms[:CFG.youtube_results_per_query]]
    queries += [f"{terms[0]} {s}" for s in _sfx[1:]]

    for q in queries:
        if len(candidates) >= MAX_CANDIDATES:
            break
        try:
            resp = _yt_get(CFG.youtube_search_url, {
                "part": "snippet", "q": q,
                "type": "video", "maxResults": CFG.youtube_results_per_query,
                "order": "relevance", "key": YOUTUBE_API_KEY,
                "relevanceLanguage": CFG.youtube_relevance_language,
                "safeSearch": CFG.youtube_safe_search,
            })
            if resp is None:
                print(f"        ⚠ YouTube search: request failed (network/timeout)")
                continue
            if resp.status_code == 403:
                body = resp.text
                if _QUOTA_RE.search(body):
                    print("        ⚠ YouTube API quota exhausted for today - skipping YouTube")
                    return []
                print(f"        ⚠ YouTube API access denied (check key restrictions): {resp.status_code}")
                return []
            resp.raise_for_status()
            for item in resp.json().get("items", []):
                vid_id  = item.get("id", {}).get("videoId", "")
                # Use .get("snippet") or {} - snippet key absent in malformed responses
                snippet = item.get("snippet") or {}
                # Use channelId (stable unique ID) not channelTitle (display name, non-unique)
                channel_id    = snippet.get("channelId", "")
                channel_title = snippet.get("channelTitle", "")
                if not vid_id or vid_id in seen_video_ids or channel_id in seen_channel_ids:
                    continue
                seen_video_ids.add(vid_id)
                if channel_id:
                    seen_channel_ids.add(channel_id)
                candidates.append({
                    "video_id":    vid_id,
                    "channel":     clean_text(channel_title),
                    "title":       clean_text(snippet.get("title", "")),
                    "description": clean_text(snippet.get("description", "")),
                    "published":   snippet.get("publishedAt", "")[:10],
                    "url":         f"https://www.youtube.com/watch?v={vid_id}",
                })
        except requests.HTTPError as e:
            print(f"        ⚠ YouTube search HTTP error: {e}")
        except Exception as e:
            print(f"        ⚠ YouTube search error: {e}")
        time.sleep(CFG.delay_youtube_search)

    return candidates[:MAX_CANDIDATES]


def _batch_stats(video_ids: list[str]) -> dict[str, dict]:
    """
    Single API call for all video stats + duration + full description.

    Requests snippet alongside statistics/contentDetails so we get the full
    video description (up to 5000 chars) instead of the ~100-char truncated
    search snippet. This is the primary content source when no transcript
    is available - richer descriptions = better trust scoring evidence.

    like_count is set to None (not 0) when the creator has disabled the like
    counter - the engagement_authenticity signal treats None as "data absent"
    and returns a low-confidence neutral score, avoiding a false manipulation
    flag that would fire if we silently defaulted to 0.
    """
    if not YOUTUBE_API_KEY or not video_ids:
        return {}
    try:
        resp = _yt_get(CFG.youtube_details_url, {
            "part": "statistics,contentDetails,snippet",
            "id":   ",".join(video_ids),
            "key":  YOUTUBE_API_KEY,
        })
        if resp is None:
            print("        ⚠ YouTube batch stats: request failed (network/timeout)")
            return {}
        if resp.status_code == 403:
            body = resp.text
            if _QUOTA_RE.search(body):
                print("        ⚠ YouTube batch stats: quota exhausted")
            else:
                print(f"        ⚠ YouTube batch stats: access denied ({resp.status_code})")
            return {}
        resp.raise_for_status()
        out: dict[str, dict] = {}
        for item in resp.json().get("items", []):
            vid_id = item.get("id", "")
            if not vid_id:
                continue
            try:
                s       = item.get("statistics", {})
                dur     = item.get("contentDetails", {}).get("duration", "")
                snippet = item.get("snippet") or {}
                # likeCount absent means the creator hid it - use None to distinguish
                # from genuine zero-like count (which is a manipulation signal).
                like_count = int(s["likeCount"]) if "likeCount" in s else None
                out[vid_id] = {
                    "view_count":       int(s.get("viewCount") or 0),
                    "like_count":       like_count,
                    "duration_sec":     _parse_duration_sec(dur),
                    "full_description": clean_text(snippet.get("description", "")),
                    "channel_title":    clean_text(snippet.get("channelTitle", "")),
                }
            except Exception:
                # One malformed stats item must not abort the entire batch.
                continue
        return out
    except Exception as e:
        print(f"        ⚠ YouTube batch stats failed: {e}")
        return {}


def _fetch_transcript(video_id: str) -> str:
    """
    Transcript fallback chain (youtube_transcript_api v1.x - instance-based API):
      1. English transcript (manual or auto-generated) via api.fetch()
      2. Any available language via api.list() → first transcript → .fetch()
    Returns "" on failure; caller uses full video description as fallback.
    """
    if not _HAS_TRANSCRIPT or _TRANSCRIPT_API is None:
        return ""
    try:
        data = _TRANSCRIPT_API.fetch(video_id, languages=CFG.youtube_transcript_languages)
        return clean_text(" ".join(s.text for s in data))
    except Exception:
        pass
    try:
        listing = list(_TRANSCRIPT_API.list(video_id))
        if listing:
            data = listing[0].fetch()
            return clean_text(" ".join(s.text for s in data))
    except Exception:
        pass
    return ""


def run(topic: str, output_dir: str) -> list[dict]:
    print(f"  [YouTube] topic='{topic}'")
    candidates = _api_search(topic)
    print(f"        candidates={len(candidates)}")

    stats = _batch_stats([c["video_id"] for c in candidates])

    # ── Duration pre-filter ────────────────────────────────────────────────────
    # Drop short videos (trailers/clips) BEFORE fetching transcripts.
    # Fetching transcripts for videos we're about to discard wastes quota and
    # time - each transcript fetch is a separate API call. Filter first, fetch
    # only for videos that can pass into the final selection.
    viable: list[dict] = []
    for cand in candidates:
        dur = stats.get(cand["video_id"], {}).get("duration_sec", MIN_DURATION_SEC)
        if dur > 0 and dur < MIN_DURATION_SEC:
            print(f"        ✗ {cand['url']}  duration={dur}s < {MIN_DURATION_SEC}s - skipping")
            continue
        viable.append(cand)
    candidates = viable

    # Pre-compute topic terms once outside the sort key - expand_terms() is pure
    # but calling it inside a sort key function runs it once per candidate.
    _topic_terms = [t.lower() for t in expand_terms(topic)]

    # Composite ranking eliminates pure engagement bias.
    # Score = log_view_score × relevance_signal × transcript_bonus × duration_bonus
    #   log_view_score: log10(views)/6 capped at 1.0 (1M views → 1.0)
    #   relevance_signal: keyword density of title+description vs topic terms
    #   transcript_bonus: 1.2× for transcribed content (richer scoring evidence)
    #   duration_bonus: 1.1× for longer videos (> 5 min, more substantive)
    # This prevents a 10M-view clickbait video from outranking a 50K-view
    # university lecture that's directly relevant with a transcript.
    def _composite_score(cand: dict) -> float:
        vs    = stats.get(cand["video_id"], {})
        views = vs.get("view_count", 0)
        dur   = vs.get("duration_sec", 0)
        log_v = math.log10(max(views, 1)) / CFG.view_log_denominator
        words = (cand["title"] + " " + cand["description"]).lower().split()
        hits  = sum(words.count(t) for t in _topic_terms)
        rel   = min(hits / max(len(words), 1) * CFG.youtube_rel_density_scale, 1.0)
        trans_bonus = CFG.youtube_transcript_bonus if cand.get("_has_transcript") else 1.0
        dur_bonus   = CFG.youtube_long_video_bonus if dur > CFG.youtube_long_video_sec else 1.0
        return log_v * max(rel, CFG.youtube_rel_floor) * trans_bonus * dur_bonus

    # Two-phase ranking to avoid fetching transcripts for candidates we'll
    # never select. Phase 1 sorts by composite_score with no transcript info
    # (all _has_transcript default to False, bonus collapses to 1.0).
    # Phase 2 fetches transcripts only for the top K = TARGET_COUNT × factor.
    # Phase 3 re-sorts with transcript bonus now applied to the K we fetched.
    # ~5× fewer transcript-API calls; outside-K candidates keep the same
    # bonus they would have had under the prior design (none, by default).
    candidates.sort(key=_composite_score, reverse=True)
    top_k = min(len(candidates), TARGET_COUNT * CFG.youtube_transcript_topk_factor)
    transcript_cache: dict[str, str] = {}
    for cand in candidates[:top_k]:
        t = _fetch_transcript(cand["video_id"])
        transcript_cache[cand["video_id"]] = t
        if t:
            cand["_has_transcript"] = True
    candidates.sort(key=_composite_score, reverse=True)

    results: list[dict] = []
    for cand in candidates:
        if len(results) >= TARGET_COUNT:
            break

        vid_stats  = stats.get(cand["video_id"], {})
        transcript = transcript_cache.get(cand["video_id"], "")
        full_desc  = vid_stats.get("full_description", "") or cand["description"]
        # Prefer transcript (richer) over description; always include full description
        if transcript:
            content = (full_desc + "\n\n" + transcript).strip()
        else:
            content = full_desc or cand["description"]
        full_text = f"{cand['title']}\n\n{content}"

        if len(full_text.split()) < CFG.youtube_min_content_words:
            print(f"        ✗ {cand['url']}  empty content - skipping")
            continue

        if not _is_relevant(full_text, topic):
            continue

        chunks = rag_chunk(content)
        if not chunks:
            chunks = [{"text": content, "chunk_index": 0,
                       "total_chunks": 1, "quality_score": CFG.chunk_fallback_quality_youtube, "has_overlap": False}]
        results.append({
            "source_url":     cand["url"],
            "source_type":    "youtube",
            "author":         vid_stats.get("channel_title") or cand["channel"],
            "published_date": cand["published"],
            "language":       detect_language(full_text),
            "region":         "Unknown",
            "topic_tags":     auto_tag(full_text),
            "view_count":     vid_stats.get("view_count"),
            "like_count":     vid_stats.get("like_count"),
            "duration_sec":   vid_stats.get("duration_sec"),
            "has_transcript": bool(transcript),
            "content_chunks": chunks,
        })
        print(f"        ✓ {cand['url']}  "
              f"views={vid_stats.get('view_count', 0):,}  "
              f"dur={vid_stats.get('duration_sec', 0)}s  "
              f"transcript={'yes' if transcript else 'no'}")

    out = Path(output_dir) / "youtube.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(results, indent=2, ensure_ascii=False))
    print(f"        → {out}")
    return results
