"""
project/utils/topic_expansion.py - Claude API-powered dynamic topic expansion.

Replaces all hardcoded _SLUG_MAP, _TERM_MAP, and _AI_TOPICS dictionaries.
Any topic the user supplies gets intelligent, domain-aware expansion - not
just the 16 topics that were previously hard-wired.

Cache: in-memory + disk (CFG.topic_expansion_cache_filename). One API call
per unique topic across the lifetime of the project, not per process.
Fallback: raw topic string used as minimal expansion if API is unavailable.

Output contract (TopicExpansion):
  terms       - 5-10 lowercase search terms / synonyms / related concepts
  slugs       - 3-6 hyphenated URL slugs for RSS tag feeds (Medium, etc.)
  is_technical- True for AI / ML / software topics; False for health / science
  pubmed_filter- additional PubMed AND-clause for technical topics (empty for health)
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, asdict, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

try:
    import anthropic as _anthropic
    _HAS_ANTHROPIC = True
except ImportError:
    _HAS_ANTHROPIC = False

from project.scoring.config import CFG


@dataclass
class TopicExpansion:
    terms:         list[str]
    slugs:         list[str]
    is_technical:  bool
    pubmed_filter: str = field(default="")


_cache: dict[str, TopicExpansion] = {}
_DISK_CACHE_PATH = ROOT / CFG.topic_expansion_cache_filename
_disk_loaded = False


def _load_disk_cache() -> None:
    """Hydrate in-memory cache from disk on first access. Tolerates corrupt files."""
    global _disk_loaded
    if _disk_loaded:
        return
    _disk_loaded = True
    try:
        if _DISK_CACHE_PATH.exists():
            data = json.loads(_DISK_CACHE_PATH.read_text())
            for key, payload in data.items():
                if isinstance(payload, dict):
                    try:
                        _cache[key] = TopicExpansion(
                            terms=list(payload.get("terms", [])),
                            slugs=list(payload.get("slugs", [])),
                            is_technical=bool(payload.get("is_technical", False)),
                            pubmed_filter=str(payload.get("pubmed_filter", "")),
                        )
                    except Exception:
                        continue
    except Exception:
        pass


def _persist_disk_cache() -> None:
    """Atomic write to avoid corruption when concurrent runs share the file."""
    try:
        tmp = _DISK_CACHE_PATH.with_suffix(_DISK_CACHE_PATH.suffix + ".tmp")
        payload = {k: asdict(v) for k, v in _cache.items()}
        tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False))
        os.replace(tmp, _DISK_CACHE_PATH)
    except Exception:
        pass


# Prompt-injection markers commonly seen in the wild. If a user-supplied topic
# string contains these, we strip the suspicious fragment before it enters the
# Claude prompt template. The orchestrator must not be hijackable via topic input.
_INJECTION_STRIP_RE = re.compile(
    r"(ignore\s+(all\s+)?previous\s+instructions"
    r"|disregard\s+(the\s+)?above"
    r"|new\s+persona\s*:"
    r"|system\s*:\s*you\s+(?:are|must|should|will)"
    r"|<\s*/?system\s*>"
    r"|\[INST\]"
    r"|<<SYS>>)",
    re.IGNORECASE,
)


def _sanitize_topic(topic: str) -> str:
    """
    Defang a topic string before it enters the Claude prompt.

    1. Drop ASCII control chars (newlines, tabs, NULs) that could break out of
       the surrounding prompt template.
    2. Strip prompt-injection patterns the trust engine already flags in
       scraped content - we apply the same filter at orchestrator input.
    3. Truncate to CFG.topic_expansion_max_topic_len so a 100KB topic string
       can't blow the prompt budget or trigger billing surprises.
    """
    s = topic.replace("\x00", "").replace("\n", " ").replace("\r", " ").replace("\t", " ")
    s = _INJECTION_STRIP_RE.sub("", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s[: CFG.topic_expansion_max_topic_len]

_SYSTEM = (
    "You are a biomedical and AI/tech content discovery specialist. "
    "Return ONLY valid JSON — no markdown, no explanation, no code fences."
)

_PROMPT = """\
For the research topic "{topic}", produce structured search expansion for a \
multi-source content pipeline (blogs, YouTube, PubMed).

Return exactly this JSON schema:
{{
  "terms": ["term1", ...],
  "slugs": ["slug1", ...],
  "is_technical": true | false,
  "pubmed_filter": "string or empty string"
}}

Rules:
- terms: {min_terms}-10 specific lowercase synonyms / related concepts. \
Precise enough to avoid noise, broad enough for recall.
- slugs: {min_slugs}-6 hyphenated lowercase URL slugs that would appear in \
health/tech article URLs on Medium, Healthline, TDS, etc.
- is_technical: true ONLY for AI, ML, software, computer science topics.
- pubmed_filter: non-empty only for technical topics that need a PubMed \
AND-clause (e.g. "machine learning OR deep learning OR neural network"). \
Empty string for health/medical topics."""


def expand_topic(topic: str) -> TopicExpansion:
    """Return cached expansion for topic. Calls Claude API on first access.

    Two-tier cache: in-memory dict (process lifetime) + disk JSON (project
    lifetime). Disk hydrate happens once on first call. New API responses are
    persisted immediately so a crashed process doesn't lose work.
    """
    _load_disk_cache()
    sanitized = _sanitize_topic(topic)
    key = sanitized.lower().strip()
    if not key:
        return _fallback(topic)
    if key in _cache:
        return _cache[key]

    expansion = _call_api(sanitized) if _HAS_ANTHROPIC else None
    if expansion is None:
        expansion = _fallback(sanitized)

    _cache[key] = expansion
    _persist_disk_cache()
    return expansion


def _call_api(topic: str) -> TopicExpansion | None:
    try:
        from dotenv import load_dotenv
        load_dotenv(ROOT / ".env")
        api_key = os.getenv("ANTHROPIC_API_KEY", "")
        if not api_key:
            print("  ⚠ topic_expansion: ANTHROPIC_API_KEY not set - using fallback")
            return None

        # Bound the API timeout so a hung connection cannot block the entire
        # pipeline. The Anthropic SDK accepts a per-client timeout in seconds.
        client = _anthropic.Anthropic(
            api_key=api_key,
            timeout=CFG.topic_expansion_timeout_sec,
        )
        msg = client.messages.create(
            model=CFG.topic_expansion_model,
            max_tokens=CFG.topic_expansion_max_tokens,
            system=_SYSTEM,
            messages=[{
                "role": "user",
                "content": _PROMPT.format(
                    topic=topic,
                    min_terms=CFG.topic_expansion_min_terms,
                    min_slugs=CFG.topic_expansion_min_slugs,
                ),
            }],
        )
        raw = msg.content[0].text.strip()
        # Strip accidental markdown fences
        raw = re.sub(r"^```[a-z]*\n?", "", raw)
        raw = re.sub(r"\n?```$", "", raw)
        data = json.loads(raw)

        terms  = [str(t).lower() for t in data.get("terms", []) if t]
        slugs  = [str(s).lower() for s in data.get("slugs", []) if s]
        if len(terms) < CFG.topic_expansion_min_terms or len(slugs) < CFG.topic_expansion_min_slugs:
            print(f"  ⚠ topic_expansion: API returned too few terms/slugs for '{topic}' - using fallback")
            return None

        expansion = TopicExpansion(
            terms=terms,
            slugs=slugs,
            is_technical=bool(data.get("is_technical", False)),
            pubmed_filter=str(data.get("pubmed_filter", "")),
        )
        print(f"  ✓ topic_expansion: '{topic}' -> {len(terms)} terms, "
              f"{len(slugs)} slugs, technical={expansion.is_technical}")
        return expansion

    except Exception as e:
        print(f"  ⚠ topic_expansion: Claude API error for '{topic}': {e}")
        return None


def _fallback(topic: str) -> TopicExpansion:
    """Minimal expansion when API is unavailable."""
    base  = topic.lower().strip()
    slug  = re.sub(r"[^a-z0-9]+", "-", base).strip("-")
    words = base.split()
    return TopicExpansion(
        terms=[base] + words if len(words) > 1 else [base],
        slugs=[slug],
        is_technical=False,
        pubmed_filter="",
    )
