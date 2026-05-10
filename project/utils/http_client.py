"""
project/utils/http_client.py

Resilient HTTP layer shared by all scrapers.

Three layered defences against real-world network chaos:

  1. Exponential backoff + jitter
     Jitter breaks correlated retry storms when blog + youtube + pubmed
     scrapers all hit the same CDN edge at the same millisecond.

  2. Per-host circuit breaker  (CLOSED → OPEN → HALF-OPEN)
     Prevents a single broken API from consuming the entire retry budget.
     After cb_threshold consecutive failures the host is fast-failed for
     cb_timeout seconds, protecting other scrapers running concurrently.
     A threading.Lock guards all mutable state so concurrent asyncio.to_thread
     scraper calls don't race on the failure counters.

  3. Parser fallback chain
     try_extractors() runs candidate HTML-to-text functions in priority
     order and returns the first result with substantive content (≥ 100 chars).
     Isolates parser errors so one broken strategy doesn't kill the record.

Retry policy:
  4xx client errors (400/401/403/404/410/422/451): no retry, host not at fault.
  429 Too Many Requests: retry with Retry-After header or exponential backoff,
      capped at MAX_429_WAIT_SEC. Does NOT permanently open the circuit.
  503/504 transient server errors: retry with exponential backoff + jitter,
      advances failure counter (host IS struggling).
  Connection/timeout errors: retry with exponential backoff (no circuit advance
      - transient network, not the host's fault).
"""
from __future__ import annotations

import random
import threading
import time
from enum import Enum
from typing import Callable
from urllib.parse import urlparse

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from project.scoring.config import CFG

# ── tuneable knobs (all values from CFG - no hardcoding) ──────────────────────
CONNECT_TIMEOUT    = CFG.http_connect_timeout_sec
READ_TIMEOUT       = CFG.http_read_timeout_sec
TIMEOUT            = (CONNECT_TIMEOUT, READ_TIMEOUT)   # (connect, read) tuple
MAX_RETRIES        = CFG.http_max_retries
BACKOFF_BASE       = CFG.http_backoff_base
BACKOFF_JITTER     = CFG.http_backoff_jitter
MAX_RESPONSE_BYTES = CFG.http_max_response_bytes
REQUEST_DELAY_SEC  = CFG.http_request_delay_sec
CB_THRESHOLD       = CFG.http_cb_threshold
CB_TIMEOUT_SEC     = CFG.http_cb_timeout_sec
MAX_429_WAIT_SEC   = CFG.http_max_429_wait_sec

# ── shared session with connection pooling ────────────────────────────────────
# Session reuses TCP connections (Keep-Alive) across requests to the same host.
# urllib3 pool_connections=10 covers up to 10 distinct hosts simultaneously;
# pool_maxsize=20 allows 20 concurrent connections per host (asyncio.to_thread).
# No urllib3-level retry: our own retry loop gives per-status control.
def _make_session() -> requests.Session:
    s       = requests.Session()
    adapter = HTTPAdapter(
        pool_connections=CFG.http_pool_connections,
        pool_maxsize=CFG.http_pool_maxsize,
        max_retries=Retry(total=0),  # disable urllib3 retries; we do our own
    )
    s.mount("https://", adapter)
    s.mount("http://",  adapter)
    return s

session = _make_session()   # public: importable by scrapers for connection reuse
_session = session          # internal alias kept so fetch() below doesn't need changes

# Status code policy from CFG - tune retry behaviour without touching this file.
_NO_RETRY_CLIENT = frozenset(CFG.http_no_retry_client_codes)
_RETRY_SERVER    = frozenset(CFG.http_retry_server_codes)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept":          "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    # Accept-Encoding is intentionally omitted - requests adds it automatically
    # and handles decompression transparently. Setting it manually bypasses
    # requests' auto-decompression, leaving gzip bytes in resp.content.
    "DNT":             "1",
}


# ── circuit breaker ────────────────────────────────────────────────────────────

class _CBState(Enum):
    CLOSED    = "closed"
    OPEN      = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker:
    """
    Per-host circuit breaker with threading.Lock for concurrent scraper safety.

    CLOSED   → count failures; normal operation
    OPEN     → fast-fail all requests to this host
    HALF_OPEN→ one probe allowed; success→CLOSED, failure→OPEN (reset timer)

    The lock guards all compound read-modify-write operations on _failures,
    _opened, and _state. This prevents races when asyncio.to_thread runs
    multiple scrapers concurrently against overlapping hosts.
    """

    def __init__(self) -> None:
        self._failures: dict[str, int]      = {}
        self._opened:   dict[str, float]    = {}
        self._state:    dict[str, _CBState] = {}
        self._lock      = threading.Lock()

    @staticmethod
    def _host(url: str) -> str:
        try:
            return urlparse(url).netloc.lower()
        except Exception:
            return url

    def allow(self, url: str) -> bool:
        h = self._host(url)
        with self._lock:
            state = self._state.get(h, _CBState.CLOSED)
            if state is _CBState.CLOSED:
                return True
            if state is _CBState.OPEN:
                if time.monotonic() - self._opened.get(h, 0.0) > CB_TIMEOUT_SEC:
                    self._state[h] = _CBState.HALF_OPEN
                    return True
                return False
            return True  # HALF_OPEN lets one probe through

    def success(self, url: str) -> None:
        h = self._host(url)
        with self._lock:
            self._failures[h] = 0
            self._state[h]    = _CBState.CLOSED

    def failure(self, url: str) -> None:
        h = self._host(url)
        with self._lock:
            count = self._failures.get(h, 0) + 1
            self._failures[h] = count
            if count >= CB_THRESHOLD:
                self._state[h]  = _CBState.OPEN
                self._opened[h] = time.monotonic()


# Module-level singleton: shared state across all scrapers in one pipeline run.
circuit = CircuitBreaker()


# ── fetch ──────────────────────────────────────────────────────────────────────

def fetch(
    url:     str,
    params:  dict | None = None,
    headers: dict | None = None,
    method:  str         = "GET",
    retries: int         = MAX_RETRIES,
) -> requests.Response | None:
    """
    HTTP request with circuit breaker + exponential backoff + jitter.

    Returns None on permanent failure so callers branch to fallback logic
    rather than catching exceptions across the whole codebase.

    Retry behaviour by status code:
      4xx client errors → return None immediately (no retry, no circuit hit)
      429 rate limit    → honour Retry-After header, retry up to MAX_429_WAIT_SEC
      503/504 server    → retry with backoff, advance failure counter
      connection/timeout→ retry with backoff (no circuit advance)
    """
    if not circuit.allow(url):
        return None

    merged = {**HEADERS, **(headers or {})}

    for attempt in range(retries):
        try:
            resp = _session.request(
                method, url, params=params, headers=merged, timeout=TIMEOUT,
            )

            if resp.status_code in _NO_RETRY_CLIENT:
                return None   # client error: URL/auth problem, host is fine

            if resp.status_code == 429:
                # Rate-limited: honour Retry-After if present, else exponential backoff.
                # Does NOT advance the circuit - the host is healthy, we're just too fast.
                try:
                    wait = float(resp.headers.get("Retry-After", 0))
                except (ValueError, TypeError):
                    wait = 0.0
                wait = max(wait, BACKOFF_BASE ** attempt)
                if attempt < retries - 1:
                    time.sleep(min(wait, MAX_429_WAIT_SEC))
                    continue
                return None

            if resp.status_code in _RETRY_SERVER:
                # Transient server error: backoff and retry, advance failure counter.
                circuit.failure(url)
                if attempt < retries - 1:
                    _sleep(attempt)
                    continue
                return None

            resp.raise_for_status()
            if len(resp.content) > MAX_RESPONSE_BYTES:
                return None
            circuit.success(url)
            time.sleep(REQUEST_DELAY_SEC)
            return resp

        except requests.HTTPError:
            circuit.failure(url)
            if attempt < retries - 1:
                _sleep(attempt)

        except (requests.ConnectionError, requests.Timeout):
            # Network issue, not host's fault - don't advance circuit.
            if attempt < retries - 1:
                _sleep(attempt)

        except Exception:
            # Unexpected error (encoding, SSL, etc.) - treat as server-side fault.
            circuit.failure(url)
            break

    # Do NOT call circuit.failure() here: the loop already advanced it per
    # server-side exception. Connection/timeout exhaustion must NOT open the
    # circuit because the host itself may be healthy (network path is the fault).
    return None


def _sleep(attempt: int) -> None:
    wait   = BACKOFF_BASE ** attempt
    jitter = wait * random.uniform(-BACKOFF_JITTER, BACKOFF_JITTER)
    time.sleep(max(CFG.http_backoff_min_sleep, wait + jitter))


def try_extractors(html: str, extractors: list[Callable[[str], str]]) -> str:
    """
    Parser fallback chain.

    Tries each function in priority order. Returns the first result with ≥ 100
    characters of content. A 100-char floor filters boilerplate (navbars, cookie
    banners) that parsers often surface when real content is blocked.
    """
    for fn in extractors:
        try:
            text = fn(html)
            if text and len(text.strip()) >= CFG.http_extractor_min_chars:
                return text.strip()
        except Exception:
            continue
    return ""
