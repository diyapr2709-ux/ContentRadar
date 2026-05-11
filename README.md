# ContentRadar

A topic-driven, trust-aware content ingestion pipeline. Give it one or more keywords and it concurrently scrapes blogs, YouTube, and PubMed, runs anomaly detection on every record, and produces confidence-weighted trust scores with a full evidence breakdown for each source.

---

## Quick start

```bash
git clone <repo-url> ContentRadar
cd ContentRadar
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` in the project root:

```bash
YOUTUBE_API_KEY=AIzaSy...        # required for YouTube scraping
NCBI_API_KEY=                    # optional; lifts PubMed rate limit
```

Run the pipeline:

```bash
# Interactive (prompts for topics — comma-separated)
python main.py

# One topic
python main.py --topics "diabetes"

# Multiple topics → one folder per topic
python main.py --topics "diabetes, RAG, climate change"

# Sequential mode (easier debugging)
python main.py --topics "gut health" --sequential

# Custom output directory
python main.py --topics "cancer" --output /tmp/out
```

Run the test suite:

```bash
pip install pytest
pytest tests/

# Or, no-pytest fallback:
python tests/run_without_pytest.py
```

---

## Architecture

```
project/
  scraper/
    blog_scraper.py        Medium RSS → topic-expanded slug search
    youtube_scraper.py     YouTube Data API v3 + transcript extraction
    pubmed_scraper.py      NCBI eUtils (esearch → esummary → efetch)
  scoring/
    trust_score.py         Confidence-weighted trust engine + abuse detection
    signals.py             Per-signal scorers (author, domain, recency, ...)
    config.py              Source-type weights, thresholds, tier cutoffs
  utils/
    http_client.py         Circuit breaker + exponential backoff + retry
    chunking.py            RAG-ready chunker with overlap + quality scoring
    tagging.py             Positional-TF semantic tagger
    sanitizer.py           NUL/bidi/injection-safe text cleaning
    anomaly.py             Heuristic content anomaly detector
    schema.py              Canonical record schema + validation
    semantic.py            Optional cosine-similarity relevance gate
    topic_expansion.py     Topic → slug variants for blog search
  output/
    topics/<topic>/        Per-topic scored output
    index.json             Cross-topic index
tests/                     Pytest suite (determinism, invariants, ...)
main.py
build_index.py
requirements.txt
```

---

## Trust score

Every record receives a `trust_score ∈ [0, 1]` computed as a confidence-weighted sum of six signals, multiplied by an abuse penalty that cannot be offset by inflating other signals.

```
raw_score   = Σ(weight_i × score_i × confidence_i) / Σ(weight_i × confidence_i)
final_score = raw_score × abuse_penalty × prior_pull
```

### Signal weights (source-type-calibrated)

| Signal               | Blog  | YouTube | PubMed |
|----------------------|-------|---------|--------|
| author_credibility   | 0.18  | 0.22    | 0.18   |
| domain_authority     | 0.30  | 0.15    | 0.10   |
| recency              | 0.20  | 0.23    | 0.15   |
| content_quality      | 0.17  | 0.15    | 0.17   |
| engagement           | 0.05  | 0.15    | 0.35   |
| medical_disclaimer   | 0.10  | 0.10    | 0.05   |

**Confidence propagation.** Each signal returns a `(score, confidence)` pair. Signals with weak evidence (e.g. unknown citation count) contribute proportionally less rather than injecting a fake default into the sum.

**Recency decay.** Smooth exponential, not a step function:

```
score = 0.95 × 2^(−age_days / halflife)
half-lives: blog = 90d, youtube = 150d, pubmed = 900d
```

---

## Abuse prevention

All penalties are multiplicative and stack.

| Vector                         | Detection                               | Penalty  |
|--------------------------------|-----------------------------------------|----------|
| Anonymous author + health      | Author empty/unknown + health keyword   | ×0.70    |
| SEO spam URL                   | ≥2 spam-pattern hits in URL             | ×0.75    |
| Keyword stuffing               | Top token > 6% of content               | ×0.80    |
| No medical disclaimer          | Health keywords + no disclaimer regex   | sub = 0  |
| Outdated health content        | >2 years old (non-PubMed)               | ×0.82    |
| Low-authority + health domain  | Domain tier < 0.30 + health content     | ×0.75    |
| Anomaly flags                  | Per flag from anomaly detector          | ×0.90    |

Penalty floor: **0.05** — no record ever hits absolute zero.

---

## Edge cases handled

| Case                      | Behaviour                                              |
|---------------------------|--------------------------------------------------------|
| Author missing            | Domain-based fallback; confidence reduced to 0.40      |
| Publish date missing      | Neutral recency 0.30; confidence reduced to 0.30       |
| Transcript unavailable    | Description-only content; quality reflects this        |
| Multiple authors          | Per-author credibility averaged, not max/min           |
| Non-English content       | Scoring is language-agnostic; language tag is metadata |
| Long articles             | Sentence-aware chunking; overlap preserves RAG context |

---

## Key engineering decisions

**Concurrent ingestion (`asyncio.to_thread`).** All three scrapers run in parallel via the OS thread pool. Wall-clock latency drops from `sum(scraper latencies)` to `max(scraper latencies)` — typically 3–5× faster on a cold run.

**Per-host circuit breaker.** After 3 consecutive failures, a host is fast-failed for 60 seconds. A broken PubMed endpoint won't block blog or YouTube scrapers.

**Parser fallback chain (blogs).** `content:encoded` → RSS description → title-only. Each step isolates failures so one broken parser doesn't discard the record entirely.

**Source reliability priors (`.priors_cache.json`).** Per-domain Bayesian mean is updated after every scored record and persisted across runs. Historically reliable domains receive a small pull toward their proven prior; noisy domains get pulled down.

**RAG-compatible chunks.** Each chunk carries `text`, `chunk_index`, `total_chunks`, `quality_score`, `has_overlap`. The `quality_score` gates what enters the vector store. Overlap (28 words by default) prevents semantic units from being split across chunk boundaries.

**Anomaly detector.** Shannon entropy (spun content), link density (SEO farms), promotional language density, excessive caps, temporal inconsistency, encoding corruption, and uniform paragraph length (templated content). Each produces a flag string that propagates to the abuse multiplier.

**Optional semantic relevance gate.** If `sentence-transformers` is installed, a cosine-similarity gate catches records that pass keyword matching but are only tangentially relevant. Falls back to keyword-only when absent.

---

## Output format

```json
{
  "source_url":     "https://...",
  "source_type":    "blog | youtube | pubmed",
  "author":         "Author Name",
  "published_date": "YYYY-MM-DD",
  "language":       "en",
  "region":         "US",
  "topic_tags":     ["diabetes", "AI / ML"],
  "anomaly_flags":  [],
  "trust_score":    0.74,
  "trust_tier":     "high",
  "sub_scores":     {"author_credibility": 0.72, "domain_authority": 0.95, "...": "..."},
  "confidence":     {"author_credibility": 0.65, "domain_authority": 0.92, "...": "..."},
  "evidence":       {"author_credibility": {"reason": "named_author"}, "...": "..."},
  "weights":        {"author_credibility": 0.18, "domain_authority": 0.30, "...": "..."},
  "abuse_penalty":  1.0,
  "abuse_reasons":  [],
  "prior_used":     null,
  "content_chunks": [
    {"text": "...", "chunk_index": 0, "total_chunks": 4, "quality_score": 0.71, "has_overlap": false},
    {"text": "...", "chunk_index": 1, "total_chunks": 4, "quality_score": 0.68, "has_overlap": true}
  ]
}
```

Output layout:

```
project/output/
  topics/<topic_slug>/
    blogs.json
    youtube.json
    pubmed.json
    scraped_data.json    # merged + sorted by trust_score
  index.json             # cross-topic index
```

---

## Requirements

- Python 3.9+
- YouTube Data API v3 key (`YOUTUBE_API_KEY`)
- NCBI API key (`NCBI_API_KEY`, optional — raises PubMed rate limit from 3 to 10 req/s)
