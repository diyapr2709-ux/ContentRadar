ContentRadar

A topic-driven, trust-aware content ingestion pipeline. Give it a keyword
and it concurrently scrapes blogs, YouTube, and PubMed, runs anomaly
detection on every record, and produces confidence-weighted trust scores
with a full evidence breakdown for each source.

---

Architecture

```
project/
  scraper/
    blog_scraper.py      Medium RSS → topic-expanded slug search
    youtube_scraper.py   YouTube Data API v3 + transcript extraction
    pubmed_scraper.py    NCBI eUtils (esearch → esummary → efetch)
  scoring/
    trust_score.py       Confidence-weighted trust engine + abuse detection
    edge_cases.py        Runnable edge-case demonstrations
    abuse_prevention.py  Runnable abuse-vector demonstrations
  utils/
    http_client.py       Circuit breaker + exponential backoff + retry
    chunking.py          RAG-ready chunker with overlap + quality scoring
    tagging.py           Positional-TF semantic tagger
    sanitizer.py         NUL/bidi/injection-safe text cleaning
    anomaly.py           Heuristic content anomaly detector
  output/
    scraped_data.json    Combined scored output
    blogs.json
    youtube.json
    pubmed.json

task1/  (original scraper — preserved)
task2/  (original scoring — preserved)
main.py
```

---

Running

```bash
pip install -r requirements.txt

# .env
YOUTUBE_API_KEY=AIzaSy...
NCBI_API_KEY=           # optional; increases PubMed rate limit

# New pipeline (concurrent scraping + full enrichment)
python main.py --topic "diabetes"

# Sequential mode (easier debugging)
python main.py --topic "gut health" --sequential

# Original task1/task2 pipeline (preserved)
python main.py --topic "diabetes" --legacy

# Verification
python project/scoring/edge_cases.py
python project/scoring/abuse_prevention.py
```

---

Trust Score

Every record receives a trust_score ∈ [0, 1] computed as a
confidence-weighted sum of six signals, then multiplied by an
abuse penalty that cannot be offset by inflating other signals.

```
raw_score   = Σ(weight_i × score_i × confidence_i) / Σ(weight_i × confidence_i)
final_score = raw_score × abuse_penalty × prior_pull
```

Signals (source-type-calibrated weights)

| Signal               | Blog  | YouTube | PubMed |
|----------------------|-------|---------|--------|
| author_credibility   | 0.18  | 0.22    | 0.18   |
| domain_authority     | 0.30  | 0.15    | 0.10   |
| recency              | 0.20  | 0.23    | 0.15   |
| content_quality      | 0.17  | 0.15    | 0.17   |
| engagement           | 0.05  | 0.15    | 0.35   |
| medical_disclaimer   | 0.10  | 0.10    | 0.05   |

Confidence propagation: each signal returns a (score, confidence) pair.
Signals with weak evidence (e.g. citation count is unknown) contribute
proportionally less rather than injecting a fake default into the sum.

Recency decay: smooth exponential curve (not step function).
  score = 0.95 × 2^(−age_days / halflife)
  half-lives: blog=90d, youtube=150d, pubmed=900d

---

Abuse Prevention (all penalties multiplicative and stacking)

| Vector                         | Detection                             | Penalty |
|--------------------------------|---------------------------------------|---------|
| Anonymous author + health      | author empty/unknown + health keyword | ×0.70   |
| SEO spam URL                   | ≥2 spam-pattern hits in URL           | ×0.75   |
| Keyword stuffing               | top token > 6% of content             | ×0.80   |
| No medical disclaimer          | health keywords + no disclaimer regex | sub = 0 |
| Outdated health content        | >2 years old (non-PubMed)             | ×0.82   |
| Low-authority + health domain  | domain tier < 0.30 + health content  | ×0.75   |
| Anomaly flags                  | per flag from anomaly detector        | ×0.90   |

Penalty floor: 0.05 — no record ever hits absolute zero.

---

Edge Cases Handled

| Case                      | Behaviour                                              |
|---------------------------|--------------------------------------------------------|
| Author missing            | Domain-based fallback; confidence reduced to 0.40      |
| Publish date missing      | Neutral recency 0.30; confidence reduced to 0.30       |
| Transcript unavailable    | Description-only content; quality signal reflects this |
| Multiple authors          | Per-author credibility averaged, not max/min           |
| Non-English content       | Scoring is language-agnostic; language tag is metadata |
| Long articles             | Sentence-aware chunking; overlap preserves RAG context |

---

Key Engineering Decisions

Concurrent ingestion (asyncio.to_thread)
  All three scrapers run in parallel. Wall-clock latency drops from
  sum(scraper latencies) to max(scraper latencies) — typically 3–5×
  faster on a cold run.

Per-host circuit breaker
  After 3 consecutive failures, a host is fast-failed for 60 seconds.
  A broken PubMed endpoint won't block the blog or YouTube scrapers.

Parser fallback chain (blogs)
  content:encoded → RSS description → title-only. Each step isolates
  failures so one broken parser doesn't discard the record entirely.

Source reliability priors (.priors_cache.json)
  Per-domain Bayesian mean is updated after every scored record and
  persisted across runs. Historically reliable domains receive a small
  pull toward their proven prior, noisy domains are pulled down.

RAG-compatible chunks
  Each chunk carries: text, chunk_index, total_chunks, quality_score,
  has_overlap. The quality_score gates what enters the vector store.
  Overlap (28 words by default) prevents semantic units from being
  split across chunk boundaries, improving retrieval precision.

Anomaly detector
  Shannon entropy (repetitive/spun content), link density (SEO farms),
  promotional language density, excessive caps, temporal inconsistency,
  encoding corruption, and uniform paragraph length (templated content)
  — each produces a flag string that propagates to the abuse multiplier.

---

Output format

```json
{
  "source_url":     "https://...",
  "source_type":    "blog | youtube | pubmed",
  "author":         "Author Name",
  "published_date": "YYYY-MM-DD",
  "language":       "en",
  "topic_tags":     ["diabetes", "AI / ML"],
  "anomaly_flags":  [],
  "trust_score":    0.74,
  "sub_scores":     {"author_credibility": 0.72, "domain_authority": 0.95, ...},
  "confidence":     {"author_credibility": 0.65, "domain_authority": 0.92, ...},
  "evidence":       {"author_credibility": {"reason": "named_author"}, ...},
  "abuse_penalty":  1.0,
  "abuse_reasons":  [],
  "prior_used":     null,
  "content_chunks": [
    {"text": "...", "chunk_index": 0, "total_chunks": 4, "quality_score": 0.71, "has_overlap": false},
    {"text": "...", "chunk_index": 1, "total_chunks": 4, "quality_score": 0.68, "has_overlap": true}
  ]
}
```

Requirements: Python 3.9+, YouTube Data API v3 key, NCBI API key (optional)
