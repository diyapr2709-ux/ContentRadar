"""
project/scoring/config.py - Single source of truth for all scoring parameters.

Every tunable value lives here. No magic numbers exist in trust_score.py,
signals.py, or anomaly.py. To tune scoring behaviour, change this file only.

Scientific references for each value are cited inline. When a parameter
lacks a peer-reviewed source, an empirical justification is given instead.

References:
  [1] Borghol et al. (2012) "On the Homogeneity of YouTube Video View Counts"
  [2] Gao et al. (2023) "Evaluating Veracity of Online Health Claims"
  [3] Metzger & Flanagin (2013) "Credibility and Trust in Online Environments"
  [4] Fiscus (1997) NIST ASR scoring - noise token thresholds
  [5] Kakkonen et al. (2006) - n-gram repetition thresholds for originality
  [6] Brown Corpus analysis - Gini coefficients for natural English word distributions
  [7] Google Search Quality Evaluator Guidelines (2023) - E-E-A-T domain tiers
  [8] NICE/AHA clinical guideline review cycles - content staleness thresholds
  [9] NIH citation percentile data - PubMed citation saturation calibration
"""
from __future__ import annotations

import dataclasses
import os
from datetime import datetime, timezone


# ── Deterministic scoring clock ───────────────────────────────────────────────
# Set CONTENTRADAR_AS_OF to an ISO 8601 timestamp to freeze the scoring clock.
# Every recency/age computation in trust_score and anomaly uses this helper, so
# `same input → same output` becomes a guarantee — eliminating the 10-percentage-
# point drift the brutal audit measured on stored records.
#   CONTENTRADAR_AS_OF=2026-05-10            → midnight UTC on that date
#   CONTENTRADAR_AS_OF=2026-05-10T15:00:00Z  → exact moment
# Default (unset) = real wall clock = unchanged behavior.
def scoring_now() -> datetime:
    raw = os.getenv("CONTENTRADAR_AS_OF", "").strip()
    if raw:
        for fmt in ("%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
            try:
                return datetime.strptime(raw, fmt)
            except ValueError:
                continue
    return datetime.now(timezone.utc).replace(tzinfo=None)


# ══════════════════════════════════════════════════════════════════════════════
# Weight Tables
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class TrustWeights:
    """
    Per-source-type signal weights for the 9-signal trust engine.
    Each instance is validated: all weights must sum to 1.0 within 1e-6.

    Design rationale per source type:
      blog    - domain_authority dominates (provenance = domain-dependent)
      youtube - author_credibility + recency dominate; engagement_authenticity
                gets real weight (manipulation is endemic on YouTube)
      pubmed  - citation_count dominates (peer-review proxy);
                engagement_authenticity = 0.0 (irrelevant for academic papers)
    """
    author_credibility:          float
    citation_count:              float
    domain_authority:            float
    recency:                     float
    medical_disclaimer_presence: float
    metadata_completeness:       float
    content_depth:               float
    engagement_authenticity:     float
    adversarial_risk:            float

    def __post_init__(self) -> None:
        total = sum(dataclasses.asdict(self).values())
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"TrustWeights must sum to 1.0, got {total:.8f}")

    def as_dict(self) -> dict[str, float]:
        return dataclasses.asdict(self)


# ══════════════════════════════════════════════════════════════════════════════
# Main Config
# ══════════════════════════════════════════════════════════════════════════════

@dataclasses.dataclass(frozen=True)
class ScoringConfig:
    """
    All tunable parameters for the ContentRadar trust engine.
    frozen=True: accidental mutation raises TypeError, not a silent bug.
    """

    # ── recency decay (exponential half-life, days) ────────────────────────────
    # Longer half-life = content retains value longer.
    # blog=90d:    health news has ~3-month relevance window (weekly news-cycle content)
    # youtube=150d: educational videos age slower than news; course content stays valid longer
    # pubmed=900d:  foundational research valid 2-3 years (NICE guideline review cycle [8])
    halflife_blog:    float = 90.0
    halflife_youtube: float = 365.0
    halflife_pubmed:  float = 900.0
    recency_ceiling:  float = 0.95   # brand-new content never 1.0 (accounts for update lag)
    recency_floor:    float = 0.05   # minimum for very old content (not zero - archival value)
    recency_missing_score: float = 0.30   # neutral when date absent (not penalised to zero)
    recency_missing_conf:  float = 0.30   # low confidence - genuine uncertainty
    recency_conf:          float = 0.88   # confidence when date is present and parseable

    # ── Bayesian domain prior ─────────────────────────────────────────────────
    # Blends historical per-domain mean trust with current record's score.
    # 0.12 = 12% weight on prior; requires >= prior_min_n records to activate.
    prior_weight:      float = 0.12
    prior_min_n:       int   = 3
    prior_init_score:  float = 0.50   # neutral starting mean for unseen domains

    # ── abuse penalty multipliers (multiplicative stacking - cannot be offset) ─
    # [2]: anonymous health content is 2.3× more likely to contain misinformation.
    # ×0.70 approximates 1 / 1.43 trust deflation.
    penalty_anon_health:          float = 0.70
    penalty_seo_spam_url:         float = 0.75   # SEO bait URL patterns
    penalty_outdated_health:      float = 0.82   # [8] 2-year clinical guideline cycle
    penalty_low_authority_health: float = 0.75   # [7] domain tier mismatch
    penalty_per_anomaly:          float = 0.90   # per anomaly flag (compounds)
    # Conspiracy/misinformation framing in the URL slug itself (one match is sufficient;
    # "big-pharma-doesnt-want-you-to-know" is unambiguous regardless of body content).
    penalty_conspiracy_url:       float = 0.65
    abuse_floor:                  float = 0.05   # floor prevents absolute zero

    # ── Adaptive keyword stuffing (Zipf-corrected top-frequency threshold) ──────
    # threshold = clamp(zipf_k / ln(vocab_size), base_threshold, max_threshold)
    # Zipf's law: the expected top-word frequency in natural text of vocabulary V is
    # approximately zipf_k / ln(V). The adaptive threshold is lenient for small
    # vocabularies (few unique words → naturally high concentration) and tight for
    # large vocabularies (diverse text should have low top-word frequency).
    #
    # Hard cap (max_threshold) prevents the formula from being exploited on tiny
    # vocabularies: vocab=2 gives k/ln(2)=0.866, which is far too lenient for a
    # document that is literally two words repeated. The cap keeps the threshold
    # at 0.45 regardless of vocabulary size below ~5 unique words.
    #
    # After clamping:
    #   vocab=2:   clamp(0.866, 0.08, 0.45) = 0.45  (cap kicks in)
    #   vocab=5:   clamp(0.373, 0.08, 0.45) = 0.373
    #   vocab=10:  clamp(0.261, 0.08, 0.45) = 0.261
    #   vocab=100: clamp(0.130, 0.08, 0.45) = 0.130
    #   vocab=500: clamp(0.097, 0.08, 0.45) = 0.097 (near floor)
    stuffing_base_threshold:   float = 0.08   # minimum (long-document hard floor)
    stuffing_max_threshold:    float = 0.45   # maximum (small-vocabulary hard cap)
    stuffing_zipf_k:           float = 0.60   # empirical Zipf correction factor
    stuffing_min_words:        int   = 80     # minimum words to run stuffing check
    penalty_keyword_stuffing:  float = 0.80   # penalty when stuffing is confirmed

    # ── SEO spam URL patterns ─────────────────────────────────────────────────
    seo_spam_min_hits: int = 2   # minimum distinct URL pattern matches to trigger

    # ── health content staleness ──────────────────────────────────────────────
    # [8] NICE/AHA: clinical guidelines reviewed every 2 years → 730 days.
    outdated_health_days: int = 730

    # ── citation / engagement scaling ─────────────────────────────────────────
    # [9] 200 citations = top ~5% of health papers within 5 years (NIH percentile).
    pubmed_citation_saturation: int   = 200
    citation_pubmed_conf:       float = 0.90   # high confidence: citation count is objective
    # [1] log10(1_000_000) / 6.0 = 1.0 → 1M views saturates. Prevents viral outliers.
    view_log_denominator:       float = 6.0
    citation_view_conf:         float = 0.65   # lower confidence: views ≠ quality
    citation_blog_evidence_saturation: int   = 8      # 8+ evidence phrases → score 1.0
    citation_blog_conf:                float = 0.55   # medium confidence: text proxy not ground truth
    citation_unknown_score:     float = 0.40   # new paper may be excellent at 0 citations
    citation_unknown_conf:      float = 0.30
    engagement_missing_score:   float = 0.35
    engagement_missing_conf:    float = 0.30

    # ── author credibility scores ─────────────────────────────────────────────
    author_score_credible_org:         float = 1.00   # NIH, Mayo Clinic, Harvard…
    author_score_credible_org_conf:    float = 0.95
    author_score_named_high_domain:    float = 0.72   # named author on trusted domain
    author_score_named_low_domain:     float = 0.52   # named author on unknown domain
    author_score_named_conf:           float = 0.65
    author_score_anon_high_domain:     float = 0.35   # no author but trusted domain
    author_score_anon_low_domain:      float = 0.12   # no author, unknown domain
    author_score_anon_conf:            float = 0.40   # proxy confidence is low
    author_domain_mismatch_conf_penalty: float = 0.25  # org claim on spam domain

    # ── domain authority scores ────────────────────────────────────────────────
    domain_score_high:         float = 0.95
    domain_score_high_conf:    float = 0.92
    domain_score_medium:       float = 0.62
    domain_score_medium_conf:  float = 0.80
    domain_score_spam:         float = 0.12
    domain_score_spam_conf:    float = 0.85
    domain_score_unknown:      float = 0.38
    domain_score_unknown_conf: float = 0.50
    domain_score_no_url:       float = 0.10
    domain_score_no_url_conf:  float = 0.90

    # ── medical disclaimer ────────────────────────────────────────────────────
    disclaimer_pubmed_score:   float = 0.50   # peer review substitutes for consumer disclaimer
    disclaimer_pubmed_conf:    float = 0.0    # signal not applicable for pubmed → zero weight (like engagement for non-YT)
    disclaimer_neutral_score:  float = 0.50   # non-health content: signal irrelevant
    disclaimer_neutral_conf:   float = 0.60
    disclaimer_present_score:  float = 1.00
    disclaimer_present_conf:   float = 0.90
    disclaimer_absent_score:   float = 0.00   # health + NO disclaimer = public-health risk floor
    disclaimer_absent_conf:    float = 0.90
    disclaimer_high_auth_implicit_score: float = 0.70  # high-auth domains have editorial policies
    disclaimer_high_auth_implicit_conf:  float = 0.75
    disclaimer_med_auth_implicit_score:  float = 0.35
    disclaimer_med_auth_implicit_conf:   float = 0.55

    # ── metadata completeness ─────────────────────────────────────────────────
    # Score = fields_present / fields_required. Confidence = 1.0 (deterministic).
    # Required fields per source type (field name → expected type check):
    #   present = non-None AND non-empty string AND non-zero for int fields
    metadata_fields_blog:    tuple = ("author", "published_date", "source_url")
    metadata_fields_youtube: tuple = ("author", "published_date", "source_url",
                                      "view_count", "duration_sec")
    metadata_fields_pubmed:  tuple = ("author", "published_date", "source_url",
                                      "pmid", "doi", "journal")
    metadata_missing_conf:   float = 1.0   # deterministic - no uncertainty in field presence

    # ── content depth ─────────────────────────────────────────────────────────
    # Composite: word_count_score × wc_weight + ttr_score × ttr_weight + gini_score × gini_weight
    content_word_count_saturation: int   = 2000   # word count at which score saturates to 1.0
    content_wc_weight:             float = 0.30   # weight of word count component
    content_ttr_weight:            float = 0.40   # weight of type-token ratio (lexical diversity)
    content_gini_weight:           float = 0.30   # weight of inverted Gini (word distribution)
    content_depth_min_words:       int   = 30     # below this: confidence=0.10
    content_depth_conf_thresholds: tuple = (      # (word_count_floor, confidence)
        (300, 0.88),
        (100, 0.65),
        (30,  0.40),
        (0,   0.10),
    )

    # ── engagement authenticity (YouTube) ─────────────────────────────────────
    # [1] Like/view ratio: ln(ratio) ~ Normal(mu, sigma) for organic content.
    # mu=-3.2: ln(0.041) ≈ -3.2; median like/view ratio from Borghol et al.
    # sigma=0.8: empirical log-normal std dev from same study.
    # outlier_z=3.0: 3σ rule → 99.7% of organic content inside boundary.
    engagement_like_view_mu:      float = -3.2
    engagement_like_view_sigma:   float = 0.8
    engagement_outlier_z:         float = 3.0
    engagement_min_views:         int   = 1000    # below this: not enough signal
    engagement_zero_likes_floor:  float = 1e-9    # avoid log(0) when likes=0

    # ── adversarial risk ──────────────────────────────────────────────────────
    # Unicode bidi attack codepoints (CVE-2021-42574 class):
    adversarial_bidi_codepoints: tuple = (
        0x202A, 0x202B, 0x202C, 0x202D, 0x202E,   # LRE, RLE, PDF, LRO, RLO
        0x2066, 0x2067, 0x2068, 0x2069,            # LRI, RLI, FSI, PDI
    )
    adversarial_zwchar_codepoints: tuple = (
        0x200B, 0x200C, 0x200D, 0xFEFF,            # ZWSP, ZWNJ, ZWJ, BOM
    )
    # Prompt injection patterns - empirically derived from LLM red-teaming:
    adversarial_injection_patterns: tuple = (
        r"ignore\s+(all\s+)?previous\s+instructions",
        r"you\s+are\s+now\s+(?:a|an|the)\s+\w",
        r"disregard\s+(the\s+)?above",
        r"new\s+persona\s*:",
        r"system\s*:\s*you\s+(?:are|must|should|will)",
        r"<\s*/?system\s*>",
        r"\[INST\]",
        r"<<SYS>>",
        r"act\s+as\s+(?:a|an)\s+(?:unrestricted|unfiltered|jailbreak)",
    )
    # [5] Kakkonen et al.: >40% repeated 4-grams → non-original / spun content
    adversarial_repetition_threshold: float = 0.40
    adversarial_ngram_size:           int   = 4
    adversarial_bidi_saturation:      int   = 5    # this many bidi chars → bidi_score=0.0
    adversarial_injection_saturation: int   = 2    # this many injection patterns → inj_score=0.0

    # ── transcript quality ────────────────────────────────────────────────────
    # [4] Fiscus (1997): >15% non-speech tokens = poor audio quality in ASR
    transcript_noise_threshold:      float = 0.15
    # Lower than body text repetition (0.40) - ASR legitimately repeats short utterances
    transcript_repetition_threshold: float = 0.25
    transcript_lang_window_size:     int   = 500   # chars per language detection window
    transcript_lang_windows:         int   = 4     # number of windows to sample
    transcript_lang_inconsistency_threshold: float = 0.20

    # ── anomaly detection ─────────────────────────────────────────────────────
    entropy_low_threshold:           float = 4.5    # bits; real articles typically 7-11 bits
    promo_hit_threshold:             int   = 3      # promotional phrases before flagging
    link_density_threshold:          float = 0.20   # fraction of content that is URLs (blog/pubmed)
    # YouTube descriptions legitimately contain many URLs (sponsors, socials,
    # affiliate links, chapters). The 0.20 floor flagged 70% of YouTube records
    # in the 5-topic audit as "high_link_density" false-positively. Use a
    # more permissive threshold for source_type == "youtube".
    link_density_threshold_youtube:  float = 0.40
    caps_density_threshold:          float = 0.08   # fraction of words that are ALL_CAPS
    template_cv_threshold:           float = 0.08   # coefficient of variation for paragraph lengths
    min_words_for_entropy:           int   = 80
    anomaly_thin_content_chars:      int   = 50     # content shorter than this skips heavy checks (entropy/stuffing)
    anomaly_entropy_skip_value:      float = 99.0   # sentinel returned for thin content to bypass entropy check
    anomaly_corruption_ratio:        float = 0.02   # fraction of encoding corruption chars (ratio gate)
    anomaly_corruption_absolute_floor: int = 5      # minimum corrupt chars that trigger flag regardless of ratio
    anomaly_template_min_paras:      int   = 5      # min paragraphs needed to test uniformity
    anomaly_template_min_mean_words: int   = 10     # min mean paragraph words for CV test
    anomaly_min_sentences_for_rep:   int   = 4      # min sentences for repetition check
    # Promotional language: base threshold + density-adaptive scaling.
    # For long articles: threshold = max(base, word_count // divisor) so a 3000-word
    # article needs 6+ hits before flagging (vs 3 hits for a 100-word snippet).
    anomaly_promo_density_divisor:   int   = 500    # words per additional allowed promo hit

    # ── tagging positional zone weights ──────────────────────────────────────
    # Title words are read first and signal salience → 3× body weight.
    # First paragraph sets the framing → 1.5× body weight.
    tagging_title_weight:      float = 3.0
    tagging_first_para_weight: float = 1.5
    tagging_body_weight:       float = 1.0

    # ── DOI / PMID validation ─────────────────────────────────────────────────
    doi_pattern:  str = r"^10\.\d{4,}/\S+"    # CrossRef DOI standard
    pmid_pattern: str = r"^\d{1,8}$"           # Valid PMIDs: 1-8 digit integers

    # ── scraper retrieval limits ──────────────────────────────────────────────
    blog_target_count:         int   = 3
    blog_min_body_words:       int   = 80      # below this, attempt full-article fetch
    blog_min_content_words:    int   = 40      # hard floor after fetch; below = unusable
    blog_min_para_chars:       int   = 40      # minimum paragraph chars in HTML extraction
    blog_min_relevance:        float = 0.003   # keyword density floor (higher = fewer FP)
    blog_max_per_domain:       int   = 2       # max records from any single base domain

    youtube_target_count:      int   = 2
    youtube_max_candidates:    int   = 20
    youtube_min_duration_sec:  int   = 120     # skip videos < 2 min (clips/trailers)
    youtube_long_video_sec:    int   = 300     # threshold for long-video ranking bonus
    youtube_results_per_query: int   = 5       # maxResults per YouTube search call
    youtube_rel_density_scale: float = 100.0   # keyword hit density scaler in composite rank
    youtube_rel_floor:         float = 0.05    # minimum relevance score in composite rank
    youtube_transcript_bonus:  float = 1.2     # composite rank multiplier for transcribed video
    youtube_long_video_bonus:  float = 1.1     # composite rank multiplier for long video
    youtube_min_content_words: int   = 50      # skip video if combined text below this (description-only records with <50w have no RAG value)

    pubmed_target_count:       int   = 1
    pubmed_max_candidates:     int   = 20
    pubmed_min_abstract_words: int   = 30      # skip editorials / case reports with no abstract

    # ── PubMed journal pre-ranking ─────────────────────────────────────────────
    # _journal_impact_score() uses these to rank candidates before efetch round-trips.
    # High-impact journal: j_score = base (2.0). With recency bonus (2025): j_score ≈ 4.5.
    # citation_proxy = max(floor, int(j_score × mult)) → 110-247 for high-impact papers.
    pubmed_journal_impact_base:   float = 2.0
    pubmed_recency_base_year:     int   = 2015   # publications before this get no recency bonus
    pubmed_recency_per_year:      float = 0.25   # score added per year after base_year
    pubmed_citation_proxy_mult:   int   = 55     # j_score × this ≈ estimated citation count
    pubmed_citation_proxy_floor:  int   = 5      # minimum proxy for new/unknown papers

    # ── HTTP request delays (seconds) ────────────────────────────────────────
    # NCBI policy: 10 req/s with API key, 3 req/s without. Conservative headroom added.
    delay_blog_rss:            float = 0.05   # http_client already delays 0.15s; minimal extra
    # Max parallel RSS feed fetches per slug. Each RSS source is a different
    # host, so parallelism doesn't risk per-host rate limits. 8 is enough to
    # cover the typical health-or-tech category set in one wave.
    blog_rss_max_workers:      int   = 8
    delay_youtube_search:      float = 0.10   # YouTube API: quota-based not rate-based
    delay_ncbi_with_key:       float = 0.12
    delay_ncbi_without_key:    float = 0.38

    # ── Signal fallback / neutral values ─────────────────────────────────────
    # Used when evidence is absent or ambiguous; all come from here so they can
    # be tuned together if the neutral baseline needs to shift.
    signal_neutral_score:            float = 0.50
    signal_neutral_conf_low:         float = 0.10   # very thin evidence
    signal_neutral_conf_med:         float = 0.30   # some uncertainty
    adversarial_thin_content_chars:  int   = 50     # signals.py: chars below which skip adversarial
    signal_adv_thin_score:           float = 0.90   # near-clean when content too short to check
    signal_adv_thin_conf:            float = 0.30
    signal_gini_no_content_words:    float = 0.50   # neutral when no content words remain
    signal_engagement_low_conf:      float = 0.10   # insufficient data (views < min or likes=None)
    signal_engagement_outlier_conf:  float = 0.80
    signal_engagement_normal_conf:   float = 0.75
    signal_adv_full_conf:            float = 0.85   # adversarial_risk when content is full-length
    signal_zero_denom_fallback:      float = 0.50   # trust fusion: zero-denominator neutral
    signal_weighted_denom_floor:     float = 1e-9   # below this denominator use fallback score
    signal_depth_min_conf_fallback:  float = 0.10   # content_depth confidence when below all thresholds

    # ── HTTP client ───────────────────────────────────────────────────────────
    # All network parameters here so the retry/backoff/circuit policy is tunable
    # without touching http_client.py. Backoff: attempt k waits base^k ± jitter.
    # Split connect/read: connect timeout is fast-fail (DNS/TCP); read timeout is
    # how long to wait for the first byte once connected. Total worst case per
    # attempt = connect + read = 3+8=11s. With 2 retries = max 22s vs old 45s.
    http_connect_timeout_sec:  float = 3.0    # TCP/DNS fast-fail
    http_read_timeout_sec:     float = 8.0    # first-byte timeout once connected
    http_max_retries:          int   = 2      # 2 attempts total (1 retry); RSS/blogs don't need 3
    http_backoff_base:         float = 1.6    # attempt 0→1.0s, 1→1.6s
    http_backoff_jitter:       float = 0.35   # ± fraction of wait added per attempt
    http_backoff_min_sleep:    float = 0.10   # floor on jittered sleep (no sub-100ms waits)
    http_max_response_bytes:   int   = 5 * 1024 * 1024   # 5 MB response cap
    http_request_delay_sec:    float = 0.15   # polite delay after each successful fetch
    http_cb_threshold:         int   = 3      # failures before circuit opens
    http_cb_timeout_sec:       float = 30.0   # seconds in OPEN state before HALF-OPEN probe
    http_max_429_wait_sec:     float = 120.0  # cap on Retry-After honour
    http_extractor_min_chars:  int   = 100    # try_extractors: min chars to accept a result

    # ── content size limits ───────────────────────────────────────────────────
    # Prevents runaway memory and slow O(n) scoring on enormous scraped pages.
    # Content is truncated at this word count before anomaly detection + scoring.
    blog_max_content_words:    int   = 8000

    # ── chunking ──────────────────────────────────────────────────────────────
    # Controls RAG chunk geometry and quality filtering.
    chunk_max_words:          int   = 180    # max words per chunk before hard-split
    chunk_overlap_words:      int   = 25     # overlap words prepended to next chunk
    chunk_min_words:          int   = 22     # chunks below this are dropped pre-index
    chunk_quality_threshold:  float = 0.22  # quality floor; chunks below this are dropped
    # Fallback quality scores when rag_chunk() returns empty (thin content)
    chunk_fallback_quality_blog:    float = 0.30
    chunk_fallback_quality_youtube: float = 0.35
    chunk_fallback_quality_pubmed:  float = 0.40
    # Chunk quality scorer internals (moved from chunking.py literals)
    chunk_quality_min_words:            int   = 3     # fewer words → quality=0.0 immediately
    chunk_quality_completeness_full:    float = 1.0   # bonus when chunk ends on sentence boundary
    chunk_quality_completeness_partial: float = 0.5   # no sentence boundary detected
    chunk_quality_entropy_max_top_freq: float = 0.10  # content-word top-freq above this → entropy_ok=0
    chunk_quality_w_diversity:          float = 0.5   # weight: type-token ratio
    chunk_quality_w_completeness:       float = 0.3   # weight: sentence-boundary completeness
    chunk_quality_w_entropy:            float = 0.2   # weight: entropy / stuffing signal

    # ── trust tier boundaries ─────────────────────────────────────────────────
    # Human-readable tier assigned alongside the float trust_score in output.
    trust_tier_authoritative: float = 0.75   # ≥ this → AUTHORITATIVE
    trust_tier_credible:      float = 0.55   # ≥ this → CREDIBLE
    trust_tier_uncertain:     float = 0.35   # ≥ this → UNCERTAIN  (else UNRELIABLE)

    # ── hard reject floor ─────────────────────────────────────────────────────
    # Records that score below this after all soft penalties are HARD REJECTED
    # from the final output (not just downranked). Soft penalties reduce the score
    # first; this is the final gate. Set to 0.0 to keep all records regardless.
    hard_reject_trust_floor:   float = 0.20

    # ── HTTP connection pool ──────────────────────────────────────────────────
    http_pool_connections: int = 10   # distinct hosts pooled simultaneously
    http_pool_maxsize:     int = 20   # concurrent connections per host (asyncio.to_thread)

    # ── chunking: fallback for non-dict legacy chunks ─────────────────────────
    chunk_fallback_quality_unknown: float = 0.50   # neutral quality when chunk format is unknown

    # ── sanitizer limits ──────────────────────────────────────────────────────
    sanitizer_max_field_len:       int = 50_000   # hard truncation for any scraped string field
    sanitizer_max_url_len:         int = 2048     # RFC 2616 practical URL length limit
    sanitizer_langdetect_min_chars: int = 20      # skip langdetect on text shorter than this
    sanitizer_langdetect_max_chars: int = 2000    # truncate before langdetect (perf guard)

    # ── schema / record ID ────────────────────────────────────────────────────
    schema_id_hex_len: int = 16   # SHA-256 hexdigest prefix length for record_id

    # ── transcript language window ────────────────────────────────────────────
    transcript_lang_window_min_chars: int = 20   # skip window if shorter than this

    # ── tagging ───────────────────────────────────────────────────────────────
    tagging_max_tags:      int = 5    # maximum topic labels returned by auto_tag()
    tagging_cache_maxsize: int = 512  # lru_cache size for compiled keyword patterns

    # ── priors cache ──────────────────────────────────────────────────────────
    priors_cache_filename: str = ".priors_cache.json"   # relative to project root

    # ── semantic relevance (optional, mirrors langdetect pattern) ─────────────
    # Auto-detected: if `sentence_transformers` is importable, _is_relevant runs
    # an additional cosine-similarity gate on top of the keyword two-gate.
    # Falls back to keyword-only when the package is absent (default state).
    # Threshold tuned for `all-MiniLM-L6-v2`: organic relevant content scores
    # 0.30-0.55; unrelated content typically <0.20. 0.25 is the safe midpoint.
    semantic_model_name:        str   = "sentence-transformers/all-MiniLM-L6-v2"
    semantic_min_similarity:    float = 0.25
    semantic_truncate_chars:    int   = 2000   # cap document chars before encoding

    # ── youtube transcript pre-fetch budget ───────────────────────────────────
    # Rank candidates first WITHOUT transcript bonus, then fetch transcripts only
    # for the top K = youtube_transcript_topk_factor × target_count. Reduces
    # transcript API calls ~5× vs fetching for every viable candidate.
    youtube_transcript_topk_factor: int = 3

    # ── topic expansion (Claude API) ──────────────────────────────────────────
    # Haiku is fast and cheap for structured JSON expansion; max_tokens is
    # sufficient for a list of 10 terms + slugs + two bool/string fields.
    topic_expansion_model:      str   = "claude-haiku-4-5-20251001"
    topic_expansion_max_tokens: int   = 512
    topic_expansion_min_terms:  int   = 3    # fallback if API returns fewer terms
    topic_expansion_min_slugs:  int   = 2    # fallback if API returns fewer slugs
    topic_expansion_timeout_sec: float = 20.0  # cap API hang time
    topic_expansion_max_topic_len: int = 200   # reject pathologically long topic strings
    topic_expansion_cache_filename: str = ".topic_expansion_cache.json"   # disk cache

    # ── NCBI eUtils API endpoints ──────────────────────────────────────────────
    # Canonical NCBI base URLs. Centralised here so changing the API version
    # (e.g. eutils/2.0) only requires a single edit.
    ncbi_esearch_url:  str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    ncbi_esummary_url: str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esummary.fcgi"
    ncbi_efetch_url:   str = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
    ncbi_contact_fallback: str = "research@example.com"  # sent in User-Agent when env var absent
    pubmed_base_url:   str = "https://pubmed.ncbi.nlm.nih.gov/"  # record URL = base + pmid + "/"

    # ── YouTube Data API v3 endpoints ─────────────────────────────────────────
    youtube_search_url:  str = "https://www.googleapis.com/youtube/v3/search"
    youtube_details_url: str = "https://www.googleapis.com/youtube/v3/videos"

    # ── PubMed high-impact journal whitelist ──────────────────────────────────
    # Used for pre-ranking candidates before efetch. NOT the trust score — that
    # is computed by citation_proxy after ingestion. Adding a journal here causes
    # it to be fetched first (higher j_score), not to inflate its trust score.
    pubmed_high_impact_journals: tuple = (
        "nature medicine", "the lancet", "new england journal of medicine",
        "jama", "bmj", "nature", "science", "cell", "npj digital medicine",
        "diabetes care", "diabetologia", "plos medicine", "annals of internal medicine",
        "circulation", "bioinformatics", "nature communications", "nature biotechnology",
        "lancet digital health", "ebiomedicine", "journal of the american medical informatics",
        "plos computational biology", "plos genetics", "plos one",
        "frontiers in medicine", "frontiers in genetics", "frontiers in neuroscience",
        "frontiers in immunology", "frontiers in oncology",
        "scientific reports", "gut", "gastroenterology", "brain", "neurology",
        "journal of clinical oncology", "blood", "cancer research", "cancer cell",
        "nature genetics", "nature neuroscience", "nature reviews",
        "nature reviews cancer", "nature reviews genetics", "nature reviews neuroscience",
        "acta neuropathologica", "journal of infectious diseases",
        "american journal of clinical nutrition", "american journal of epidemiology",
        "international journal of epidemiology", "clinical infectious diseases",
        "ieee transactions on neural networks", "artificial intelligence in medicine",
        "npj precision oncology", "npj genomic medicine",
        "journal of the american college of cardiology", "european heart journal",
        "molecular cell", "developmental cell", "immunity", "cell metabolism",
        "trends in cell biology", "trends in genetics", "trends in neurosciences",
        "cell reports", "cell reports medicine",
    )

    # ── Domain authority tiers ────────────────────────────────────────────────
    # Used by trust_score._domain_signal() and blog_scraper._domain_tier().
    # high_authority_domains: institutional, peer-reviewed, major clinical publishers.
    # med_authority_domains:  quality journalism, known tech/health platforms.
    # Domains not in either set → unknown tier (score 0.38, conf 0.50).
    high_authority_domains: tuple = (
        "pubmed.ncbi.nlm.nih.gov", "ncbi.nlm.nih.gov", "nlm.nih.gov", "who.int", "cdc.gov",
        "nih.gov", "nature.com", "science.org", "nejm.org", "thelancet.com",
        "jamanetwork.com", "bmj.com", "cell.com", "arxiv.org", "ieee.org",
        "acm.org", "mit.edu", "stanford.edu", "harvard.edu", "cam.ac.uk",
        "ox.ac.uk", "openai.com", "anthropic.com", "deepmind.google", "deepmind.com",
        "research.google", "ai.google", "mayoclinic.org", "diabetes.org",
        "hopkinsmedicine.org", "cancer.gov", "heart.org",
        "healthline.com", "medicalnewstoday.com", "webmd.com",
        "statnews.com", "theconversation.com", "sciencedaily.com",
        "eurekalert.org", "medpagetoday.com", "plos.org", "frontiersin.org",
        "technologyreview.com", "newsinhealth.nih.gov", "medlineplus.gov",
        "ama-assn.org", "acpjournals.org", "springer.com", "elsevier.com",
    )
    med_authority_domains: tuple = (
        "medium.com", "towardsdatascience.com", "hackernoon.com", "techcrunch.com",
        "wired.com", "theverge.com", "zdnet.com", "youtube.com", "youtu.be",
        "substack.com", "huggingface.co", "blog.langchain.dev", "everydayhealth.com",
        "theguardian.com", "bbc.com", "bbc.co.uk", "npr.org",
        "scientificamerican.com", "axios.com", "vox.com", "arstechnica.com",
        "verywellhealth.com", "verywell.com", "psychologytoday.com",
        "harvardmagazine.com", "theatlantic.com", "analyticsvidhya.com",
    )

    # ── author / date sentinel strings ────────────────────────────────────────
    # Centralised here so any change propagates to every caller automatically.
    missing_author_sentinels: tuple = ("unknown", "n/a", "", "anonymous")
    missing_date_sentinels:   tuple = ("unknown", "n/a", "")

    # ── anomaly flag routing ───────────────────────────────────────────────────
    # Prefixes of flags that the abuse penalty loop must SKIP because trust_score
    # checks them directly via Zipf-adaptive logic (no double-counting).
    anomaly_direct_check_prefixes: tuple = ("keyword_stuffing",)

    # ── HTTP status code policy ────────────────────────────────────────────────
    # Client errors: URL/auth problem, host is fine - do not retry, do not open circuit.
    http_no_retry_client_codes: tuple = (400, 401, 403, 404, 410, 422, 451)
    # Transient server errors: retry with backoff, advance failure counter.
    # 429 is handled separately (Retry-After header support).
    http_retry_server_codes:    tuple = (503, 504)
    # YouTube API additionally includes 500/502 in its retry set.
    http_yt_retry_server_codes: tuple = (500, 502, 503, 504)

    # ── YouTube search parameters ──────────────────────────────────────────────
    youtube_safe_search:          str   = "moderate"
    youtube_relevance_language:   str   = "en"
    # Suffixes appended to topic terms to bias toward educational content.
    # "explained" → higher transcript rate; "overview"/"guide" → broader recall.
    youtube_query_suffixes:       tuple = ("explained", "overview", "guide")
    # Preferred languages for transcript fetching (tried in order).
    youtube_transcript_languages: tuple = ("en", "en-US", "en-GB")

    # ── tagging taxonomy boost weights ─────────────────────────────────────────
    # Higher = stronger signal for that category. Precision categories (RAG,
    # diabetes) boosted above 1.0; noisy / broad categories below 1.0.
    tagging_boost_rag:         float = 1.3
    tagging_boost_llm:         float = 1.2
    tagging_boost_agents:      float = 1.1
    tagging_boost_ai_ml:       float = 1.0
    tagging_boost_diabetes:    float = 1.2
    tagging_boost_obesity:     float = 1.1
    tagging_boost_cancer:      float = 1.1
    tagging_boost_cardiovasc:  float = 1.0
    tagging_boost_gut:         float = 1.1
    tagging_boost_mental:      float = 1.0
    tagging_boost_neuro:       float = 1.0
    tagging_boost_genomics:    float = 1.1
    tagging_boost_healthcare:  float = 0.9
    tagging_boost_public_hlth: float = 1.0
    tagging_boost_research:    float = 0.8
    tagging_boost_tech:        float = 0.7
    # Added after the 5-topic audit revealed taxonomy gaps. 'general' tag was
    # firing for prompt-engineering and sleep-science articles whose subject
    # matter isn't covered by the existing buckets.
    tagging_boost_prompt_eng:  float = 1.2
    tagging_boost_sleep:       float = 1.1
    tagging_boost_vaccines:    float = 1.1
    tagging_boost_devops:      float = 1.1
    tagging_boost_climate:     float = 1.1
    tagging_boost_quantum:     float = 1.2
    tagging_boost_security:    float = 1.1

    # ── terminal display / log formatting ────────────────────────────────────
    # All display widths and truncation lengths live here so they can be tuned
    # without touching scraper or main.py code.
    display_bar_narrow:           int = 64   # inner pipeline separator width
    display_bar_wide:             int = 68   # outer results separator width
    display_author_max_chars:     int = 35   # author name truncation in terminal
    display_tags_count:           int = 3    # max tags shown per record in output
    display_preview_chars:        int = 160  # content preview chars in results
    display_url_tail_chars:       int = 50   # URL tail chars in rejection / index messages
    display_url_fetch_chars:      int = 55   # URL tail chars in fetch-upgrade log lines
    display_url_accept_chars:     int = 60   # URL tail chars in accepted-record log lines
    display_pubmed_doi_chars:     int = 30   # DOI truncation in pubmed accept log
    display_pubmed_journal_chars: int = 50   # journal name truncation in pubmed accept log
    display_audit_bar_width:      int = 70   # build_index audit table separator width
    display_audit_inner_width:    int = 64   # build_index audit table inner separator

    def __post_init__(self) -> None:
        if self.abuse_floor <= 0.0:
            raise ValueError(f"abuse_floor must be > 0.0, got {self.abuse_floor}")
        if not (0.0 <= self.hard_reject_trust_floor < 1.0):
            raise ValueError(f"hard_reject_trust_floor must be in [0.0, 1.0), got {self.hard_reject_trust_floor}")
        if self.prior_weight < 0.0 or self.prior_weight > 1.0:
            raise ValueError(f"prior_weight must be in [0.0, 1.0], got {self.prior_weight}")

    def halflife(self, source_type: str) -> float:
        return {
            "blog":    self.halflife_blog,
            "youtube": self.halflife_youtube,
            "pubmed":  self.halflife_pubmed,
        }.get(source_type, self.halflife_blog)

    def metadata_fields(self, source_type: str) -> tuple:
        return {
            "blog":    self.metadata_fields_blog,
            "youtube": self.metadata_fields_youtube,
            "pubmed":  self.metadata_fields_pubmed,
        }.get(source_type, self.metadata_fields_blog)


# ══════════════════════════════════════════════════════════════════════════════
# Weight Tables (9 signals, must sum to 1.0)
# ══════════════════════════════════════════════════════════════════════════════

WEIGHTS: dict[str, TrustWeights] = {
    "blog": TrustWeights(
        author_credibility          = 0.18,
        citation_count              = 0.08,
        domain_authority            = 0.27,
        recency                     = 0.20,
        medical_disclaimer_presence = 0.08,
        metadata_completeness       = 0.05,
        content_depth               = 0.08,
        engagement_authenticity     = 0.02,
        adversarial_risk            = 0.04,
    ),
    "youtube": TrustWeights(
        author_credibility          = 0.20,
        citation_count              = 0.08,
        domain_authority            = 0.15,
        recency                     = 0.20,
        medical_disclaimer_presence = 0.08,
        metadata_completeness       = 0.04,
        content_depth               = 0.06,
        engagement_authenticity     = 0.12,
        adversarial_risk            = 0.07,
    ),
    "pubmed": TrustWeights(
        author_credibility          = 0.20,
        citation_count              = 0.30,
        domain_authority            = 0.12,
        recency                     = 0.12,
        medical_disclaimer_presence = 0.08,
        metadata_completeness       = 0.06,
        content_depth               = 0.08,
        engagement_authenticity     = 0.00,
        adversarial_risk            = 0.04,
    ),
}

# Singleton - imported by all scoring modules
CFG = ScoringConfig()
