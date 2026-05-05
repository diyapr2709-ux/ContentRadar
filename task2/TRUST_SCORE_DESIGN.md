The weighted sum produces a base reliability estimate. The abuse penalty is
a multiplier in `(0, 1]` applied **after** the sum so manipulation cannot be
offset by inflating other signals.

# Five Sub-Scores

| Signal | Range | What it measures |
|---|---|---|
| `author_credibility` | 0–1 | Known org / named / unknown author |
| `citation_count` | 0–1 | Academic citations (PubMed) or download/view proxy |
| `domain_authority` | 0–1 | Tiered whitelist: high / medium / unknown / spam |
| `recency` | 0–1 | Source-type-aware time decay |
| `medical_disclaimer` | 0–1 | Required only for blog/youtube health content |

# Source-Type Calibrated Weights

Each source type uses a different weight set because they're judged differently:

| Weight | Blog | YouTube | PubMed |
|---|---|---|---|
| author_credibility | 0.20 | 0.30 | 0.25 |
| citation_count | 0.10 | 0.10 | **0.35** |
| domain_authority | **0.35** | 0.20 | 0.15 |
| recency | 0.25 | **0.30** | 0.15 |
| medical_disclaimer | 0.10 | 0.10 | 0.10 |

- **Blogs** rely heavily on domain authority — anyone can publish anywhere
- **YouTube** weights channel credibility + recency (channel = author proxy)
- **PubMed** weights citations heaviest — academic standard for trust

---

# 2. Edge Cases Handled

| Edge Case | Handling |
|---|---|
| Author not available | Falls back to domain-based author score (0.10–0.30, not zero) |
| Publish date missing | Recency sub-score = 0.30 (neutral, not penalised to zero) |
| Transcript unavailable | Description used as content; chunking still works on empty input → `[]` |
| Multiple authors | Comma-separated names parsed; per-author scores averaged |
| Non-English content | `langdetect` populates `language` field; scoring is language-agnostic |
| Long articles | Paragraph-aware chunker hard-splits any paragraph > max_words |

All edge cases are demonstrated with runnable examples in `edge_cases.py`.

---

# 3. Abuse Prevention

Penalties are **multiplicative** so they stack and cannot be averaged out by
high sub-scores. The penalty floor is 0.05 — never absolute zero.

| Abuse Vector | Detection | Penalty |
|---|---|---|
| Fake / anonymous authors on health content | author is empty/unknown + health keywords present | × 0.70 |
| SEO spam blogs | URL contains 2+ patterns: `top10`, `best5`, `clickbait`, `secret`, etc. | × 0.75 |
| Misleading medical content | Health keywords present + no disclaimer regex match | sub_score = 0.0 |
| Outdated information | Health content > 2 years old (blogs/youtube only) | × 0.80 |
| Low-authority domain + health | Health content on domain with score < 0.30 | × 0.75 |
| Keyword stuffing | Top word density > 5% across full content | × 0.80 |

All abuse vectors are demonstrated with runnable examples in `abuse_prevention.py`.

---

## 4. Security Factors

| Concern | Mitigation |
|---|---|
| NUL byte injection | `str.translate()` strips `\x00` in O(N) single pass |
| Control char injection | All ASCII control chars except `\t`, `\n`, `\r` stripped |
| Unicode bidi spoofing | RLO/LRO/PDF override codepoints stripped (CVE-2021-42574) |
| Memory bombs | Field length capped at 50KB, response size at 5MB |
| URL scheme abuse | Only `http`/`https` accepted; `javascript:`, `data:`, `file:` rejected |
| ReDoS | All regexes anchored and length-bounded |
| Code injection | No `eval`, `exec`, `os.system`, or shell calls anywhere |
| Secret leakage | All API keys loaded from `.env`; `.env` is gitignored |

---

# 5. Complexity

| Operation | Time | Space |
|---|---|---|
| `clean_text` | O(N) — single C-level translate pass | O(N) for output |
| `auto_tag` | O(T × K) where K ~ 600 (constant) | O(1) |
| `chunk_text` | O(N) single pass | O(C) chunks |
| `calculate` (full score) | O(T) for content checks; O(1) elsewhere | O(1) |

The overall scoring of one record is dominated by the content regex checks
which are linear in content length.