ContentRadar

A topic-driven pipeline that scrapes content from blogs, YouTube, and PubMed, scores each source on a 0 to 1 trust scale, and outputs clean structured JSON. Give it a keyword like diabetes or gut health and it handles the rest — finding sources, filtering noise, and telling you how reliable each one is.

What It Does

ContentRadar runs in two stages. The first stage scrapes 3 blog posts, 2 YouTube videos, and 1 PubMed article for your topic. The second stage reads that raw output and computes a trust score for every record based on author credibility, domain authority, recency, citation count, and medical disclaimer presence.
Both stages write to separate output folders so you can inspect the raw scraped data before scoring touches it.

Trust Score

Every record gets a score between 0 and 1. It is a weighted sum of five signals, with weights calibrated per source type because a peer-reviewed paper and a Medium blog are not judged on the same criteria.
On top of the weighted sum, an abuse penalty multiplier catches anonymous health content, SEO spam URLs, keyword stuffing, missing medical disclaimers on health topics, and outdated information. The penalty is multiplicative so it cannot be offset by inflating other signals.

Requirements
Python 3.9 or higher
YouTube Data API v3 key (free at console.cloud.google.com)
NCBI API key (optional, raises PubMed rate limit from 3 to 10 requests per second)

Setup

git clone https://github.com/your-username/ContentRadar.git
cd ContentRadar
python3 -m pip install -r requirements.txt

Create a .env file in the root:

YOUTUBE_API_KEY=AIzaSy...
NCBI_API_KEY=

Running
python3 main.py --topic "diabetes"
or
Enter topic to scrape: gut health

To verify the scoring engine handles edge cases and abuse vectors correctly:
python3 task2/edge_cases.py
python3 task2/abuse_prevention.py

Output format

{
  "source_url":     "https://...",
  "source_type":    "blog | youtube | pubmed",
  "author":         "Author Name",
  "published_date": "YYYY-MM-DD",
  "language":       "en",
  "region":         "Unknown",
  "topic_tags":     ["healthcare", "machine learning"],
  "trust_score":    0.74,
  "content_chunks": ["paragraph one...", "paragraph two..."]
}
