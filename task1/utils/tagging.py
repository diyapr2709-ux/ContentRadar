"""
task1/utils/tagging.py — keyword-frequency topic tagger.
Time O(T × K), Space O(1).
"""

from __future__ import annotations

_TOPICS: dict[str, tuple[str, ...]] = {
    "AI":               ("artificial intelligence", " ai ", "machine learning",
                         "deep learning", "neural network", "nlp", "llm",
                         "gpt", "transformer", "generative"),
    "agents":           ("agent", "agentic", "multi-agent", "autonomous",
                         "orchestrat", "tool use", "agent2agent", "a2a"),
    "RAG":              ("retrieval", " rag ", "retrieval-augmented",
                         "vector store", "embedding", "semantic search"),
    "diabetes":         ("diabetes", "diabetic", "insulin", "glucose",
                         "glycemic", "type 2", "type 1", "hba1c"),
    "obesity":          ("obesity", "overweight", "bmi", "weight loss",
                         "metabolic syndrome"),
    "healthcare":       ("health", "medical", "clinical", "patient",
                         "disease", "treatment", "therapy", "drug",
                         "diagnosis", "hospital"),
    "public health":    ("public health", "epidemiology", "population health",
                         "prevention", "surveillance", "mortality"),
    "machine learning": ("machine learning", " ml ", "regression",
                         "classification", "supervised", "unsupervised",
                         "random forest", "xgboost", "gradient boost"),
    "research":         ("study", "research", "paper", "journal", "findings",
                         "methodology", "results", "abstract", "doi"),
    "technology":       ("software", "platform", "api", "cloud",
                         "developer", "infrastructure"),
}


def auto_tag(text: str, max_tags: int = 5) -> list[str]:
    if not text:
        return ["general"]
    low = text.lower()
    scores: dict[str, int] = {}
    for topic, keywords in _TOPICS.items():
        hits = sum(low.count(kw) for kw in keywords)
        if hits:
            scores[topic] = hits
    if not scores:
        return ["general"]
    return sorted(scores, key=scores.__getitem__, reverse=True)[:max_tags]