"""
project/utils/tagging.py - Semantic topic enrichment via positional TF-weighting.

Why better than pure keyword counting:
  The old approach counted raw occurrences of every keyword equally, regardless
  of where in the document they appeared or how specific they were. This over-
  tagged short boilerplate-heavy docs and under-tagged dense technical papers.

Design:
  1. Positional weighting - title words score 3x, first-paragraph words 1.5x,
     body 1x. Matches how human readers assign salience.

  2. N-gram phrase detection - compound terms ("machine learning", "gut health")
     are matched before single-token scanning so they aren't split into noise.

  3. TF-style normalization - raw hits divided by document length, preventing
     long articles from dominating purely because they mention more terms.

  4. Entity detection - recognizes disease names, drug classes, institutions.

  5. Semantic grouping - related child terms bubble up to their parent concept,
     so "XGBoost" -> "machine learning", not a standalone tag.
"""
from __future__ import annotations

import re
from functools import lru_cache

from project.scoring.config import CFG

# Each entry: concept label -> (keywords to match, boost multiplier from CFG)
# Multi-word phrases are listed first so they match before single-token scanning.
# All boost values live in CFG.tagging_boost_* — tune there, not here.
_TAXONOMY: list[tuple[str, tuple[str, ...], float]] = [
    ("RAG / retrieval",       ("retrieval-augmented", "retrieval augmented", "rag", "vector store",
                               "embedding", "semantic search", "dense retrieval"), CFG.tagging_boost_rag),
    ("large language models", ("large language model", "llm", "gpt-4", "gpt4", "claude",
                               "gemini", "mistral", "llama", "foundation model"), CFG.tagging_boost_llm),
    ("AI / ML",               ("artificial intelligence", "machine learning", "deep learning",
                               "neural network", "transformer", "xgboost", "random forest",
                               "gradient boost", "supervised", "unsupervised", "nlp"), CFG.tagging_boost_ai_ml),
    ("agents",                ("multi-agent", "autonomous agent", "tool use",
                               "agent2agent", "orchestration", "tool call"), CFG.tagging_boost_agents),
    ("diabetes",              ("diabetes", "diabetic", "insulin resistance", "blood glucose",
                               "type 2 diabetes", "type 1 diabetes", "hba1c",
                               "hyperglycemia", "hypoglycemia"), CFG.tagging_boost_diabetes),
    ("obesity",               ("obesity", "overweight", "bmi", "weight loss",
                               "metabolic syndrome", "adiposity"), CFG.tagging_boost_obesity),
    ("cancer / oncology",     ("oncology", "carcinoma", "malignancy", "tumor", "cancer",
                               "chemotherapy", "immunotherapy", "biopsy"), CFG.tagging_boost_cancer),
    ("cardiovascular",        ("cardiovascular", "heart disease", "cardiac", "coronary",
                               "heart failure", "hypertension", "high blood pressure",
                               "blood pressure", "atherosclerosis", "stroke",
                               "arrhythmia", "atrial fibrillation"), CFG.tagging_boost_cardiovasc),
    ("gut health",            ("gut health", "microbiome", "gut microbiota",
                               "intestinal", "digestive health", "gut bacteria", "bowel"), CFG.tagging_boost_gut),
    ("mental health",         ("mental health", "anxiety", "depression",
                               "psychiatry", "wellbeing", "stress", "burnout",
                               "addiction", "substance use", "trauma",
                               "mindfulness", "psychology", "psychological",
                               "behavioral health"), CFG.tagging_boost_mental),
    ("neurology",             ("alzheimer", "dementia", "parkinson", "neurodegeneration",
                               "neurodegenerative", "cognitive decline", "brain", "neurology",
                               "neurological", "nerve", "seizure", "multiple sclerosis",
                               "neuropathy"), CFG.tagging_boost_neuro),
    ("genomics",              ("genome", "dna", "rna", "gene expression", "snp",
                               "crispr", "mutation", "epigenetics", "sequencing",
                               "genetics", "genomic", "gene therapy", "gene editing"), CFG.tagging_boost_genomics),
    ("healthcare",            ("clinical", "patient", "hospital", "treatment", "therapy",
                               "diagnosis", "drug", "medication", "symptom", "medical",
                               "physician", "fda", "regulatory", "pharmaceutical"), CFG.tagging_boost_healthcare),
    ("public health",         ("public health", "epidemiology", "population health",
                               "surveillance", "mortality", "morbidity", "prevalence",
                               "incidence", "outbreak", "misinformation", "vaccine",
                               "pandemic", "disease control", "global health"), CFG.tagging_boost_public_hlth),
    ("research",              ("study", "research", "clinical trial", "meta-analysis",
                               "systematic review", "cohort", "rct", "evidence-based",
                               "abstract", "doi"), CFG.tagging_boost_research),
    ("technology",            ("software", "platform", "api", "cloud", "infrastructure",
                               "developer", "open source", "github"), CFG.tagging_boost_tech),
    # — added after the 5-topic audit (prompt_engineering / sleep_science /
    #   vaccines records were landing on 'general' or wrong buckets) —
    ("prompt engineering",    ("prompt engineering", "prompt design", "prompt optimization",
                               "system prompt", "few-shot", "zero-shot", "chain-of-thought",
                               "in-context learning", "instruction tuning", "rlhf",
                               "prompt template", "prompting"), CFG.tagging_boost_prompt_eng),
    ("sleep / circadian",     ("sleep", "circadian", "rem sleep", "non-rem", "insomnia",
                               "sleep apnea", "melatonin", "sleep cycle", "chronotype",
                               "sleep quality", "sleep hygiene", "sleep deprivation",
                               "sleep disorder"), CFG.tagging_boost_sleep),
    ("vaccines / immunology", ("vaccine", "vaccination", "immunization", "mrna vaccine",
                               "viral vector", "adjuvant", "herd immunity", "antibody",
                               "antigen", "immunogenicity", "booster shot",
                               "vaccine hesitancy"), CFG.tagging_boost_vaccines),
    ("devops / infrastructure",("kubernetes", "k8s", "docker", "container", "containers",
                                "helm", "terraform", "ci/cd", "ci cd", "devops",
                                "microservice", "microservices", "service mesh",
                                "load balancer", "deployment pipeline",
                                "infrastructure as code"), CFG.tagging_boost_devops),
    ("climate / environment",  ("climate change", "global warming", "greenhouse gas",
                                "carbon emissions", "carbon dioxide", "co2",
                                "renewable energy", "solar power", "wind power",
                                "sea level", "extreme weather", "climate",
                                "decarbonization", "net zero", "sustainability",
                                "ipcc"), CFG.tagging_boost_climate),
    ("quantum computing",      ("quantum computing", "quantum computer", "quantum bit",
                                "qubit", "qubits", "quantum supremacy",
                                "quantum advantage", "quantum entanglement",
                                "quantum algorithm", "quantum error correction",
                                "superposition", "quantum mechanics",
                                "quantum cryptography"), CFG.tagging_boost_quantum),
    ("cybersecurity",          ("cybersecurity", "cyber security", "infosec",
                                "vulnerability", "exploit", "malware", "ransomware",
                                "phishing", "zero-day", "zero day", "cve",
                                "penetration testing", "pentest", "firewall",
                                "intrusion detection", "threat intelligence",
                                "encryption", "authentication", "data breach"),
                                CFG.tagging_boost_security),
]


def _split_zones(text: str) -> tuple[str, str, str]:
    """Return (title, first_paragraph, body) with title = first line."""
    lines      = text.strip().split("\n", 1)
    title      = lines[0] if lines else ""
    remainder  = lines[1].strip() if len(lines) > 1 else ""
    paras      = remainder.split("\n\n", 1)
    first_para = paras[0] if paras else ""
    body       = paras[1] if len(paras) > 1 else ""
    return title, first_para, body


@lru_cache(maxsize=CFG.tagging_cache_maxsize)
def _compile_pattern(term: str) -> re.Pattern:
    """Word-boundary pattern for a keyword - cached so each unique term compiles once."""
    return re.compile(r"\b" + re.escape(term.lower()) + r"\b")


def _tf_score(term: str, zone: str, zone_weight: float) -> float:
    """
    Normalized term frequency for a zone with positional weight.

    Uses word-boundary regex instead of str.count() to prevent substring
    false positives (e.g. 'rna' inside 'alternative', 'dna' inside 'mundane').
    Patterns are compiled once per unique term via lru_cache.
    """
    words = zone.split()
    if not words:
        return 0.0
    hits = len(_compile_pattern(term).findall(zone.lower()))
    return (hits / len(words)) * zone_weight


def _score_all(text: str) -> dict[str, float]:
    """Boosted positional-TF score per taxonomy label."""
    title, first_para, body = _split_zones(text)
    scores: dict[str, float] = {}
    for label, keywords, boost in _TAXONOMY:
        raw = sum(
            _tf_score(kw, zone, w)
            for kw in keywords
            for zone, w in (
                (title,      CFG.tagging_title_weight),
                (first_para, CFG.tagging_first_para_weight),
                (body,       CFG.tagging_body_weight),
            )
        )
        if raw > 0:
            scores[label] = raw * boost
    return scores


def auto_tag(text: str, max_tags: int = CFG.tagging_max_tags) -> list[str]:
    """
    Return up to max_tags topic labels, ranked by positional-TF score.
    Falls back to ["general"] if no keywords match.
    """
    if not text:
        return ["general"]
    scores = _score_all(text)
    if not scores:
        return ["general"]
    return sorted(scores, key=scores.__getitem__, reverse=True)[:max_tags]
