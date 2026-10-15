"""Lightweight, interpretable verification signals."""
import re

import numpy as np

STOPWORDS = {
    "the", "and", "for", "that", "with", "this", "from", "are", "was", "were",
    "has", "have", "had", "not", "but", "its", "into", "than", "then", "also",
    "can", "may", "one", "two", "such", "when", "which", "their", "they",
    "them", "these", "those", "been", "being", "because", "been", "over",
    "only", "very", "more", "most", "some", "any", "each", "both", "between",
    "using", "used", "use", "it", "its", "on", "in", "of", "to", "as", "by",
    "at", "an", "be", "is", "or", "if", "from", "about",
}

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\s?(?:percent|%)?\b")
_ENTITY = re.compile(r"\b([A-Z][a-zA-Z]+(?:\s[A-Z][a-zA-Z]+)*)\b")

SUPPORT_THRESHOLD = 0.55
WEAK_THRESHOLD = 0.30


def _content_tokens(text: str) -> set:
    return {t.lower() for t in _TOKEN.findall(text)
            if t.lower() not in STOPWORDS and len(t) > 2}


def _clip01(x: float) -> float:
    return float(min(1.0, max(0.0, x)))


def heuristic_verify(sentence: str, chunks: list, embedder) -> dict:
    if not chunks:
        return {"score": 0.0, "overlap": 0.0, "semantic": 0.0,
                "number_match": 0.0, "entity_match": 0.0, "label": "Unsupported"}

    s_tokens = _content_tokens(sentence)
    s_numbers = [n.replace(" ", "") for n in _NUMBER.findall(sentence)]

    body = sentence.split(" ", 1)[1] if " " in sentence else sentence
    s_entities = {m.group(1) for m in _ENTITY.finditer(body)}

    best = {"overlap": 0.0, "semantic": 0.0}
    all_chunk_text = " ".join(c.text for c in chunks)

    for c in chunks:
        c_tokens = _content_tokens(c.text)
        if s_tokens:
            inter = len(s_tokens & c_tokens)
            best["overlap"] = max(best["overlap"], inter / len(s_tokens))

    s_vec = embedder.encode(sentence)
    c_vecs = embedder.encode([c.text for c in chunks])
    best["semantic"] = float((c_vecs @ s_vec.T).max())

    if s_numbers:
        hits = sum(1 for n in s_numbers if n in all_chunk_text.replace(" ", "")
                   or n.rstrip("%") in all_chunk_text)
        number_match = hits / len(s_numbers)
    else:
        number_match = 1.0

    if s_entities:
        ent_hits = sum(1 for e in s_entities if e in all_chunk_text)
        entity_match = ent_hits / len(s_entities)
    else:
        entity_match = 1.0

    semantic = _clip01(best["semantic"])
    score = (0.30 * best["overlap"] + 0.30 * semantic +
             0.20 * number_match + 0.20 * entity_match)

    if score >= SUPPORT_THRESHOLD:
        label = "Supported"
    elif score >= WEAK_THRESHOLD:
        label = "Weak"
    else:
        label = "Unsupported"

    return {"score": round(score, 4), "overlap": round(best["overlap"], 4),
            "semantic": round(semantic, 4),
            "number_match": round(number_match, 4),
            "entity_match": round(entity_match, 4), "label": label}
