"""Offline extractive answer generator (no API, no GPU).

Selects the most query-relevant sentences from the retrieved chunks and can
optionally inject hallucinated sentences so the demo has something to flag.
"""
import re

import numpy as np

HALLUCINATION_POOL = [
    "The RAG method completely eliminates hallucination and never falls back to parametric knowledge, as stated in the original paper.",
    "This approach was first introduced in 2015 by researchers at MIT, and it achieves 99.4 percent accuracy on every standard benchmark.",
    "A 2019 meta-analysis showed that this technique reduces factual error rates by 87 percent in large-scale clinical deployments.",
    "In production, the method reports zero factual errors across more than one million queries, with no unsupported claims ever observed.",
    "Follow-up studies in 2021 demonstrated that the model outperforms all competing systems by a margin of 42 percent on private test sets.",
]

_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def _candidate_sentences(retrieved):
    out = []
    for chunk, chunk_score in retrieved:
        for s in _SENT_SPLIT.split(chunk.text):
            s = s.strip()
            if len(s.split()) < 6 or len(s.split()) > 45:
                continue
            if not s[0].isupper() or s[-1] not in ".!?":
                continue
            if s.lower().startswith(("and ", "but ", "or ", "which ", "that ")):
                continue
            out.append((s, chunk))
    return out


def generate(query: str, retrieved, embedder, hallucination_level: str = "low",
             seed: int = 0) -> str:
    cands = _candidate_sentences(retrieved)
    if not cands:
        return "No relevant evidence was retrieved for this question."

    texts = [c[0] for c in cands]
    sims = embedder.encode(texts) @ embedder.encode(query).ravel()
    order = np.argsort(-sims)

    picked, used_chunks = [], set()
    for i in order:
        s, chunk = cands[i]
        if chunk.id in used_chunks:
            continue
        picked.append((s, float(sims[i])))
        used_chunks.add(chunk.id)
        if len(picked) >= 3:
            break
    picked.sort(key=lambda x: -x[1])
    answer_sentences = [s for s, _ in picked]

    n_inject = {"none": 0, "low": 1, "high": 2}.get(hallucination_level, 1)
    if n_inject:
        inject = [HALLUCINATION_POOL[(seed + i) % len(HALLUCINATION_POOL)]
                  for i in range(n_inject)]
        pos = min(2, len(answer_sentences))
        for j, s in enumerate(inject):
            answer_sentences.insert(pos + j, s)

    return " ".join(answer_sentences)
