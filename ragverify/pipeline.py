"""End-to-end pipeline: retrieve -> generate -> verify -> cite -> score."""
from .chunking import chunk_all
from .corpus import load_documents
from .credibility import verification_percentage
from .embedder import Embedder
from .generator import generate
from .retriever import Retriever
from .sentence_split import split_sentences
from .verifiers.aggregator import aggregate, hallucination_score
from .verifiers.heuristics import heuristic_verify
from .verifiers.nli import NLIVerifier

CITATION_THRESHOLD = 0.35


def build_engine():
    docs = load_documents()
    chunks = chunk_all(docs)
    embedder = Embedder()
    retriever = Retriever(chunks, embedder)
    nli = NLIVerifier()
    docs_by_id = {d.id: d for d in docs}
    return {"docs": docs, "docs_by_id": docs_by_id, "chunks": chunks,
            "embedder": embedder, "retriever": retriever, "nli": nli}


def _citations(sentence, chunks, embedder):
    s_vec = embedder.encode(sentence)
    c_vecs = embedder.encode([c.text for c in chunks])
    sims = (c_vecs @ s_vec.T).ravel()
    order = [i for i in sims.argsort()[::-1] if sims[i] >= CITATION_THRESHOLD]
    return [(chunks[i], float(sims[i])) for i in order[:2]]


def run_pipeline(query: str, engine: dict, k: int = 5,
                 mode: str = "nli", hallucination_level: str = "low",
                 seed: int = 0) -> dict:
    retriever, embedder, nli = engine["retriever"], engine["embedder"], engine["nli"]

    retrieved = retriever.retrieve(query, k)
    chunks = [c for c, _ in retrieved]
    scores = {c.id: s for c, s in retrieved}

    answer = generate(query, retrieved, embedder, hallucination_level, seed)
    sentences = split_sentences(answer)

    results = []
    for s in sentences:
        nli_res = nli.verify_sentence(s, chunks)
        heur = heuristic_verify(s, chunks, embedder)
        label = aggregate(nli_res, heur, mode)
        cites = _citations(s, chunks, embedder)
        results.append({
            "text": s,
            "label": label,
            "nli": {"entailment": round(nli_res["entailment"], 4),
                    "contradiction": round(nli_res["contradiction"], 4),
                    "neutral": round(max(nli_res["neutral"], 0.0), 4),
                    "model_label": nli_res["label"]},
            "heuristics": heur,
            "citations": [{"chunk_id": c.id, "doc_id": c.doc_id,
                           "score": round(sc, 4)} for c, sc in cites],
        })

    labels = [r["label"] for r in results]
    h_score = hallucination_score(labels)

    doc_ids = list(dict.fromkeys(c.doc_id for c in chunks))
    docs = [engine["docs_by_id"][d] for d in doc_ids]
    pct, credible, total, sources = verification_percentage(docs)

    return {
        "query": query,
        "answer": answer,
        "sentences": results,
        "counts": {
            "Supported": labels.count("Supported"),
            "Contradicted": labels.count("Contradicted"),
            "Unsupported": labels.count("Unsupported"),
            "total": len(labels),
        },
        "hallucination_score": h_score,
        "verification_percentage": pct,
        "credible_sources": credible,
        "total_sources": total,
        "sources": sources,
        "chunks": [{"id": c.id, "doc_id": c.doc_id, "text": c.text,
                    "score": round(scores[c.id], 4)} for c in chunks],
        "mode": mode,
    }
