import numpy as np

try:
    import faiss
    FAISS_AVAILABLE = True
except Exception:
    FAISS_AVAILABLE = False


class VectorIndex:
    """Cosine-similarity index over normalized vectors (FAISS when available)."""

    def __init__(self, dim: int):
        self.dim = dim
        self.vectors = None
        self._faiss = faiss.IndexFlatIP(dim) if FAISS_AVAILABLE else None

    def add(self, vectors: np.ndarray):
        self.vectors = np.asarray(vectors, dtype="float32")
        if self._faiss is not None:
            self._faiss.add(self.vectors)

    def search(self, query_vec: np.ndarray, k: int):
        q = np.asarray(query_vec, dtype="float32").reshape(1, -1)
        k = min(k, len(self.vectors))
        if self._faiss is not None and self._faiss.ntotal:
            scores, idx = self._faiss.search(q, k)
        else:
            sims = (self.vectors @ q.T).ravel()
            idx = np.argsort(-sims)[:k]
            scores = sims[idx][None, :]
        return [int(i) for i in idx[0]], [float(s) for s in scores[0]]


class Retriever:
    def __init__(self, chunks, embedder):
        self.chunks = chunks
        self.embedder = embedder
        self.index = VectorIndex(dim=384)
        self.index.add(embedder.encode([c.text for c in chunks]))

    def retrieve(self, query: str, k: int = 5):
        qv = self.embedder.encode(query)
        idx, scores = self.index.search(qv, k)
        return [(self.chunks[i], float(s)) for i, s in zip(idx, scores)]
