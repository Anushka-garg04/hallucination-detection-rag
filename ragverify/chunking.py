from dataclasses import dataclass


@dataclass
class Chunk:
    id: str
    doc_id: str
    text: str
    index: int


def chunk_document(doc, words_per_chunk=80, overlap=20) -> list:
    words = doc.text.split()
    chunks = []
    step = max(1, words_per_chunk - overlap)
    i, n = 0, 0
    while i < len(words):
        piece = " ".join(words[i:i + words_per_chunk])
        if piece.strip():
            chunks.append(Chunk(id=f"{doc.id}#c{n}", doc_id=doc.id, text=piece, index=n))
            n += 1
        i += step
    return chunks


def chunk_all(docs, words_per_chunk=80, overlap=20) -> list:
    out = []
    for d in docs:
        out.extend(chunk_document(d, words_per_chunk, overlap))
    return out
