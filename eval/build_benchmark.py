"""Build a silver-labeled sentence-level faithfulness benchmark from the corpus.

Labels (by construction, relative to the supplied evidence chunks):
  Supported    - verbatim sentence extracted from its own source document.
  Contradicted - controlled perturbation of a corpus sentence (negation,
                 number substitution, year substitution) that conflicts with
                 the evidence text.
  Unsupported  - a sentence drawn from a different document than the evidence
                 (cross-document distractor), filtered for low overlap.

Each sample carries the evidence chunk set used by the verifier: the gold
chunk plus distractor chunks (simulating top-k retrieval with the relevant
chunk present).
"""
import json
import random
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragverify.chunking import chunk_all
from ragverify.corpus import load_documents
from ragverify.sentence_split import split_sentences

EVAL_DIR = Path(__file__).resolve().parent
SEED = 13
N_DISTRACTORS = 3

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was",
         "were", "has", "have", "had", "not", "but", "its", "into", "than",
         "then", "also", "can", "may", "one", "two", "such", "when", "which",
         "their", "they", "them", "these", "those", "been", "being", "over",
         "only", "very", "more", "most", "some", "any", "each", "both",
         "between", "using", "used", "use", "it", "on", "in", "of", "to",
         "as", "by", "at", "an", "be", "is", "or", "if", "about"}
_NUMBER = re.compile(r"\b\d+(?:\.\d+)?\b")
_YEAR = re.compile(r"\b(?:19|20)\d{2}\b")
_MODALS = re.compile(r"\b(is|are|was|were|has|have|had|can|could|may|might|"
                     r"will|would|do|does|did)\b", re.IGNORECASE)


def content_tokens(text):
    return {t.lower() for t in _TOKEN.findall(text)
            if t.lower() not in _STOP and len(t) > 2}


def probe(sentence, n=8):
    return " ".join(sentence.split()[:n]).lower()


def find_gold_chunk(sentence, chunks):
    """Return a chunk that fully contains the sentence (exact substring)."""
    for c in chunks:
        if sentence in c.text:
            return c
    return None


def overlap_with(sentence, chunks):
    st = content_tokens(sentence)
    if not st:
        return 0.0
    best = 0.0
    for c in chunks:
        ct = content_tokens(c.text)
        if ct:
            best = max(best, len(st & ct) / len(st))
    return best


def distractors_for(gold, chunks, rng, banned_texts=()):
    pool = [c for c in chunks
            if c.doc_id != gold.doc_id and c.id != gold.id
            and all(probe(t) not in c.text.lower() for t in banned_texts)]
    take = rng.sample(pool, min(N_DISTRACTORS, len(pool)))
    return [gold] + take


# ---------------------------------------------------------------- perturbations
def negation(sentence, gold_text, rng):
    m = _MODALS.search(sentence)
    if not m or m.group(0).lower() not in gold_text.lower():
        return None
    return (sentence[:m.end()] + " not" + sentence[m.end():],
            "negation", m.group(1))


def number_swap(sentence, gold_text, rng):
    for m in _NUMBER.finditer(sentence):
        old = m.group(0)
        if old in gold_text:
            if "." in old:
                cands = ["7.5", "12.3", "34.8", "56.1", "89.6"]
            else:
                val = int(old)
                cands = [str(val + 3), str(val + 11), str(val * 3 + 1),
                         str(max(1, val // 2))]
            for new in cands:
                if new != old and new not in gold_text:
                    return (sentence.replace(old, new, 1), "number", old)
    return None


def year_swap(sentence, gold_text, rng):
    for m in _YEAR.finditer(sentence):
        old = m.group(0)
        if old in gold_text:
            for new in ("1998", "2004", "2011", "2017", "2026"):
                if new not in gold_text:
                    return (sentence.replace(old, new, 1), "year", old)
    return None


_ANTONYMS = {
    "outperforms": "underperforms", "outperformed": "underperformed",
    "increase": "decrease", "increases": "decreases", "increased": "decreased",
    "reduce": "increase", "reduces": "increases", "reduced": "increased",
    "improve": "worsen", "improves": "worsens", "improved": "worsened",
    "higher": "lower", "greater": "smaller", "larger": "smaller",
    "stronger": "weaker", "strong": "weak", "best": "worst",
    "accurate": "inaccurate", "accurately": "inaccurately",
    "succeeds": "fails", "succeed": "fail", "succeeded": "failed",
    "supported": "unsupported", "consistent": "inconsistent",
    "success": "failure", "effective": "ineffective",
}


def antonym_swap(sentence, gold_text, rng):
    for word, ant in _ANTONYMS.items():
        pat = re.compile(r"\b" + word + r"\b")
        if pat.search(sentence) and re.search(r"\b" + word + r"\b", gold_text):
            return (pat.sub(ant, sentence, count=1), "antonym", word)
    return None


OPS = [negation, number_swap, year_swap, antonym_swap]


def build():
    rng = random.Random(SEED)
    docs = load_documents()
    chunks = chunk_all(docs)

    gold_sentences = []
    for d in docs:
        for s in split_sentences(d.text):
            if 6 <= len(s.split()) <= 45 and find_gold_chunk(s, chunks):
                gold_sentences.append((s, d))
    rng.shuffle(gold_sentences)

    n_target = len(gold_sentences)
    samples = []
    seen = set()

    # ---- Supported -----------------------------------------------------
    for s, d in gold_sentences:
        seen.add(s)
        gold = find_gold_chunk(s, chunks)
        samples.append({
            "id": f"sup_{len([x for x in samples if x['label'] == 'Supported'])}",
            "label": "Supported",
            "sentence": s,
            "evidence": [c.id for c in
                         distractors_for(gold, chunks, rng, banned_texts=[s])],
            "gold_chunk": gold.id,
            "doc_id": d.id,
            "perturbation": None,
        })

    # ---- Contradicted --------------------------------------------------
    contras = []
    for s, d in gold_sentences:
        gold = find_gold_chunk(s, chunks)
        order = OPS[:]
        rng.shuffle(order)
        for op in order:
            res = op(s, gold.text, rng)
            if not res:
                continue
            new_s, kind, _ = res
            if new_s in seen:
                break
            seen.add(new_s)
            contras.append({
                "label": "Contradicted",
                "sentence": new_s,
                "evidence": [c.id for c in
                             distractors_for(gold, chunks, rng, banned_texts=[s])],
                "gold_chunk": gold.id,
                "doc_id": d.id,
                "perturbation": kind,
            })
            break
    rng.shuffle(contras)
    contras = contras[:n_target]
    for i, c in enumerate(contras):
        c["id"] = f"con_{i}"
        samples.append(c)

    # ---- Unsupported (cross-document) ----------------------------------
    unsup = []
    for s, d in gold_sentences:
        for other in docs:
            if other.id == d.id:
                continue
            other_chunks = [c for c in chunks if c.doc_id == other.id]
            ev = rng.choice(other_chunks)
            if overlap_with(s, other_chunks) > 0.50:
                continue
            ev_set = distractors_for(ev, chunks, rng, banned_texts=[s])
            if overlap_with(s, ev_set) > 0.50:
                continue
            unsup.append({
                "label": "Unsupported",
                "sentence": s,
                "evidence": [c.id for c in ev_set],
                "gold_chunk": ev.id,
                "doc_id": other.id,
                "perturbation": None,
            })
            break
    rng.shuffle(unsup)
    unsup = unsup[:n_target]
    for i, u in enumerate(unsup):
        u["id"] = f"uns_{i}"
        samples.append(u)

    counts = {}
    for s in samples:
        counts[s["label"]] = counts.get(s["label"], 0) + 1

    pert_counts = {}
    for s in samples:
        if s["perturbation"]:
            pert_counts[s["perturbation"]] = pert_counts.get(s["perturbation"], 0) + 1

    chunks_out = {c.id: {"doc_id": c.doc_id, "text": c.text} for c in chunks}
    docs_out = {d.id: {"title": d.title, "authors": d.authors,
                       "venue": d.venue, "year": d.year,
                       "scopus_indexed": d.scopus_indexed} for d in docs}

    out = {"seed": SEED, "n_distractors": N_DISTRACTORS, "counts": counts,
           "perturbations": pert_counts, "samples": samples,
           "chunks": chunks_out, "documents": docs_out}
    (EVAL_DIR / "benchmark.json").write_text(json.dumps(out, indent=1),
                                              encoding="utf-8")
    print("counts:", counts, "total:", len(samples))
    print("perturbations:", pert_counts)


if __name__ == "__main__":
    build()
