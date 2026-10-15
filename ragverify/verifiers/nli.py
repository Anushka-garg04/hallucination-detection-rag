import re

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

DEFAULT_MODEL = "cross-encoder/nli-deberta-v3-base"

CONTRA_THRESHOLD = 0.70
ENTAIL_THRESHOLD = 0.55
MAX_CANDIDATES_PER_CHUNK = 2

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_STOP = {"the", "and", "for", "that", "with", "this", "from", "are", "was",
         "were", "has", "have", "had", "not", "but", "its", "into", "than",
         "then", "also", "can", "may", "one", "two", "such", "when", "which",
         "their", "they", "them", "these", "those", "been", "being", "over",
         "only", "very", "more", "most", "some", "any", "each", "both",
         "between", "using", "used", "use", "it", "on", "in", "of", "to",
         "as", "by", "at", "an", "be", "is", "or", "if", "about"}
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z0-9\"'])")


def _content_tokens(text):
    return {t.lower() for t in _TOKEN.findall(text)
            if t.lower() not in _STOP and len(t) > 2}


class NLIVerifier:
    """DeBERTa NLI model: reads (evidence, answer sentence) pairs.

    granularity='sentence' scores the hypothesis against the constituent
    sentences of each evidence chunk (SummaC-style); 'chunk' scores it
    against the raw window. Sentence granularity is the default because
    long windows dilute entailment: verbatim-contained sentences receive
    mean entailment 0.396 under chunk premises vs 0.991 under sentence
    premises, and contradiction detection nearly collapses at chunk level.
    """

    def __init__(self, model_name: str = DEFAULT_MODEL,
                 granularity: str = "sentence"):
        assert granularity in ("sentence", "chunk")
        self.granularity = granularity
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
        self.model.eval()
        labels = {int(k): v.lower() for k, v in self.model.config.id2label.items()}
        self.idx = {"contradiction": None, "neutral": None, "entailment": None}
        for i, lab in labels.items():
            for key in self.idx:
                if key in lab:
                    self.idx[key] = i
        self._sent_cache = {}

    def _chunk_sentences(self, chunk) -> list:
        if chunk.id not in self._sent_cache:
            self._sent_cache[chunk.id] = [
                s.strip() for s in _SENT_SPLIT.split(chunk.text.strip())
                if len(s.split()) >= 4]
        return self._sent_cache[chunk.id]

    @torch.no_grad()
    def _pair_scores(self, pairs, batch_size=8) -> list:
        out = []
        for i in range(0, len(pairs), batch_size):
            batch = pairs[i:i + batch_size]
            enc = self.tokenizer([p[0] for p in batch], [p[1] for p in batch],
                                 padding=True, truncation=True, max_length=256,
                                 return_tensors="pt")
            logits = self.model(**enc).logits
            probs = torch.softmax(logits, dim=-1).tolist()
            for p in probs:
                out.append({
                    "contradiction": p[self.idx["contradiction"]],
                    "neutral": p[self.idx["neutral"]],
                    "entailment": p[self.idx["entailment"]],
                })
        return out

    def _build_pairs(self, sentence, chunks):
        """Candidate (premise, hypothesis) pairs with their chunk ids."""
        if self.granularity == "chunk":
            return [((c.text, sentence), c.id) for c in chunks]
        hyp = _content_tokens(sentence)
        pairs = []
        for c in chunks:
            cands = []
            for s in self._chunk_sentences(c):
                ov = len(hyp & _content_tokens(s))
                if ov:
                    cands.append((ov, s))
            cands.sort(key=lambda x: -x[0])
            pairs.extend(((s, sentence), c.id)
                         for _, s in cands[:MAX_CANDIDATES_PER_CHUNK])
        return pairs

    def verify_sentence(self, sentence: str, chunks: list) -> dict:
        if not chunks:
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0,
                    "label": "Unsupported", "best_chunk": None}

        built = self._build_pairs(sentence, chunks)
        if not built:
            # no lexical overlap with any evidence sentence
            return {"entailment": 0.0, "contradiction": 0.0, "neutral": 1.0,
                    "label": "Unsupported", "best_chunk": None}

        pairs = [p for p, _ in built]
        scores = self._pair_scores(pairs)
        owner = [cid for _, cid in built]

        i_ent = max(range(len(scores)), key=lambda i: scores[i]["entailment"])
        i_con = max(range(len(scores)), key=lambda i: scores[i]["contradiction"])
        primary, max_con = scores[i_ent], scores[i_con]["contradiction"]

        if primary["entailment"] >= ENTAIL_THRESHOLD:
            label, best = "Supported", owner[i_ent]
        elif max_con >= CONTRA_THRESHOLD:
            label, best = "Contradicted", owner[i_con]
        else:
            label, best = "Unsupported", owner[i_ent]

        return {"entailment": primary["entailment"],
                "contradiction": max_con,
                "neutral": primary["neutral"],
                "label": label, "best_chunk": best}
