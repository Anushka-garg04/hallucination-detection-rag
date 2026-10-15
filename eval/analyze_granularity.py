"""Granularity analysis: NLI scores under chunk vs sentence premises.

For each benchmark sample, scores the hypothesis against
  (a) the raw evidence chunk windows        -> chunk premise
  (b) the constituent sentences of windows  -> sentence premise

Reports mean entailment on Supported samples and mean max-contradiction on
Contradicted samples under both granularities.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ragverify.pipeline import build_engine
from ragverify.sentence_split import split_sentences

EVAL_DIR = Path(__file__).resolve().parent
RES = EVAL_DIR / "results"


class ChunkChunk:
    def __init__(self, cid, doc_id, text):
        self.id = cid
        self.doc_id = doc_id
        self.text = text


def main():
    bench = json.loads((EVAL_DIR / "benchmark.json").read_text(encoding="utf-8"))
    chunks = {cid: ChunkChunk(cid, v["doc_id"], v["text"])
              for cid, v in bench["chunks"].items()}
    eng = build_engine()

    from ragverify.verifiers.nli import NLIVerifier
    v_chunk = NLIVerifier(granularity="chunk")
    v_sent = eng["nli"]

    out = {}
    for label, metric in (("Supported", "entailment"),
                          ("Contradicted", "contradiction")):
        samples = [s for s in bench["samples"] if s["label"] == label]
        acc = {("chunk"): [], ("sent"): []}
        for s in samples:
            ev = [chunks[c] for c in s["evidence"]]
            for tag, v in (("chunk", v_chunk), ("sent", v_sent)):
                r = v.verify_sentence(s["sentence"], ev)
                acc[tag].append(r[metric])
        out[label] = {
            "metric": metric,
            "n": len(samples),
            "chunk_mean": round(sum(acc["chunk"]) / len(samples), 4),
            "sentence_mean": round(sum(acc["sent"]) / len(samples), 4),
        }
        print(label, out[label])

    (RES / "granularity.json").write_text(json.dumps(out, indent=1),
                                           encoding="utf-8")


if __name__ == "__main__":
    main()
