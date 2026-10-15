"""Experiment 1: verifier comparison on the silver benchmark.

Compares verification strategies at the sentence level:
  nli_sent     - DeBERTa NLI, hypothesis vs constituent sentences of each
                 evidence chunk (SummaC-style; production default)
  nli_chunk    - DeBERTa NLI, hypothesis vs raw 80-word chunk window
                 (granularity ablation)
  heuristics   - token overlap / semantic similarity / number & entity match
  combined     - rule-based fusion of NLI + heuristics (production default)

Reports 3-class metrics, binary hallucination-detection metrics (AUROC),
confusion matrices, and per-verifier latency.
"""
import json
import sys
import time
from pathlib import Path

from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragverify.pipeline import build_engine
from ragverify.verifiers.aggregator import aggregate
from ragverify.verifiers.heuristics import heuristic_verify
from ragverify.verifiers.nli import NLIVerifier

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"
HALL_CLASSES = {"Contradicted", "Unsupported"}


class ChunkChunk:
    def __init__(self, cid, doc_id, text):
        self.id = cid
        self.doc_id = doc_id
        self.text = text


def heur_label(heur):
    lab = heur["label"]
    return "Unsupported" if lab == "Weak" else lab


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    bench = json.loads((EVAL_DIR / "benchmark.json").read_text(encoding="utf-8"))
    chunks = {cid: ChunkChunk(cid, v["doc_id"], v["text"])
              for cid, v in bench["chunks"].items()}

    print("Loading models...")
    engine = build_engine()
    nli_sent = engine["nli"]
    nli_chunk = NLIVerifier(granularity="chunk")
    embedder = engine["embedder"]

    rows = []
    t_sent = t_chunk = t_heur = 0.0
    for i, s in enumerate(bench["samples"]):
        ev = [chunks[cid] for cid in s["evidence"]]

        t0 = time.perf_counter()
        n_sent = nli_sent.verify_sentence(s["sentence"], ev)
        t_sent += time.perf_counter() - t0

        t0 = time.perf_counter()
        n_chunk = nli_chunk.verify_sentence(s["sentence"], ev)
        t_chunk += time.perf_counter() - t0

        t0 = time.perf_counter()
        heur = heuristic_verify(s["sentence"], ev, embedder)
        t_heur += time.perf_counter() - t0

        rows.append({
            "id": s["id"], "gold": s["label"],
            "nli_sent_label": n_sent["label"],
            "nli_chunk_label": n_chunk["label"],
            "heur_label": heur_label(heur),
            "comb_label": aggregate({"label": n_sent["label"]}, heur, "combined"),
            "nli_sent_score": 1.0 - n_sent["entailment"],
            "nli_chunk_score": 1.0 - n_chunk["entailment"],
            "heur_score": 1.0 - float(heur["score"]),
        })
        if (i + 1) % 36 == 0:
            print(f"  {i + 1}/{len(bench['samples'])}")

    n = len(rows)
    gold3 = [r["gold"] for r in rows]
    goldb = [1 if r["gold"] in HALL_CLASSES else 0 for r in rows]

    modes = {}
    for mode, key in (("nli_sent", "nli_sent_label"),
                      ("nli_chunk", "nli_chunk_label"),
                      ("heuristics", "heur_label"),
                      ("combined", "comb_label")):
        pred3 = [r[key] for r in rows]
        predb = [1 if r[key] in HALL_CLASSES else 0 for r in rows]
        modes[mode] = {
            "acc_3class": round(accuracy_score(gold3, pred3), 4),
            "macro_f1_3class": round(f1_score(gold3, pred3, average="macro"), 4),
            "precision_3class_macro": round(
                precision_score(gold3, pred3, average="macro", zero_division=0), 4),
            "recall_3class_macro": round(
                recall_score(gold3, pred3, average="macro", zero_division=0), 4),
            "per_class_f1": {
                c: round(f1_score([1 if g == c else 0 for g in gold3],
                                  [1 if p == c else 0 for p in pred3],
                                  zero_division=0), 4)
                for c in ("Supported", "Contradicted", "Unsupported")},
            "per_class_recall": {
                c: round(recall_score([1 if g == c else 0 for g in gold3],
                                      [1 if p == c else 0 for p in pred3],
                                      zero_division=0), 4)
                for c in ("Supported", "Contradicted", "Unsupported")},
            "binary_precision": round(
                precision_score(goldb, predb, zero_division=0), 4),
            "binary_recall": round(recall_score(goldb, predb, zero_division=0), 4),
            "binary_f1": round(f1_score(goldb, predb, zero_division=0), 4),
            "binary_accuracy": round(accuracy_score(goldb, predb), 4),
            "confusion_matrix": confusion_matrix(
                gold3, pred3, labels=["Supported", "Contradicted",
                                      "Unsupported"]).tolist(),
        }

    score_map = {
        "nli_sent": [r["nli_sent_score"] for r in rows],
        "nli_chunk": [r["nli_chunk_score"] for r in rows],
        "heuristics": [r["heur_score"] for r in rows],
        "combined": [0.5 * r["nli_sent_score"] + 0.5 * r["heur_score"]
                     for r in rows],
    }
    for mode, scores in score_map.items():
        modes[mode]["auroc"] = round(roc_auc_score(goldb, scores), 4)

    latency = {
        "nli_sentence_ms": round(1000 * t_sent / n, 1),
        "nli_chunk_ms": round(1000 * t_chunk / n, 1),
        "heuristics_ms": round(1000 * t_heur / n, 1),
        "combined_ms": round(1000 * (t_sent + t_heur) / n, 1),
        "n_sentences": n,
        "hardware": "CPU (PyTorch 2.8, no GPU)",
    }

    out = {"benchmark_counts": bench["counts"],
           "perturbations": bench.get("perturbations", {}),
           "modes": modes, "latency": latency,
           "gold_distribution": {c: gold3.count(c) for c in
                                 ("Supported", "Contradicted", "Unsupported")},
           "hallucinated_rate": round(sum(goldb) / n, 4)}
    (RESULTS_DIR / "verifier_results.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS_DIR / "verifier_rows.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    for mode in modes:
        m = modes[mode]
        print(f"\n[{mode}] 3-class acc={m['acc_3class']} "
              f"macro-F1={m['macro_f1_3class']} | binary P={m['binary_precision']} "
              f"R={m['binary_recall']} F1={m['binary_f1']} AUROC={m['auroc']}")
        print("   per-class F1:", m["per_class_f1"])
        print("   confusion:", m["confusion_matrix"])
    print("\nlatency:", latency)


if __name__ == "__main__":
    main()
