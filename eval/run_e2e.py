"""Experiment 2: end-to-end pipeline evaluation.

Runs the full RAG + verification pipeline over multiple queries, three
hallucination-injection levels (none/low/high) and five seeds. Ground truth:
sentences drawn from the retrieved evidence are Supported; sentences injected
from the hallucination pool are hallucinated.

Measures, per verifier mode:
  - hallucinated-sentence recall, supported-sentence false-positive rate,
    precision / F1 / accuracy of sentence labels
  - the aggregate hallucination score reported by the system
  - citation correctness (does the top cited chunk actually contain the
    sentence?) and citation over-attachment on injected sentences
  - verification percentage (source credibility) per query
  - end-to-end latency per answer
"""
import json
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from ragverify.generator import HALLUCINATION_POOL
from ragverify.pipeline import build_engine, run_pipeline
from ragverify.verifiers.aggregator import aggregate

EVAL_DIR = Path(__file__).resolve().parent
RESULTS_DIR = EVAL_DIR / "results"

QUERIES = [
    "How does retrieval-augmented generation reduce hallucination?",
    "How is NLI used to check factual consistency of summaries?",
    "What datasets are used to evaluate hallucination in LLMs?",
    "What accuracy did the deep learning system achieve on chest X-ray screening?",
    "What is SelfCheckGPT and how does it detect hallucinations?",
    "Which corpus provides span-level hallucination annotations for RAG responses?",
]
LEVELS = ["none", "low", "high"]
SEEDS = [0, 1, 2, 3, 4]
MODES = ["nli", "heuristics", "combined"]


def gold_of(text):
    return 1 if text in HALLUCINATION_POOL else 0


def source_chunk_of(text, chunks):
    probe = " ".join(text.split()[:8]).lower()
    for c in chunks:
        if probe in c["text"].lower():
            return c["id"]
    return None


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    engine = build_engine()

    rows = []
    verif_pct = {}
    latencies = []
    for qi, q in enumerate(QUERIES):
        for level in LEVELS:
            for seed in SEEDS:
                t0 = time.perf_counter()
                res = run_pipeline(q, engine, k=5, mode="combined",
                                   hallucination_level=level, seed=seed)
                latencies.append(time.perf_counter() - t0)

                if q not in verif_pct:
                    verif_pct[q] = {
                        "verification_percentage": res["verification_percentage"],
                        "credible_sources": res["credible_sources"],
                        "total_sources": res["total_sources"],
                        "titles": [s["title"] for s in res["sources"]],
                    }

                for s in res["sentences"]:
                    row = {
                        "query": q, "level": level, "seed": seed,
                        "text": s["text"], "gold": gold_of(s["text"]),
                        "hallucination_score": res["hallucination_score"],
                    }
                    for mode in MODES:
                        nli_l = s["nli"]["model_label"]
                        lab = aggregate({"label": nli_l}, s["heuristics"], mode)
                        if lab == "Weak":
                            lab = "Unsupported"
                        row[f"pred_{mode}"] = 1 if lab in (
                            "Contradicted", "Unsupported") else 0
                        row[f"label_{mode}"] = lab
                    # citation quality
                    src = source_chunk_of(s["text"], res["chunks"])
                    cites = [c["chunk_id"] for c in s["citations"]]
                    row["source_chunk"] = src
                    row["cited"] = bool(cites)
                    row["cite_hit"] = bool(cites) and src is not None and src in cites
                    row["top1_hit"] = bool(cites) and src is not None and cites[0] == src
                    rows.append(row)
        print(f"  query {qi + 1}/{len(QUERIES)} done ({len(rows)} sentences)")

    # ---------------- metrics ----------------
    def prf(preds, golds):
        tp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 1)
        fp = sum(1 for p, g in zip(preds, golds) if p == 1 and g == 0)
        fn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 1)
        tn = sum(1 for p, g in zip(preds, golds) if p == 0 and g == 0)
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        acc = (tp + tn) / len(preds) if preds else 0.0
        fpr = fp / (fp + tn) if fp + tn else 0.0
        return {k: round(v, 4) for k, v in
                {"precision": prec, "recall": rec, "f1": f1,
                 "accuracy": acc, "fpr": fpr, "tp": tp, "fp": fp,
                 "fn": fn, "tn": tn}.items()}

    per_level_mode = {}
    for level in LEVELS:
        sub = [r for r in rows if r["level"] == level]
        golds = [r["gold"] for r in sub]
        entry = {"n_sentences": len(sub),
                 "n_hallucinated": sum(golds)}
        for mode in MODES:
            preds = [r[f"pred_{mode}"] for r in sub]
            entry[mode] = prf(preds, golds)
        hs = [r["hallucination_score"] for r in sub]
        entry["hallucination_score_mean"] = round(statistics.mean(hs), 1)
        entry["hallucination_score_std"] = round(
            statistics.pstdev(hs), 1) if len(hs) > 1 else 0.0
        per_level_mode[level] = entry

    overall = {}
    golds = [r["gold"] for r in rows]
    for mode in MODES:
        overall[mode] = prf([r[f"pred_{mode}"] for r in rows], golds)

    supported_rows = [r for r in rows if r["gold"] == 0]
    injected_rows = [r for r in rows if r["gold"] == 1]
    citations = {
        "n_supported": len(supported_rows),
        "supported_with_any_citation": round(
            sum(1 for r in supported_rows if r["cited"]) / len(supported_rows), 4),
        "supported_top1_correct": round(
            sum(1 for r in supported_rows if r["top1_hit"]) / len(supported_rows), 4),
        "supported_topk_correct": round(
            sum(1 for r in supported_rows if r["cite_hit"]) / len(supported_rows), 4),
        "n_injected": len(injected_rows),
        "injected_with_citation": round(
            sum(1 for r in injected_rows if r["cited"]) / len(injected_rows), 4)
        if injected_rows else 0.0,
    }

    latency = {
        "mean_s_per_answer": round(statistics.mean(latencies), 2),
        "median_s_per_answer": round(statistics.median(latencies), 2),
        "n_runs": len(latencies),
        "hardware": "CPU (no GPU)",
    }

    out = {"queries": QUERIES, "levels": LEVELS, "seeds": SEEDS,
           "per_level": per_level_mode, "overall": overall,
           "citations": citations, "verification_percentage": verif_pct,
           "latency": latency, "n_rows": len(rows)}
    (RESULTS_DIR / "e2e_results.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8")
    (RESULTS_DIR / "e2e_rows.json").write_text(
        json.dumps(rows, indent=1), encoding="utf-8")

    print(json.dumps({"per_level": {lv: {"hall": e["hallucination_score_mean"],
                                         "n": e["n_sentences"]}
                                    for lv, e in per_level_mode.items()},
                      "overall": overall, "citations": citations,
                      "latency": latency}, indent=1))


if __name__ == "__main__":
    main()
