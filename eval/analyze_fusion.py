"""Offline fusion ablation: recompute original vs corrected fusion rules
from the saved per-sample rows (no model inference required)."""
import json
from pathlib import Path

from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)

RES = Path(__file__).resolve().parent / "results"
HALL = ("Contradicted", "Unsupported")


def old_fusion(nli_label, heur_label, heur_score):
    if nli_label == "Contradicted" and heur_score < 0.80:
        return "Contradicted"
    if nli_label == "Supported" and heur_score >= 0.25:
        return "Supported"
    if nli_label == "Supported":
        return "Supported"
    if heur_label == "Supported":
        return "Supported"
    if nli_label == "Contradicted":
        return "Contradicted"
    return "Unsupported"


def new_fusion(nli_label, heur_label, heur_score):
    if nli_label == "Contradicted":
        return "Contradicted"
    if nli_label == "Supported":
        return "Supported"
    if heur_label == "Supported":
        return "Supported"
    return "Unsupported"


def metrics(rows, pred_fn, nli_key):
    gold3 = [r["gold"] for r in rows]
    pred3 = [pred_fn(r) for r in rows]
    goldb = [1 if g in HALL else 0 for g in gold3]
    predb = [1 if p in HALL else 0 for p in pred3]
    return {
        "acc": round(accuracy_score(gold3, pred3), 4),
        "macro_f1": round(f1_score(gold3, pred3, average="macro"), 4),
        "f1_contradicted": round(f1_score(
            [1 if g == "Contradicted" else 0 for g in gold3],
            [1 if p == "Contradicted" else 0 for p in pred3],
            zero_division=0), 4),
        "binary_precision": round(precision_score(goldb, predb, zero_division=0), 4),
        "binary_recall": round(recall_score(goldb, predb, zero_division=0), 4),
        "binary_f1": round(f1_score(goldb, predb, zero_division=0), 4),
    }


def main():
    rows = json.loads((RES / "verifier_rows.json").read_text(encoding="utf-8"))
    res = json.loads((RES / "verifier_results.json").read_text(encoding="utf-8"))

    out = {}
    for nli_key in ("nli_sent_label", "nli_chunk_label"):
        for name, fn in (("original_fusion", old_fusion),
                         ("corrected_fusion", new_fusion)):
            out[f"{name}+{nli_key}"] = metrics(
                rows,
                lambda r, fn=fn, k=nli_key: fn(r[k], r["heur_label"],
                                               1.0 - r["heur_score"]),
                nli_key)

    # sanity: corrected fusion with sentence NLI must match run_eval's combined
    sanity = out["corrected_fusion+nli_sent_label"]
    expected = res["modes"]["combined"]
    assert abs(sanity["macro_f1"] - expected["macro_f1_3class"]) < 0.001, \
        (sanity, expected)
    print("sanity check passed: corrected+sentence == combined")

    # how many contradictions each fusion drops, sentence premises
    con = [r for r in rows if r["gold"] == "Contradicted"]
    for name, fn in (("original", old_fusion), ("corrected", new_fusion)):
        lost = sum(1 for r in con
                   if fn(r["nli_sent_label"], r["heur_label"],
                         1.0 - r["heur_score"]) != "Contradicted")
        print(f"{name} fusion: {lost}/{len(con)} contradictions lost "
              f"(sentence premises)")

    (RES / "fusion_ablation.json").write_text(json.dumps(out, indent=1),
                                               encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
