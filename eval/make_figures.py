"""Generate paper figures from evaluation results (PDF, IEEE two-column)."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import auc, roc_curve

EVAL = Path(__file__).resolve().parent
RES = EVAL / "results"
OUT = EVAL / "figures"
OUT.mkdir(exist_ok=True)

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "font.family": "serif", "mathtext.fontset": "dejavuserif",
    "axes.grid": True, "grid.alpha": 0.3, "grid.linewidth": 0.4,
})

STYLES = {
    "nli_sent":   ("NLI (sentence-level)", "solid", "#1f77b4"),
    "nli_chunk":  ("NLI (chunk-level)", "dashed", "#7f7f7f"),
    "heuristics": ("Heuristics", "dotted", "#d62728"),
    "combined":   ("Combined", "dashdot", "#2ca02c"),
}


def fig_roc():
    rows = json.loads((RES / "verifier_rows.json").read_text(encoding="utf-8"))
    gold = [1 if r["gold"] in ("Contradicted", "Unsupported") else 0 for r in rows]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for mode, (label, ls, color) in STYLES.items():
        scores = {
            "nli_sent": [r["nli_sent_score"] for r in rows],
            "nli_chunk": [r["nli_chunk_score"] for r in rows],
            "heuristics": [r["heur_score"] for r in rows],
            "combined": [0.5 * r["nli_sent_score"] + 0.5 * r["heur_score"]
                         for r in rows],
        }[mode]
        fpr, tpr, _ = roc_curve(gold, scores)
        ax.plot(fpr, tpr, ls=ls, color=color, lw=1.2,
                label=f"{label} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], color="black", lw=0.5, ls=":")
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1.02)
    ax.legend(loc="lower right", framealpha=0.9)
    fig.tight_layout()
    fig.savefig(OUT / "fig_roc.pdf")
    fig.savefig(OUT / "fig_roc.png", dpi=200)
    print("wrote fig_roc")


def fig_f1():
    res = json.loads((RES / "verifier_results.json").read_text(encoding="utf-8"))
    modes = ["nli_sent", "nli_chunk", "heuristics", "combined"]
    labels = ["NLI\n(sentence)", "NLI\n(chunk)", "Heur.", "Combined"]
    classes = ["Supported", "Contradicted", "Unsupported"]
    colors = ["#4c72b0", "#c44e52", "#55a868"]
    x = range(len(modes))
    w = 0.26
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for i, c in enumerate(classes):
        vals = [res["modes"][m]["per_class_f1"][c] for m in modes]
        ax.bar([v + (i - 1) * w for v in x], vals, w, label=c, color=colors[i])
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel("F1 score")
    ax.set_ylim(0, 1.05)
    ax.legend(ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.18),
              frameon=False, columnspacing=1.0, handletextpad=0.4)
    fig.tight_layout()
    fig.savefig(OUT / "fig_f1.pdf")
    fig.savefig(OUT / "fig_f1.png", dpi=200)
    print("wrote fig_f1")


def fig_hscore():
    res = json.loads((RES / "e2e_scores.json").read_text(encoding="utf-8"))
    levels = ["none", "low", "high"]
    means = [res["per_level"][l]["nli"]["mean"] for l in levels]
    stds = [res["per_level"][l]["nli"]["std"] for l in levels]
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    bars = ax.bar(levels, means, yerr=stds, capsize=3,
                  color=["#55a868", "#dd8452", "#c44e52"], width=0.55,
                  error_kw={"lw": 0.8})
    for b, m in zip(bars, means):
        ax.text(b.get_x() + b.get_width() / 2, m + 1.5, f"{m:.1f}",
                ha="center", va="bottom", fontsize=7)
    ax.set_xlabel("Hallucination injection level")
    ax.set_ylabel("Hallucination score (%)")
    ax.set_ylim(0, max(means) + max(stds) + 8)
    fig.tight_layout()
    fig.savefig(OUT / "fig_hscore.pdf")
    fig.savefig(OUT / "fig_hscore.png", dpi=200)
    print("wrote fig_hscore")


def fig_granularity():
    g = json.loads((RES / "granularity.json").read_text(encoding="utf-8"))
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    groups = ["Chunk premise", "Sentence premise"]
    ent = [g["Supported"]["chunk_mean"], g["Supported"]["sentence_mean"]]
    con = [g["Contradicted"]["chunk_mean"], g["Contradicted"]["sentence_mean"]]
    x = range(len(groups))
    w = 0.35
    b1 = ax.bar([i - w / 2 for i in x], ent, w, label="Entailment (supported)",
                color="#4c72b0")
    b2 = ax.bar([i + w / 2 for i in x], con, w,
                label="Max contradiction (perturbed)", color="#c44e52")
    ax.set_xticks(list(x))
    ax.set_xticklabels(groups)
    ax.set_ylabel("Mean NLI score")
    ax.set_ylim(0, 1.12)
    ax.legend(ncol=1, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              frameon=False)
    for bars in (b1, b2):
        for b in bars:
            ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.02,
                    f"{b.get_height():.2f}", ha="center", fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "fig_granularity.pdf")
    fig.savefig(OUT / "fig_granularity.png", dpi=200)
    print("wrote fig_granularity")


if __name__ == "__main__":
    fig_roc()
    fig_f1()
    fig_hscore()
    fig_granularity()
