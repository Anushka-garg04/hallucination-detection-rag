"""Recompute the system hallucination score per verifier mode from the
saved end-to-end rows (offline; no model runs)."""
import json
import statistics
from collections import defaultdict
from pathlib import Path

RES = Path(__file__).resolve().parent / "results"
MODES = ["nli", "heuristics", "combined"]
LEVELS = ["none", "low", "high"]


def hall_score(labels):
    if not labels:
        return 0.0
    c = labels.count("Contradicted")
    u = labels.count("Unsupported")
    return (2 * c + u) / (2 * len(labels)) * 100


def main():
    rows = json.loads((RES / "e2e_rows.json").read_text(encoding="utf-8"))
    runs = defaultdict(list)
    for r in rows:
        runs[(r["query"], r["level"], r["seed"])].append(r)

    out = {"per_level": {}}
    for level in LEVELS:
        entry = {}
        for mode in MODES:
            scores = [hall_score([g[f"label_{mode}"] for g in grp])
                      for key, grp in runs.items() if key[1] == level]
            entry[mode] = {
                "mean": round(statistics.mean(scores), 1),
                "std": round(statistics.pstdev(scores), 1) if len(scores) > 1 else 0.0,
            }
        out["per_level"][level] = entry

    (RES / "e2e_scores.json").write_text(json.dumps(out, indent=1),
                                         encoding="utf-8")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
