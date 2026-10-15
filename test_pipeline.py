"""Quick end-to-end test without the UI: python test_pipeline.py"""
import json

from ragverify.pipeline import build_engine, run_pipeline

QUERY = "How does retrieval-augmented generation reduce hallucination?"

print("Building engine (first run loads models)...")
engine = build_engine()
print("Running pipeline...")
res = run_pipeline(QUERY, engine)

print("\nQUERY:", res["query"])
print("\nANSWER:\n", res["answer"])
print("\nSENTENCES:")
for s in res["sentences"]:
    print(f"  [{s['label']:>12}] {s['text'][:80]}...")
    print(f"       NLI={s['nli']}  heur={s['heuristics']['score']}  cites={[c['chunk_id'] for c in s['citations']]}")
print("\nHallucination score:", res["hallucination_score"])
print("Verification percentage:", res["verification_percentage"],
      f"({res['credible_sources']}/{res['total_sources']} credible sources)")
for src in res["sources"]:
    print("  -", src["title"], "| credible:", src["credible"], "|", src["reasons"])
