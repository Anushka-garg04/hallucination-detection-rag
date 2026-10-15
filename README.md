# Hallucination Detection & Verification in LLMs — Final Year Project

End-to-end working prototype: **RAG pipeline** + **sentence-level verification** (DeBERTa NLI + heuristics) + **citations** + **Verification Percentage** (source credibility).

## Quick start

```bat
pip install -r requirements.txt
python warmup.py          # one-time: downloads the models (~2-5 min)
streamlit run app.py      # or just double-click run.bat
```

Browser opens automatically at http://localhost:8501

## Tests (no browser needed)

```bat
python test_pipeline.py   # CLI end-to-end test
python smoke_test.py      # headless UI test (Streamlit AppTest)
```

Expected output from `test_pipeline.py`: 4 answer sentences labelled (3 Supported / 1 Contradicted), **Hallucination score 25%**, **Verification percentage 33.3%** (1 of 3 retrieved sources credible).

## What the demo shows

1. **RAG pipeline** — question → FAISS retrieval over 8 research-paper chunks → offline extractive answer generator (no API key needed).
2. **Sentence-level verification** — every answer sentence is labelled `Supported` / `Contradicted` / `Unsupported` by:
   - `cross-encoder/nli-deberta-v3-base` (premise = evidence chunk, hypothesis = answer sentence)
   - heuristics: token overlap, number match, entity match, semantic similarity
3. **Citations** — each sentence shows which retrieved chunk(s) support it (with cosine score).
4. **Verification Percentage** — `(credible sources / total retrieved sources) × 100`
   Credible = author NOT in `data/blacklist.csv` AND venue Scopus-indexed.
   Example: 10 papers used, 5 blacklisted/non-Scopus → **50%**.
5. **Hallucination score** — `(2×Contradicted + Unsupported) / (2×Total sentences) × 100`

## Demo controls (sidebar)

- **Verifier mode**: Combined / NLI only / Heuristics only (ablation for the presentation)
- **Hallucination injection**: none / low / high — simulates an LLM adding unsupported claims so the highlighting has something to show
- **Top-k retrieval**: number of evidence chunks

## Sample queries (buttons in the UI)

- "How does retrieval-augmented generation reduce hallucination?"
- "How is NLI used to check factual consistency of summaries?"
- "What datasets are used to evaluate hallucination in LLMs?"
- "What accuracy did the deep learning system achieve on chest X-ray screening?"

## Project structure

```
prototype/
  app.py                  Streamlit UI
  run.bat                 one-click launch (Windows)
  warmup.py               one-time model downloader
  test_pipeline.py        CLI end-to-end test (no UI)
  smoke_test.py           headless UI test (no browser)
  requirements.txt        Python dependencies
  data/corpus.json        8 papers with metadata (venue, scopus flag)
  data/blacklist.csv      blacklisted authors
  eval/                   benchmark, metrics, figures (F1, ROC, ablations)
    benchmark.json        60+ labelled QA instances
    run_eval.py           verifier comparison (NLI / heuristics / combined)
    run_e2e.py            end-to-end pipeline evaluation
    figures/              fig_f1, fig_roc, fig_granularity, fig_hscore
    results/              raw metric results
  ragverify/
    pipeline.py           orchestrator
    corpus.py             documents + credibility
    chunking.py           word-window chunker
    embedder.py           MiniLM embeddings
    retriever.py          FAISS / cosine index
    generator.py          offline extractive generator (+ hallucination injector)
    sentence_split.py     regex sentence splitter
    credibility.py        verification percentage
    verifiers/nli.py      DeBERTa NLI verifier
    verifiers/heuristics.py  token/number/entity/similarity signals
    verifiers/aggregator.py  label fusion + hallucination score
```

## Models

| Model | Purpose |
|---|---|
| `cross-encoder/nli-deberta-v3-base` | sentence-level NLI verification |
| `sentence-transformers/all-MiniLM-L6-v2` | retrieval embeddings + semantic similarity |
| FAISS (cpu) | vector index |
