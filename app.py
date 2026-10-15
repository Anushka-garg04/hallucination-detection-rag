# Final Year Project
# Hallucination Detection and Mitigation in LLMs
# Streamlit demo for the project presentation

import streamlit as st
import pandas as pd

from ragverify.pipeline import build_engine, run_pipeline

st.set_page_config(page_title="Hallucination Detection", layout="wide")

# Load the models (DeBERTa + MiniLM) once and reuse them
@st.cache_resource
def load_models():
    return build_engine()


# ---------------- Sidebar ----------------
st.sidebar.header("Project Settings")

mode_choice = st.sidebar.radio(
    "Verification Mode",
    ["NLI only", "Combined (NLI + Heuristics)", "Heuristics only"],
    help="NLI only is the best-performing verifier in our benchmark "
         "(macro-F1 0.89); combined adds heuristic signals.",
)

injection_choice = st.sidebar.selectbox(
    "Simulated LLM Hallucination",
    ["low", "none", "high"],
    help="Injects fake claims in the generated answer so we can show detection",
)

k_choice = st.sidebar.slider("Number of Retrieved Chunks (top-k)", 3, 8, 5)

st.sidebar.markdown("---")
st.sidebar.markdown("**Formulas used**")
st.sidebar.write(
    "Verification Percentage = (Credible Sources / Total Sources) x 100\n\n"
    "Hallucination Score = (2 x Contradicted + Unsupported) / (2 x Total Sentences) x 100"
)
st.sidebar.markdown("---")
st.sidebar.write("Models used: DeBERTa-v3-base (NLI), MiniLM-L6-v2 (Embeddings), FAISS (Retrieval)")

# ---------------- Header ----------------
st.title("Hallucination Detection and Mitigation in LLMs")
st.caption("Retrieval Augmented Generation (RAG) with sentence-level verification, citations and verification percentage")

with st.expander("About this project"):
    st.write(
        "This system answers a question using a RAG pipeline (retrieval + answer generation), "
        "then checks every sentence of the answer against the retrieved documents. "
        "Each sentence is labelled as Supported, Contradicted or Unsupported using a "
        "DeBERTa NLI model and simple heuristic methods (token overlap, number match, "
        "entity match, semantic similarity). "
        "The system also gives a citation for each sentence and calculates a "
        "Verification Percentage based on how many of the retrieved sources are credible "
        "(authors not blacklisted and venue Scopus-indexed)."
    )

# Sample questions
st.write("**Sample questions:**")
sample_questions = [
    "How does retrieval-augmented generation reduce hallucination?",
    "How is NLI used to check factual consistency of summaries?",
    "What datasets are used to evaluate hallucination in LLMs?",
    "What accuracy did the deep learning system achieve on chest X-ray screening?",
]
cols = st.columns(len(sample_questions))
for i, q in enumerate(sample_questions):
    if cols[i].button("Q" + str(i + 1), help=q):
        st.session_state["query_box"] = q

if "query_box" not in st.session_state:
    st.session_state["query_box"] = sample_questions[0]

query = st.text_input("Enter your question", key="query_box")

# ---------------- Main ----------------
if st.button("Generate and Verify", type="primary"):

    if query.strip() == "":
        st.warning("Please enter a question first")
        st.stop()

    engine = load_models()

    if mode_choice.startswith("Combined"):
        verify_mode = "combined"
    elif mode_choice.startswith("NLI"):
        verify_mode = "nli"
    else:
        verify_mode = "heuristics"

    with st.spinner("Retrieving documents, generating answer and verifying each sentence..."):
        result = run_pipeline(
            query,
            engine,
            k=k_choice,
            mode=verify_mode,
            hallucination_level=injection_choice,
        )

    # ---- 1. Main numbers ----
    st.markdown("## Results")
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Verification Percentage", str(result["verification_percentage"]) + "%")
    col2.metric("Hallucination Score", str(result["hallucination_score"]) + "%")
    col3.metric(
        "Supported Sentences",
        str(result["counts"]["Supported"]) + " / " + str(result["counts"]["total"]),
    )
    col4.metric(
        "Problem Sentences",
        str(result["counts"]["Contradicted"] + result["counts"]["Unsupported"]),
    )
    st.write(
        "Verification Percentage is calculated as "
        + str(result["credible_sources"]) + " credible sources out of "
        + str(result["total_sources"]) + " retrieved sources = "
        + str(result["verification_percentage"]) + "%"
    )

    # ---- 2. Generated answer sentence by sentence ----
    st.markdown("## 1. Generated Answer (sentence-level verification)")
    for sent in result["sentences"]:
        citation_text = ", ".join(c["chunk_id"] for c in sent["citations"])
        if citation_text == "":
            citation_text = "no supporting chunk found"
        line = sent["text"] + "   [Citation: " + citation_text + "]"

        if sent["label"] == "Supported":
            st.success("**Supported** | " + line)
        elif sent["label"] == "Contradicted":
            st.error("**Contradicted** | " + line)
        else:
            st.warning("**Unsupported** | " + line)

    # ---- 3. Source credibility table ----
    st.markdown("## 2. Source Credibility Check (Verification Percentage)")
    source_rows = []
    for s in result["sources"]:
        status = "Credible" if s["credible"] else "; ".join(s["reasons"])
        source_rows.append({
            "Title": s["title"],
            "Authors": ", ".join(s["authors"]),
            "Venue": s["venue"],
            "Year": s["year"],
            "Scopus Indexed": "Yes" if s["scopus_indexed"] else "No",
            "Status": status,
        })
    st.dataframe(pd.DataFrame(source_rows))

    st.write(
        "Verification Percentage = " + str(result["credible_sources"]) + " / "
        + str(result["total_sources"]) + " x 100 = **"
        + str(result["verification_percentage"]) + "%**"
    )
    if result["verification_percentage"] < 50:
        st.error("More than half of the sources used are NOT credible. Do not trust this answer.")
    elif result["verification_percentage"] < 100:
        st.warning("Some sources failed the credibility check.")
    else:
        st.success("All retrieved sources passed the credibility check.")

    # ---- 4. Sentence level score table ----
    st.markdown("## 3. Sentence-Level Verification Details")
    detail_rows = []
    for i, sent in enumerate(result["sentences"], 1):
        detail_rows.append({
            "#": i,
            "Sentence": sent["text"],
            "Label": sent["label"],
            "NLI Entailment": sent["nli"]["entailment"],
            "NLI Contradiction": sent["nli"]["contradiction"],
            "Heuristic Score": sent["heuristics"]["score"],
            "Token Overlap": sent["heuristics"]["overlap"],
            "Semantic Similarity": sent["heuristics"]["semantic"],
            "Number Match": sent["heuristics"]["number_match"],
            "Entity Match": sent["heuristics"]["entity_match"],
            "Citation": ", ".join(c["chunk_id"] for c in sent["citations"]),
        })
    st.dataframe(pd.DataFrame(detail_rows))

    # ---- 5. Bar chart of labels ----
    st.markdown("## 4. Label Distribution")
    st.bar_chart({
        "Supported": result["counts"]["Supported"],
        "Contradicted": result["counts"]["Contradicted"],
        "Unsupported": result["counts"]["Unsupported"],
    })

    # ---- 6. Retrieved evidence ----
    st.markdown("## 5. Retrieved Evidence Chunks")
    for chunk in result["chunks"]:
        with st.expander(chunk["id"] + "  (similarity = " + str(chunk["score"]) + ")"):
            st.write(chunk["text"])

else:
    st.info("Enter a question and click **Generate and Verify** to run the pipeline.")
