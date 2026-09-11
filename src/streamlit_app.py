"""AI Legal Document Assistant — production Streamlit UI.

Tabs:
  1. Standard RAG   — single-shot grounded Q&A (legacy behavior, extended with
     history, evidence highlighting, and confidence bars).
  2. Agentic Mode    — Planner -> Researcher -> Reviewer multi-agent pipeline
     with a live, explainable execution trace and task plan.
  3. Knowledge Graph — explore entity relationships extracted from documents.
  4. Multi-Modal     — combined text + chart/diagram image retrieval.
"""
import base64
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd

import streamlit as st

st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️", layout="wide")

if "history" not in st.session_state:
    st.session_state.history = []

st.title("⚖️ AI Legal Document Assistant")
st.caption("RAG + Multi-Agent Planning · LangChain · LangGraph · WatsonX/FAISS · Knowledge Graphs · Multi-Modal Retrieval")

with st.expander("📄 Upload & Index Legal PDFs", expanded=False):
    st.write("Upload one or more legal PDFs to extract, chunk, embed, and index them before asking questions below.")
    uploaded_files = st.file_uploader("Choose PDF file(s)", type=["pdf"], accept_multiple_files=True, key="pdf_uploader")

    if st.button("Process & Index", key="ingest_submit"):
        if not uploaded_files:
            st.warning("Choose at least one PDF file first.")
        else:
            from src.ingest import ingest_pdf_bytes

            summaries = []
            for uploaded in uploaded_files:
                with st.spinner(f"Processing {uploaded.name}..."):
                    try:
                        summaries.append(ingest_pdf_bytes(uploaded.getvalue(), uploaded.name))
                    except Exception as e:
                        st.error(f"Failed to process {uploaded.name}: {e}")

            if summaries:
                st.success(f"Indexed {len(summaries)} document(s). You can now ask questions about them below.")
                st.dataframe(pd.DataFrame(summaries))

tab_rag, tab_agent, tab_graph, tab_mm = st.tabs(
    ["Standard RAG", "Agentic Mode", "Knowledge Graph", "Multi-Modal Search"]
)

# ---------------------------------------------------------------- Standard RAG
with tab_rag:
    with st.sidebar:
        st.header("Standard RAG Options")
        mode = st.selectbox("Response Mode", ["Standard Answer", "Summary Only", "Answer with Evidence"])
        top_k = st.slider("Passages to retrieve", 1, 10, 3)
        highlight = st.checkbox("Highlight evidence", value=True)
        confidence = st.checkbox("Show confidence bars", value=True)
        show_sources = st.checkbox("Show sources", value=True)
        if st.button("Clear History"):
            st.session_state.history = []

    query = st.text_area("Enter your legal question", height=150, key="rag_query")

    if st.button("Submit", key="rag_submit"):
        if not query.strip():
            st.warning("Enter a valid question.")
        else:
            from src.rag_watsonx import generate_answer

            with st.spinner("Retrieving evidence and generating answer..."):
                try:
                    result = generate_answer(query, top_k=top_k)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            answer = result["answer"]
            passages = result["passages"]

            st.subheader("Response")
            if mode == "Summary Only":
                st.write(answer.split(".")[0] + ".")
            else:
                st.write(answer)

            if passages:
                st.markdown("### Retrieved Passages")
                for p in passages:
                    text = p["text"]
                    if highlight:
                        for term in query.split():
                            text = text.replace(term, f"**{term}**")
                    if show_sources:
                        st.markdown(f"Source: `{p['source']}`")
                    st.write(text)
                    if confidence and p.get("score") is not None:
                        st.progress(min(max(float(p["score"]), 0.0), 1.0))
                    st.markdown("---")

            st.session_state.history.append({
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": query,
                "answer": answer,
            })

            b64 = base64.b64encode(answer.encode()).decode()
            href = f'<a href="data:text/plain;base64,{b64}" download="answer.txt">Download answer as .txt</a>'
            st.markdown(href, unsafe_allow_html=True)

    if st.session_state.history:
        st.subheader("History")
        st.dataframe(pd.DataFrame(st.session_state.history))

# ---------------------------------------------------------------- Agentic Mode
with tab_agent:
    st.markdown(
        "Runs the full **Planner → Researcher → Reviewer** multi-agent pipeline "
        "(LangGraph). The Planner decomposes your question into sub-tasks, the "
        "Researcher retrieves grounded evidence per sub-task, and the Reviewer "
        "synthesizes a cited answer — re-running research automatically if the "
        "answer isn't well-grounded (adaptive multi-hop reasoning)."
    )
    agent_query = st.text_area("Enter your legal question", height=150, key="agent_query")

    if st.button("Run Agentic Pipeline", key="agent_submit"):
        if not agent_query.strip():
            st.warning("Enter a valid question.")
        else:
            from src.agents.orchestrator import run_agentic_query

            with st.spinner("Planning, researching, and reviewing..."):
                try:
                    result = run_agentic_query(agent_query)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            st.subheader("Final Answer")
            st.write(result["answer"])
            st.caption(f"Completed in {result['hops']} reasoning hop(s).")

            st.subheader("Task Plan")
            st.dataframe(pd.DataFrame(result["plan"]))

            st.subheader("Citations")
            if result["citations"]:
                st.dataframe(pd.DataFrame(result["citations"]))
            else:
                st.info("No supporting citations were retrieved.")

            with st.expander("Execution Trace (explainability)"):
                for event in result["trace"]:
                    st.markdown(f"**[{event['agent']}]** {event['action']} — {event['detail']}")

# ------------------------------------------------------------- Knowledge Graph
with tab_graph:
    st.markdown(
        "Explore entities (courts, statutes, organizations, dates) extracted "
        "from ingested documents and how they relate to each other."
    )
    entity = st.text_input("Entity to look up (e.g. 'Supreme Court')", key="graph_entity")
    hops = st.slider("Relationship depth (hops)", 1, 4, 1, key="graph_hops")

    if st.button("Query Graph", key="graph_submit"):
        if not entity.strip():
            st.warning("Enter an entity name.")
        else:
            from src.graph.knowledge_graph import query_related_entities

            with st.spinner("Querying knowledge graph..."):
                related = query_related_entities(entity, hops=hops)

            if related:
                st.dataframe(pd.DataFrame(related))
            else:
                st.info("No related entities found. Has the knowledge graph been built via `python -m src.ingest`?")

# ---------------------------------------------------------------- Multi-Modal
with tab_mm:
    st.markdown(
        "Search both document text **and** visually similar charts/diagrams "
        "extracted from uploaded legal PDFs (CLIP-based joint embedding space)."
    )
    mm_query = st.text_area("Enter your search query", height=120, key="mm_query")
    mm_top_k = st.slider("Results per modality", 1, 10, 3, key="mm_top_k")

    if st.button("Search", key="mm_submit"):
        if not mm_query.strip():
            st.warning("Enter a valid query.")
        else:
            from src.multimodal import multimodal_search

            with st.spinner("Searching text and image indexes..."):
                try:
                    result = multimodal_search(mm_query, text_top_k=mm_top_k, image_top_k=mm_top_k)
                except Exception as e:
                    st.error(f"Error: {e}")
                    st.stop()

            st.subheader("Text Passages")
            for p in result["text"]:
                st.markdown(f"Source: `{p['source']}`")
                st.write(p["text"])
                st.markdown("---")

            st.subheader("Related Images")
            if result["images"]:
                cols = st.columns(min(3, len(result["images"])))
                for i, img in enumerate(result["images"]):
                    with cols[i % len(cols)]:
                        if os.path.exists(img["path"]):
                            st.image(img["path"], caption=f"{img['source_pdf']} (p.{img['page']}) score={img['score']:.2f}")
            else:
                st.info("No images indexed yet. Run image extraction + indexing first.")

