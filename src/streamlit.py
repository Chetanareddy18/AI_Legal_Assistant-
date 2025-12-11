import streamlit as st
import pandas as pd
import base64
from datetime import datetime
from rag_watsonx import generate_answer

st.set_page_config(page_title="AI Legal Assistant", layout="wide")

st.title("AI Legal Assistant")
st.markdown("""
This assistant analyzes legal documents using a Retrieval-Augmented Generation (RAG) system.  
Enter a question, apply filters, and review extracted evidence.
""")

if "history" not in st.session_state:
    st.session_state.history = []

with st.sidebar:
    st.header("Options")
    mode = st.selectbox("Response Mode", ["Standard Answer", "Summary Only", "Answer with Evidence"])
    top_k = st.slider("Passages to retrieve", 1, 10, 3)
    highlight = st.checkbox("Highlight evidence")
    confidence = st.checkbox("Show confidence bars")
    show_sources = st.checkbox("Show sources")
    upload = st.file_uploader("Upload PDF files", type=["pdf"])
    st.markdown("---")
    if st.button("Clear History"):
        st.session_state.history = []

query = st.text_area("Enter your legal question", height=180)

if st.button("Submit"):
    if not query.strip():
        st.warning("Enter a valid question.")
    else:
        with st.spinner("Processing request..."):
            result = generate_answer(query, top_k=top_k)

        answer = result["answer"]
        passages = result["passages"]

        st.subheader("Response")
        if mode == "Summary Only":
            st.write(answer.split(".")[0] + ".")
        elif mode == "Answer with Evidence":
            st.write(answer)
            st.write("Relevant Evidence")
        else:
            st.write(answer)

        if passages:
            st.markdown("### Retrieved Passages")
            for p in passages:
                text = p["text"]

                if highlight:
                    key_terms = query.split()
                    for term in key_terms:
                        text = text.replace(term, f"**{term}**")

                st.markdown(f"Source: `{p['source']}`") if show_sources else None
                st.write(text)

                if confidence:
                    st.progress(min(max(p["score"], 0.0), 1.0))

                st.markdown("---")

        history_entry = {
            "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "query": query,
            "answer": answer
        }
        st.session_state.history.append(history_entry)

        st.subheader("Download Answer")
        b64 = base64.b64encode(answer.encode()).decode()
        href = f'<a href="data:text/plain;base64,{b64}" download="answer.txt">Download as .txt</a>'
        st.markdown(href, unsafe_allow_html=True)

if st.session_state.history:
    st.subheader("History")
    df = pd.DataFrame(st.session_state.history)
    st.dataframe(df)
