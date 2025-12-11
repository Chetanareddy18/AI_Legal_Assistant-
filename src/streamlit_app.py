import streamlit as st
from rag_watsonx import generate_answer

st.set_page_config(page_title="AI Legal Assistant", page_icon="⚖️")

st.title("⚖️ AI Legal Assistant (WatsonX + LangChain RAG)")

st.write("Ask any legal question based on your indexed documents.")

query = st.text_area("Enter your legal question:", height=150)

top_k = st.slider("Number of passages to retrieve:", 1, 10, 3)

if st.button("Ask"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Generating answer..."):
            try:
                out = generate_answer(query, top_k=top_k)
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()

        st.subheader("Answer")
        st.write(out["answer"])

        st.subheader("Retrieved Passages")
        for p in out["passages"]:
            st.markdown(f"### Source: `{p['source']}`")
            st.write(p["text"])
            st.markdown(f"**Score:** `{p['score']}`")
            st.write("---")
