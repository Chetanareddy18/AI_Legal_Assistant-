import tempfile

from src.vectorstore.faiss_store import FaissRetriever


def _make_store():
    tmp_dir = tempfile.mkdtemp()
    return FaissRetriever(index_dir=tmp_dir, embedding_model="sentence-transformers/all-MiniLM-L6-v2")


def test_empty_store_returns_no_documents():
    store = _make_store()
    assert store.get_relevant_documents("anything", k=3) == []


def test_add_and_search_returns_most_relevant_document():
    store = _make_store()
    store.add_documents(
        texts=[
            "The Supreme Court ordered relocation of stray dogs to shelters.",
            "Section 302 IPC prescribes punishment for murder.",
        ],
        sources=["case1.pdf", "ipc.pdf"],
    )

    results = store.get_relevant_documents("What did the court say about stray dogs?", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "case1.pdf"
    assert 0.0 <= results[0].metadata["score"] <= 1.0001


def test_persists_and_reloads_index():
    tmp_dir = tempfile.mkdtemp()
    store = FaissRetriever(index_dir=tmp_dir, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    store.add_documents(["Some legal clause about liability."], ["doc.pdf"])

    reloaded = FaissRetriever(index_dir=tmp_dir, embedding_model="sentence-transformers/all-MiniLM-L6-v2")
    results = reloaded.get_relevant_documents("liability clause", k=1)

    assert len(results) == 1
    assert results[0].metadata["source"] == "doc.pdf"
