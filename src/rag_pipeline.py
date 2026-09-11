"""RAG retrieval pipeline: thin wrapper around the configured vector store.

The vector store backend (local FAISS or cloud Pinecone) is selected via the
``VECTOR_BACKEND`` environment variable and resolved through
``src.vectorstore.get_retriever``, so this module no longer hardcodes a
specific provider.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.exceptions import RetrievalError
from src.logging_config import get_logger
from src.vectorstore import get_retriever

logger = get_logger(__name__)

# Backwards-compatible module-level handle used by legacy scripts/imports.
retriever = get_retriever()


def rag_search(query: str, top_k: int = 5) -> list[dict]:
    """Return the top-k retrieved passages for ``query`` as plain dicts."""
    if not query or not query.strip():
        raise RetrievalError("Query must not be empty")

    try:
        docs = retriever.get_relevant_documents(query, k=top_k)
    except RetrievalError:
        raise
    except Exception as exc:
        logger.exception("Unexpected retrieval failure")
        raise RetrievalError(str(exc)) from exc

    return [
        {
            "score": d.metadata.get("score"),
            "text": d.page_content,
            "source": d.metadata.get("source", ""),
        }
        for d in docs
    ]


if __name__ == "__main__":
    q = "What did the Supreme Court direct regarding stray dogs in NCR?"
    for r in rag_search(q, top_k=5):
        print("Score:", r["score"])
        print("Source:", r["source"])
        print("Text:", r["text"][:500])
        print()
