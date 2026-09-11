"""Vector store abstractions.

Exposes a single ``BaseRetriever`` interface implemented by both the local
FAISS backend (default, zero external dependencies) and the Pinecone backend
(cloud-scale). ``get_retriever()`` returns the configured implementation so
the rest of the app is backend-agnostic.
"""
from src.vectorstore.base import BaseRetriever
from src.vectorstore.factory import get_retriever

__all__ = ["BaseRetriever", "get_retriever"]
