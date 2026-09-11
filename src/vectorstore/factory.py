"""Factory that returns the configured retriever backend as a singleton."""
import os
from functools import lru_cache

from src.config import get_settings
from src.exceptions import ConfigurationError
from src.logging_config import get_logger
from src.vectorstore.base import BaseRetriever

logger = get_logger(__name__)

# Project root (parent of src/), used to anchor relative paths regardless of
# the process's current working directory (uvicorn, streamlit, pytest, etc.
# may all be launched from different cwds).
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@lru_cache
def get_retriever() -> BaseRetriever:
    settings = get_settings()

    if settings.vector_backend == "faiss":
        from src.vectorstore.faiss_store import FaissRetriever

        index_dir = settings.faiss_index_dir
        if not os.path.isabs(index_dir):
            index_dir = os.path.join(_PROJECT_ROOT, index_dir)

        logger.info("Using local FAISS vector store at %s", index_dir)
        return FaissRetriever(index_dir, settings.embedding_model)

    if settings.vector_backend == "pinecone":
        from src.vectorstore.pinecone_store import PineconeRetriever

        if not settings.pinecone_api_key:
            raise ConfigurationError("PINECONE_API_KEY must be set when VECTOR_BACKEND=pinecone")

        logger.info("Using Pinecone vector store index '%s'", settings.pinecone_index_name)
        return PineconeRetriever(
            api_key=settings.pinecone_api_key,
            index_name=settings.pinecone_index_name,
            embedding_model=settings.embedding_model,
        )

    raise ConfigurationError(f"Unknown VECTOR_BACKEND: {settings.vector_backend}")
