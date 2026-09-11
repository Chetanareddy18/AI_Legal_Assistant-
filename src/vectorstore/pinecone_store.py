"""Pinecone-backed retriever (cloud, scalable) implementing ``BaseRetriever``."""

from langchain_core.documents import Document
from pinecone import Pinecone, ServerlessSpec
from sentence_transformers import SentenceTransformer

from src.exceptions import ConfigurationError, RetrievalError
from src.logging_config import get_logger
from src.vectorstore.base import BaseRetriever

logger = get_logger(__name__)


class PineconeRetriever(BaseRetriever):
    def __init__(self, api_key: str, index_name: str, embedding_model: str, dimension: int = 384):
        if not api_key:
            raise ConfigurationError("PINECONE_API_KEY is required for the pinecone backend")

        self.pc = Pinecone(api_key=api_key)
        self.index_name = index_name
        self.embedder = SentenceTransformer(embedding_model)

        existing = self.pc.list_indexes().names()
        if index_name not in existing:
            logger.info("Creating Pinecone index '%s'", index_name)
            self.pc.create_index(
                name=index_name,
                dimension=dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
        self.index = self.pc.Index(index_name)

    def add_documents(self, texts: list[str], sources: list[str]) -> int:
        if not texts:
            return 0
        embeddings = self.embedder.encode(texts, show_progress_bar=False)
        start_id = self.index.describe_index_stats().get("total_vector_count", 0)
        payload = [
            {
                "id": str(start_id + i),
                "values": embeddings[i].tolist(),
                "metadata": {"text": texts[i], "source": sources[i]},
            }
            for i in range(len(texts))
        ]
        self.index.upsert(payload)
        logger.info("Upserted %d vectors into Pinecone index '%s'", len(payload), self.index_name)
        return len(payload)

    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        try:
            qvec = self.embedder.encode(query).tolist()
            resp = self.index.query(vector=qvec, top_k=k, include_metadata=True)
        except Exception as exc:
            raise RetrievalError(f"Pinecone query failed: {exc}") from exc

        docs: list[Document] = []
        for match in resp.matches:
            meta = dict(match.metadata or {})
            meta["score"] = match.score
            docs.append(Document(page_content=meta.get("text", ""), metadata=meta))
        return docs
