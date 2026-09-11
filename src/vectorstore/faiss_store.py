"""Local, dependency-free vector store backed by FAISS.

This backend requires no external API keys or network access, which makes it
ideal for local development, demos, and offline evaluation. It persists its
index + metadata to disk so it survives process restarts.
"""
import json
import os
import threading

import faiss
import numpy as np
from langchain_core.documents import Document
from sentence_transformers import SentenceTransformer

from src.exceptions import RetrievalError
from src.logging_config import get_logger
from src.vectorstore.base import BaseRetriever

logger = get_logger(__name__)


class FaissRetriever(BaseRetriever):
    """Flat, cosine-similarity FAISS index with a JSON metadata sidecar."""

    def __init__(self, index_dir: str, embedding_model: str):
        self.index_dir = index_dir
        self.index_path = os.path.join(index_dir, "index.faiss")
        self.meta_path = os.path.join(index_dir, "metadata.json")
        self._lock = threading.Lock()

        os.makedirs(index_dir, exist_ok=True)
        self.embedder = SentenceTransformer(embedding_model)
        dim_fn = getattr(self.embedder, "get_embedding_dimension", None) or self.embedder.get_sentence_embedding_dimension
        self.dimension = dim_fn()

        self.index = faiss.IndexFlatIP(self.dimension)
        self.metadata: list[dict] = []
        self._load()

    def _load(self) -> None:
        if os.path.exists(self.index_path) and os.path.exists(self.meta_path):
            try:
                self.index = faiss.read_index(self.index_path)
                with open(self.meta_path, encoding="utf-8") as f:
                    self.metadata = json.load(f)
                logger.info("Loaded FAISS index with %d vectors", self.index.ntotal)
            except Exception as exc:  # corrupted index shouldn't crash startup
                logger.warning("Failed to load existing FAISS index, starting fresh: %s", exc)
                self.index = faiss.IndexFlatIP(self.dimension)
                self.metadata = []

    def _persist(self) -> None:
        faiss.write_index(self.index, self.index_path)
        with open(self.meta_path, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    @staticmethod
    def _normalize(vectors: np.ndarray) -> np.ndarray:
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1e-12
        return vectors / norms

    def add_documents(self, texts: list[str], sources: list[str]) -> int:
        if not texts:
            return 0
        with self._lock:
            embeddings = self.embedder.encode(texts, show_progress_bar=False)
            embeddings = self._normalize(np.array(embeddings, dtype="float32"))
            self.index.add(embeddings)
            for text, source in zip(texts, sources, strict=True):
                self.metadata.append({"text": text, "source": source})
            self._persist()
        logger.info("Indexed %d chunks into FAISS store", len(texts))
        return len(texts)

    def get_relevant_documents(self, query: str, k: int = 5) -> list[Document]:
        if self.index.ntotal == 0:
            logger.warning("FAISS index is empty; no documents to retrieve")
            return []
        try:
            qvec = self.embedder.encode([query])
            qvec = self._normalize(np.array(qvec, dtype="float32"))
            scores, indices = self.index.search(qvec, min(k, self.index.ntotal))
        except Exception as exc:
            raise RetrievalError(f"FAISS search failed: {exc}") from exc

        docs: list[Document] = []
        for score, idx in zip(scores[0], indices[0], strict=True):
            if idx < 0 or idx >= len(self.metadata):
                continue
            meta = dict(self.metadata[idx])
            meta["score"] = float(score)
            docs.append(Document(page_content=meta.get("text", ""), metadata=meta))
        return docs
