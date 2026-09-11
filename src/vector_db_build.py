"""Deprecated entry point kept for backward compatibility.

Use ``python -m src.ingest`` instead, which supports both the FAISS and
Pinecone backends via ``VECTOR_BACKEND``. This script now delegates to it,
reading the same pre-computed chunk/embedding arrays and pushing them through
``retriever.add_documents`` so either backend works unmodified.
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.logging_config import get_logger
from src.vectorstore import get_retriever

logger = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(__file__))
EMB_DIR = os.path.join(ROOT, "embeddings")


def main() -> None:
    chunks = np.load(os.path.join(EMB_DIR, "chunks.npy"), allow_pickle=True)
    refs = np.load(os.path.join(EMB_DIR, "refs.npy"), allow_pickle=True)

    retriever = get_retriever()
    count = retriever.add_documents(list(chunks), list(refs))
    logger.info("Vector DB updated successfully: %d chunks indexed.", count)


if __name__ == "__main__":
    main()
