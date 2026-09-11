"""Unified ingestion pipeline: results/*_full.json -> chunks -> vector store.

Works transparently with either backend (FAISS or Pinecone) since it only
talks to the ``BaseRetriever`` interface returned by ``get_retriever()``.
Replaces the previous Pinecone-only ``vector_db_build.py`` flow.
"""
import io
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.exceptions import DocumentProcessingError
from src.graph.knowledge_graph import build_graph_from_chunks, save_graph
from src.logging_config import get_logger
from src.vectorstore import get_retriever

logger = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(__file__))
RESULTS_DIR = os.path.join(ROOT, "results")

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    separators=["\n\n", "\n", ".", " "],
)


def _load_text(path: str) -> str:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        raise DocumentProcessingError(f"Could not read {path}: {exc}") from exc
    return data.get("text") or data.get("full_text") or ""


def ingest_results(results_dir: str = RESULTS_DIR) -> int:
    """Chunk every ``*_full.json`` under ``results_dir`` and index it."""
    if not os.path.isdir(results_dir):
        logger.warning("Results directory %s does not exist; nothing to ingest", results_dir)
        return 0

    all_chunks, sources = [], []
    for file in sorted(os.listdir(results_dir)):
        if not file.endswith("_full.json"):
            continue
        text = _load_text(os.path.join(results_dir, file))
        if not text.strip():
            continue
        chunks = splitter.split_text(text)
        all_chunks.extend(chunks)
        sources.extend([file] * len(chunks))

    retriever = get_retriever()
    count = retriever.add_documents(all_chunks, sources)
    logger.info("Ingested %d chunks from %s", count, results_dir)

    graph = build_graph_from_chunks(all_chunks, sources)
    save_graph(graph)

    return count


def ingest_pdf_bytes(file_bytes: bytes, filename: str) -> dict:
    """Extract, chunk, index, and graph a single uploaded PDF (used by the UI's
    upload widget). Returns a small summary dict for display/confirmation."""
    import pdfplumber

    from src.graph.knowledge_graph import load_graph

    try:
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            text = "\n".join(page.extract_text() or "" for page in pdf.pages)
    except Exception as exc:
        raise DocumentProcessingError(f"Could not read PDF '{filename}': {exc}") from exc

    if not text.strip():
        raise DocumentProcessingError(f"No extractable text found in '{filename}' (scanned/image-only PDFs need OCR).")

    chunks = splitter.split_text(text)
    sources = [filename] * len(chunks)

    retriever = get_retriever()
    indexed = retriever.add_documents(chunks, sources)

    existing_graph = load_graph()
    new_graph = build_graph_from_chunks(chunks, sources)
    merged = nx_compose(existing_graph, new_graph)
    save_graph(merged)

    logger.info("Ingested uploaded PDF '%s': %d chunk(s) indexed", filename, indexed)
    return {"filename": filename, "chunks_indexed": indexed, "characters_extracted": len(text)}


def nx_compose(a, b):
    import networkx as nx

    return nx.compose(a, b)


if __name__ == "__main__":
    ingest_results()
