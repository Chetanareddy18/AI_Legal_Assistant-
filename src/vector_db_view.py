"""Export a CSV preview of the currently configured vector store's contents.

Supports both backends: for FAISS it reads the local metadata sidecar
directly; for Pinecone it fetches vectors by id as before.
"""
import csv
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.config import get_settings

settings = get_settings()


def export_faiss_preview(out_path: str = "vector_db_preview.csv") -> None:
    from src.vectorstore import get_retriever
    from src.vectorstore.faiss_store import FaissRetriever

    retriever = get_retriever()
    if not isinstance(retriever, FaissRetriever):
        print("VECTOR_BACKEND is not 'faiss'; nothing to export.")
        return

    metadata = retriever.metadata
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "source"])
        for i, item in enumerate(metadata):
            writer.writerow([i, item.get("text", ""), item.get("source", "")])

    print(f"Exported {out_path} ({len(metadata)} rows)")


def export_pinecone_preview(out_path: str = "vector_db_preview.csv", max_ids: int = 5000, batch: int = 100) -> None:
    from pinecone import Pinecone

    pc = Pinecone(api_key=settings.pinecone_api_key)
    index = pc.Index(settings.pinecone_index_name)

    ids = [str(i) for i in range(max_ids)]
    rows = []
    for i in range(0, len(ids), batch):
        result = index.fetch(ids=ids[i:i + batch])
        for vid, vector in result.vectors.items():
            rows.append([vid, vector.metadata.get("text", ""), vector.metadata.get("source", ""), vector.values[:10]])

    with open(out_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "text", "source", "embedding_preview"])
        writer.writerows(rows)

    print(f"Exported {out_path} ({len(rows)} rows)")


if __name__ == "__main__":
    if settings.vector_backend == "faiss":
        export_faiss_preview()
    else:
        export_pinecone_preview()
