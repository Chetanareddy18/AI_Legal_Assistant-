"""Multi-modal RAG: combines text passages with visually similar chart/diagram images."""
from src.multimodal.image_extraction import (
    extract_images_from_dir,
    extract_images_from_pdf,
)
from src.multimodal.multimodal_index import get_multimodal_index


def multimodal_search(query: str, text_top_k: int = 4, image_top_k: int = 3) -> dict:
    from src.rag_pipeline import rag_search

    text_results = rag_search(query, top_k=text_top_k)
    image_results = get_multimodal_index().search(query, k=image_top_k)
    return {"text": text_results, "images": image_results}


__all__ = [
    "extract_images_from_dir",
    "extract_images_from_pdf",
    "get_multimodal_index",
    "multimodal_search",
]
