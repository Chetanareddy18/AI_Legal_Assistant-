"""Multi-modal (text + image) retrieval using a shared CLIP embedding space.

Charts, diagrams, and exhibit images are encoded with a CLIP model into the
same embedding space as text queries, enabling a single similarity search
across both modalities. Falls back gracefully (text-only) if no images have
been indexed or the CLIP model can't be loaded, so multimodal support is
strictly additive.
"""
import json
import os
from typing import TypedDict

import numpy as np

from src.config import get_settings
from src.logging_config import get_logger

logger = get_logger(__name__)

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
MM_DIR = os.path.join(ROOT, "embeddings", "multimodal")
MM_INDEX_PATH = os.path.join(MM_DIR, "image_index.npy")
MM_META_PATH = os.path.join(MM_DIR, "image_meta.json")


class ImageMatch(TypedDict):
    path: str
    source_pdf: str
    page: int
    score: float


class MultiModalIndex:
    """Thin wrapper around a CLIP model + flat numpy index for image search."""

    def __init__(self, model_name: str | None = None):
        settings = get_settings()
        self.model_name = model_name or settings.clip_model_name
        self._model = None
        self.embeddings: np.ndarray = np.zeros((0, 512), dtype="float32")
        self.metadata: list[dict] = []
        self._load()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _load(self) -> None:
        if os.path.exists(MM_INDEX_PATH) and os.path.exists(MM_META_PATH):
            self.embeddings = np.load(MM_INDEX_PATH)
            with open(MM_META_PATH, encoding="utf-8") as f:
                self.metadata = json.load(f)

    def _persist(self) -> None:
        os.makedirs(MM_DIR, exist_ok=True)
        np.save(MM_INDEX_PATH, self.embeddings)
        with open(MM_META_PATH, "w", encoding="utf-8") as f:
            json.dump(self.metadata, f, indent=2)

    def index_images(self, image_paths: list[dict]) -> int:
        """``image_paths``: list of dicts with keys path/source_pdf/page."""
        if not image_paths:
            return 0
        from PIL import Image

        vectors = []
        kept_meta = []
        for item in image_paths:
            try:
                img = Image.open(item["path"]).convert("RGB")
                vectors.append(self.model.encode(img))
                kept_meta.append(item)
            except Exception as exc:
                logger.warning("Skipping unreadable image %s: %s", item.get("path"), exc)

        if not vectors:
            return 0

        new_vecs = np.array(vectors, dtype="float32")
        self.embeddings = (
            np.vstack([self.embeddings, new_vecs]) if self.embeddings.size else new_vecs
        )
        self.metadata.extend(kept_meta)
        self._persist()
        logger.info("Indexed %d image(s) into multimodal store", len(kept_meta))
        return len(kept_meta)

    def search(self, query: str, k: int = 3) -> list[ImageMatch]:
        if self.embeddings.shape[0] == 0:
            return []
        qvec = self.model.encode(query)
        qvec = qvec / (np.linalg.norm(qvec) + 1e-12)
        mat = self.embeddings / (np.linalg.norm(self.embeddings, axis=1, keepdims=True) + 1e-12)
        scores = mat @ qvec
        top_idx = np.argsort(-scores)[:k]
        return [
            {**self.metadata[i], "score": float(scores[i])}
            for i in top_idx
        ]


_index_singleton: MultiModalIndex | None = None


def get_multimodal_index() -> MultiModalIndex:
    global _index_singleton
    if _index_singleton is None:
        _index_singleton = MultiModalIndex()
    return _index_singleton
