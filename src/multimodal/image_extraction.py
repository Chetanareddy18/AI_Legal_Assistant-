"""Extract images (charts, diagrams, exhibits) embedded in legal PDFs.

Uses PyMuPDF (``fitz``) which is fast and dependency-light compared to
rendering full pages. Extracted images are saved to disk so they can be
indexed by the multi-modal retriever and displayed alongside text evidence.
"""
import os
from typing import TypedDict

import fitz  # PyMuPDF

from src.exceptions import DocumentProcessingError
from src.logging_config import get_logger

logger = get_logger(__name__)


class ExtractedImage(TypedDict):
    path: str
    source_pdf: str
    page: int


def extract_images_from_pdf(pdf_path: str, output_dir: str) -> list[ExtractedImage]:
    if not os.path.exists(pdf_path):
        raise DocumentProcessingError(f"PDF not found: {pdf_path}")

    os.makedirs(output_dir, exist_ok=True)
    base_name = os.path.splitext(os.path.basename(pdf_path))[0]
    extracted: list[ExtractedImage] = []

    try:
        doc = fitz.open(pdf_path)
    except Exception as exc:
        raise DocumentProcessingError(f"Failed to open {pdf_path}: {exc}") from exc

    try:
        for page_index in range(len(doc)):
            page = doc[page_index]
            for image_index, img in enumerate(page.get_images(full=True)):
                xref = img[0]
                try:
                    pix = fitz.Pixmap(doc, xref)
                    if pix.n - pix.alpha >= 4:  # CMYK -> RGB for portability
                        pix = fitz.Pixmap(fitz.csRGB, pix)
                    out_path = os.path.join(
                        output_dir, f"{base_name}_p{page_index + 1}_{image_index}.png"
                    )
                    pix.save(out_path)
                    extracted.append({"path": out_path, "source_pdf": base_name, "page": page_index + 1})
                except Exception as exc:
                    logger.warning("Skipping image %d on page %d of %s: %s", image_index, page_index, pdf_path, exc)
    finally:
        doc.close()

    logger.info("Extracted %d image(s) from %s", len(extracted), pdf_path)
    return extracted


def extract_images_from_dir(pdf_dir: str, output_dir: str) -> list[ExtractedImage]:
    all_images: list[ExtractedImage] = []
    if not os.path.isdir(pdf_dir):
        return all_images
    for file in sorted(os.listdir(pdf_dir)):
        if file.lower().endswith(".pdf"):
            all_images.extend(extract_images_from_pdf(os.path.join(pdf_dir, file), output_dir))
    return all_images
