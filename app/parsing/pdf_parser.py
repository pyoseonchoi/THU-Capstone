"""Text-first PDF parser using PyMuPDF.

Extracts text, blocks, and metadata from each page while preserving
page numbers and reading order. Never silently omits an unreadable page.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import fitz  # PyMuPDF

from app.exceptions import DocumentParseError, UnsupportedDocumentError
from app.logging_config import get_logger
from app.schemas import (
    DocumentMetadata,
    DocumentPage,
    PageExtractionStatus,
    PageQualityRecord,
)

logger = get_logger("parsing.pdf_parser")

# Threshold: pages with fewer characters than this are flagged
LOW_TEXT_THRESHOLD = 50


def compute_file_hash(path: Path) -> str:
    """Return the SHA-256 hex digest of a file (first 32 chars)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()[:32]


def parse_pdf(
    pdf_path: Path,
    *,
    render_low_text: bool = False,
) -> tuple[DocumentMetadata, list[DocumentPage]]:
    """Parse a PDF file and return metadata plus per-page content.

    Args:
        pdf_path: Path to the PDF file.
        render_low_text: If True, render pages with very little text
                         as images (stored in the page object for optional
                         vision fallback).

    Returns:
        Tuple of (DocumentMetadata, list of DocumentPage).

    Raises:
        UnsupportedDocumentError: If the file is not a PDF.
        DocumentParseError: If PyMuPDF cannot open the file.
    """
    pdf_path = Path(pdf_path)
    if not pdf_path.exists():
        raise DocumentParseError(f"File not found: {pdf_path}")
    if pdf_path.suffix.lower() != ".pdf":
        raise UnsupportedDocumentError(f"Expected PDF, got: {pdf_path.suffix}")

    file_hash = compute_file_hash(pdf_path)

    try:
        doc = fitz.open(str(pdf_path))
    except Exception as exc:
        raise DocumentParseError(f"Cannot open PDF: {exc}") from exc

    pages: list[DocumentPage] = []
    total_chars = 0

    for page_num in range(len(doc)):
        page = doc[page_num]
        try:
            text = page.get_text("text") or ""
            blocks_raw = page.get_text("dict", flags=fitz.TEXT_PRESERVE_WHITESPACE).get(
                "blocks", []
            )
        except Exception as exc:
            logger.warning("Page %d extraction failed: %s", page_num + 1, exc)
            quality = PageQualityRecord(
                page_number=page_num + 1,
                extraction_status=PageExtractionStatus.UNRESOLVED,
                low_text_warning=True,
            )
            pages.append(
                DocumentPage(page_number=page_num + 1, text="", quality=quality)
            )
            continue

        # Count images vs text blocks
        text_blocks = [b for b in blocks_raw if b.get("type", 0) == 0]
        image_blocks = [b for b in blocks_raw if b.get("type", 0) == 1]

        char_count = len(text.strip())
        total_chars += char_count

        # Determine extraction status
        if char_count == 0:
            status = PageExtractionStatus.NO_TEXT
        elif char_count < LOW_TEXT_THRESHOLD:
            status = PageExtractionStatus.LOW_TEXT
        else:
            status = PageExtractionStatus.OK

        quality = PageQualityRecord(
            page_number=page_num + 1,
            character_count=char_count,
            block_count=len(text_blocks),
            image_count=len(image_blocks),
            extraction_status=status,
            low_text_warning=status in (
                PageExtractionStatus.LOW_TEXT,
                PageExtractionStatus.NO_TEXT,
            ),
        )

        # Simplified block data for downstream processing
        simplified_blocks: list[dict] = []
        for b in text_blocks:
            lines_text = ""
            font_sizes_set: set[float] = set()
            for line in (b.get("lines") or []):
                if not isinstance(line, dict):
                    continue
                for span in (line.get("spans") or []):
                    if not isinstance(span, dict):
                        continue
                    text_val = span.get("text", "")
                    if text_val:
                        lines_text += text_val
                    size_val = span.get("size")
                    if isinstance(size_val, (int, float)):
                        font_sizes_set.add(round(float(size_val), 1))
                lines_text += "\n"

            simplified_blocks.append({
                "bbox": b.get("bbox", []),
                "text": lines_text.strip(),
                "font_sizes": list(font_sizes_set) if font_sizes_set else [12.0],
            })

        pages.append(
            DocumentPage(
                page_number=page_num + 1,
                text=text,
                blocks=simplified_blocks,
                quality=quality,
            )
        )

    doc.close()

    metadata = DocumentMetadata(
        filename=pdf_path.name,
        file_hash=file_hash,
        page_count=len(pages),
        total_characters=total_chars,
    )

    logger.info(
        "Parsed %s: %d pages, %d total chars",
        pdf_path.name,
        len(pages),
        total_chars,
    )
    return metadata, pages


def save_parsed_output(
    metadata: DocumentMetadata,
    pages: list[DocumentPage],
    output_dir: Path,
) -> Path:
    """Save parsed output as JSON for reproducibility."""
    output_dir.mkdir(parents=True, exist_ok=True)
    out_path = output_dir / f"{metadata.document_id}_parsed.json"
    data = {
        "metadata": metadata.model_dump(mode="json"),
        "pages": [p.model_dump(mode="json") for p in pages],
    }
    out_path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Saved parsed output to %s", out_path)
    return out_path


def load_parsed_output(path: Path) -> tuple[DocumentMetadata, list[DocumentPage]]:
    """Load previously parsed output from JSON."""
    data = json.loads(path.read_text(encoding="utf-8"))
    metadata = DocumentMetadata(**data["metadata"])
    pages = [DocumentPage(**p) for p in data["pages"]]
    return metadata, pages
