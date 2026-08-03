"""Plain-text parser with encoding detection and virtual page boundaries."""

from __future__ import annotations

import re
from pathlib import Path

from app.exceptions import DocumentParseError, UnsupportedDocumentError
from app.logging_config import get_logger
from app.parsing.pdf_parser import compute_file_hash
from app.schemas import (
    DocumentMetadata,
    DocumentPage,
    PageExtractionStatus,
    PageQualityRecord,
)

logger = get_logger("parsing.text_parser")

DEFAULT_VIRTUAL_PAGE_CHARACTERS = 8_000
PAGE_MARKER_RE = re.compile(
    r"<!--\s*page-start-marker-(\d+)\s*-->",
    re.IGNORECASE,
)


def _decode_text(content: bytes) -> tuple[str, str]:
    """Decode common UTF and Korean text encodings."""
    if content.startswith((b"\xff\xfe", b"\xfe\xff")):
        try:
            return content.decode("utf-16"), "utf-16"
        except UnicodeDecodeError:
            pass

    for encoding in ("utf-8-sig", "cp949", "euc-kr"):
        try:
            return content.decode(encoding), encoding
        except UnicodeDecodeError:
            continue

    # Latin-1 is a lossless byte fallback and never raises.
    return content.decode("latin-1"), "latin-1"


def normalize_text_source(text: str) -> str:
    """Restore escaped line breaks in serialized Markdown exports.

    Some Docling downloads contain literal ``\\n`` sequences and no physical
    line breaks. Only decode that representation when it clearly dominates so
    ordinary prose containing a backslash remains untouched.
    """
    physical_newlines = text.count("\n")
    escaped_newlines = text.count(r"\n")
    marker_serialized = (
        physical_newlines == 0
        and escaped_newlines >= 2
        and PAGE_MARKER_RE.search(text) is not None
    )
    looks_serialized = marker_serialized or (
        escaped_newlines >= 10
        and escaped_newlines > max(physical_newlines * 4, 20)
    )
    if looks_serialized:
        text = text.replace(r"\r\n", "\n")
        text = text.replace(r"\n", "\n").replace(r"\r", "\n")
        text = text.replace(r'\"', '"')
    return text.replace("\r\n", "\n").replace("\r", "\n")


def split_marked_pages(text: str) -> list[tuple[int, str]]:
    """Split Markdown using Docling page markers while preserving page IDs."""
    matches = list(PAGE_MARKER_RE.finditer(text))
    if not matches:
        return []

    pages: list[tuple[int, str]] = []
    seen: set[int] = set()
    for index, match in enumerate(matches):
        page_number = int(match.group(1))
        if page_number in seen:
            continue
        body_start = match.end()
        body_end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        pages.append((page_number, text[body_start:body_end].strip()))
        seen.add(page_number)
    return pages


def _split_large_block(block: str, target_characters: int) -> list[str]:
    """Split one oversized paragraph, preferring line boundaries."""
    pieces: list[str] = []
    current: list[str] = []
    current_length = 0

    for line in block.splitlines() or [block]:
        if len(line) > target_characters:
            if current:
                pieces.append("\n".join(current).strip())
                current = []
                current_length = 0
            pieces.extend(
                line[index:index + target_characters].strip()
                for index in range(0, len(line), target_characters)
                if line[index:index + target_characters].strip()
            )
        elif current and current_length + len(line) + 1 > target_characters:
            pieces.append("\n".join(current).strip())
            current = [line]
            current_length = len(line)
        else:
            current.append(line)
            current_length += len(line) + (1 if current_length else 0)

    if current:
        pieces.append("\n".join(current).strip())
    return [piece for piece in pieces if piece]


def split_virtual_pages(
    text: str,
    *,
    target_characters: int = DEFAULT_VIRTUAL_PAGE_CHARACTERS,
) -> list[str]:
    """Split text into stable, paragraph-aware virtual pages."""
    if target_characters < 500:
        raise ValueError("target_characters must be at least 500")

    normalized = normalize_text_source(text)
    normalized = normalized.replace("\x00", "").strip()
    if not normalized:
        return []

    paragraphs: list[str] = []
    for form_feed_part in normalized.split("\f"):
        paragraphs.extend(
            part.strip()
            for part in re.split(r"\n\s*\n", form_feed_part)
            if part.strip()
        )

    pages: list[str] = []
    current: list[str] = []
    current_length = 0
    for paragraph in paragraphs:
        paragraph_parts = (
            _split_large_block(paragraph, target_characters)
            if len(paragraph) > target_characters
            else [paragraph]
        )
        for part in paragraph_parts:
            additional = len(part) + (2 if current else 0)
            if current and current_length + additional > target_characters:
                pages.append("\n\n".join(current))
                current = []
                current_length = 0
            current.append(part)
            current_length += len(part) + (2 if current_length else 0)

    if current:
        pages.append("\n\n".join(current))
    return pages


def parse_text(
    text_path: Path,
    *,
    virtual_page_characters: int = DEFAULT_VIRTUAL_PAGE_CHARACTERS,
) -> tuple[DocumentMetadata, list[DocumentPage]]:
    """Parse a .txt file into virtual pages compatible with the PDF pipeline."""
    text_path = Path(text_path)
    if not text_path.exists():
        raise DocumentParseError(f"File not found: {text_path}")
    if text_path.suffix.lower() != ".txt":
        raise UnsupportedDocumentError(f"Expected TXT, got: {text_path.suffix}")

    try:
        text, encoding = _decode_text(text_path.read_bytes())
    except OSError as exc:
        raise DocumentParseError(f"Cannot read text file: {exc}") from exc

    normalized_text = normalize_text_source(text)
    marked_pages = split_marked_pages(normalized_text)
    page_parts = marked_pages or [
        (index, page_text)
        for index, page_text in enumerate(
            split_virtual_pages(
                normalized_text,
                target_characters=virtual_page_characters,
            ),
            start=1,
        )
    ]
    if not page_parts or not any(page_text.strip() for _, page_text in page_parts):
        raise DocumentParseError("Text file is empty or contains no readable text")

    pages: list[DocumentPage] = []
    for page_number, page_text in page_parts:
        character_count = len(page_text)
        status = (
            PageExtractionStatus.OK
            if character_count >= 50
            else PageExtractionStatus.LOW_TEXT
        )
        pages.append(DocumentPage(
            page_number=page_number,
            text=page_text,
            blocks=[{
                "text": page_text,
                "font_sizes": [12.0],
                "bbox": [],
            }],
            quality=PageQualityRecord(
                page_number=page_number,
                character_count=character_count,
                block_count=1,
                image_count=0,
                extraction_status=status,
                low_text_warning=status == PageExtractionStatus.LOW_TEXT,
            ),
        ))

    metadata = DocumentMetadata(
        filename=text_path.name,
        file_hash=compute_file_hash(text_path),
        page_count=len(pages),
        total_characters=sum(len(page.text) for page in pages),
    )
    logger.info(
        "Parsed %s as %s: %d virtual pages, %d total chars",
        text_path.name,
        encoding,
        len(pages),
        metadata.total_characters,
    )
    return metadata, pages
