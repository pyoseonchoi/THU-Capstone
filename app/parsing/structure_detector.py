"""Structure detection for PDF documents.

Identifies headings, sections, and repeated entity entries using
font-size heuristics and text patterns. No embeddings or ML models.
"""

from __future__ import annotations

import re

from app.logging_config import get_logger
from app.schemas import DocumentPage, DocumentSection

logger = get_logger("parsing.structure_detector")

# Heuristic: text blocks with font size >= this factor of median are headings
HEADING_FONT_RATIO = 1.3


def detect_sections(pages: list[DocumentPage]) -> list[DocumentSection]:
    """Detect document sections from page blocks using font-size heuristics.

    Identifies headings as blocks with font sizes significantly larger
    than the document median and with short text content.

    Returns:
        List of DocumentSection objects in document order.
    """
    # Markdown headings carry structure even when TXT blocks have uniform fonts.
    markdown_headings: dict[int, list[tuple[str, int]]] = {}
    for page in pages:
        headings: list[tuple[str, int]] = []
        for match in re.finditer(
            r"^[ \t]*(#{1,6})[ \t]+([^\r\n]+?)[ \t]*$",
            page.text,
            re.MULTILINE,
        ):
            title = match.group(2).strip()
            if title and title not in {item[0] for item in headings}:
                headings.append((title, len(match.group(1))))
        if headings:
            markdown_headings[page.page_number] = headings

    if markdown_headings:
        sections = _sections_from_headings(pages, markdown_headings)
        logger.info("Detected %d Markdown sections", len(sections))
        return sections

    # Collect all font sizes across the document
    all_sizes: list[float] = []
    for page in pages:
        for block in page.blocks:
            all_sizes.extend(block.get("font_sizes", []))

    if not all_sizes:
        # Fallback: treat the whole document as one section
        logger.warning("No font size data available; using single section.")
        return [DocumentSection(
            title="Full Document",
            level=0,
            page_start=pages[0].page_number if pages else 1,
            page_end=pages[-1].page_number if pages else 1,
        )]

    median_size = sorted(all_sizes)[len(all_sizes) // 2]
    heading_threshold = median_size * HEADING_FONT_RATIO

    headings_by_page: dict[int, list[tuple[str, int]]] = {}

    for page in pages:
        for block in page.blocks:
            text = block.get("text", "").strip()
            font_sizes = block.get("font_sizes", [])
            if not text or not font_sizes:
                continue

            max_font = max(font_sizes)
            # Heading heuristic: large font, short text, often uppercase
            is_heading = (
                max_font >= heading_threshold
                and len(text) < 200
                and "\n" not in text.strip()
            )

            if is_heading:
                level = 1 if max_font >= median_size * 1.6 else 2
                page_headings = headings_by_page.setdefault(page.page_number, [])
                heading = text.strip()
                if heading not in {title for title, _ in page_headings}:
                    page_headings.append((heading, level))

    sections = _sections_from_headings(pages, headings_by_page)

    # If no headings detected, create a single section
    if not sections:
        sections.append(DocumentSection(
            title="Full Document",
            level=0,
            page_start=pages[0].page_number if pages else 1,
            page_end=pages[-1].page_number if pages else 1,
        ))

    logger.info("Detected %d sections", len(sections))
    return sections


def _sections_from_headings(
    pages: list[DocumentPage],
    headings_by_page: dict[int, list[tuple[str, int]]],
) -> list[DocumentSection]:
    """Build non-overlapping page-range sections from detected headings."""
    if not headings_by_page or not pages:
        return []

    sections: list[DocumentSection] = []
    first_page = pages[0].page_number
    last_page = pages[-1].page_number
    heading_pages = sorted(headings_by_page)
    if first_page < heading_pages[0]:
        sections.append(DocumentSection(
            title="Front Matter",
            level=0,
            page_start=first_page,
            page_end=heading_pages[0] - 1,
        ))

    for index, page_number in enumerate(heading_pages):
        page_headings = headings_by_page[page_number]
        next_page = (
            heading_pages[index + 1]
            if index + 1 < len(heading_pages)
            else last_page + 1
        )
        sections.append(DocumentSection(
            title=" / ".join(title for title, _ in page_headings),
            level=min(level for _, level in page_headings),
            page_start=page_number,
            page_end=next_page - 1,
        ))
    return sections


def detect_repeated_entries(
    text: str,
    *,
    min_repeats: int = 3,
) -> list[tuple[str, list[int]]]:
    """Detect repeated entity-like entries in text.

    Looks for patterns like numbered lists, repeated heading patterns,
    or structured entries that suggest individual entity descriptions.

    Returns:
        List of (pattern_label, [start_positions]) tuples.
    """
    # Pattern: numbered or lettered entries
    patterns = [
        (r"^\d+\.\s+[A-Z]", "numbered_entry"),
        (r"^[A-Z][a-z]+\s*:", "labeled_entry"),
        (r"^[-•]\s+", "bullet_entry"),
    ]
    results: list[tuple[str, list[int]]] = []

    for pattern, label in patterns:
        matches = [m.start() for m in re.finditer(pattern, text, re.MULTILINE)]
        if len(matches) >= min_repeats:
            results.append((label, matches))

    return results
