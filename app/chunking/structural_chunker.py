"""Structure-aware document chunker.

Splitting priority:
  document → chapter/section → repeated entity entry → heading
  → paragraph → sentence-aware token split

Token counting uses an approximate tokenizer: len(text) / 4.
This is documented as an approximation since the exact Mistral
tokenizer is not in the required dependencies.

No embeddings or semantic similarity are used.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from app.logging_config import get_logger
from app.schemas import (
    DocumentChunk,
    DocumentPage,
    DocumentSection,
)

logger = get_logger("chunking.structural_chunker")


def approx_token_count(text: str) -> int:
    """Approximate token count using chars/4 heuristic.

    This is an approximation. Mistral models use a SentencePiece/BPE
    tokenizer that averages ~4 characters per token for English text.
    For other languages or technical content, the ratio may differ.
    """
    return max(1, len(text) // 4)


def _split_into_paragraphs(text: str) -> list[str]:
    """Split text into paragraphs on double newlines or single blank lines."""
    parts = re.split(r"\n\s*\n", text)
    return [p.strip() for p in parts if p.strip()]


def _split_into_sentences(text: str) -> list[str]:
    """Split text into sentences using simple regex."""
    sentences = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in sentences if s.strip()]


def _build_token_splits(
    text: str,
    target_tokens: int,
    overlap_tokens: int,
) -> list[str]:
    """Split text by token budget with overlap, respecting sentence boundaries."""
    sentences = _split_into_sentences(text)
    if not sentences:
        return [text] if text.strip() else []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_tokens = 0

    for sentence in sentences:
        sent_tokens = approx_token_count(sentence)
        if current_tokens + sent_tokens > target_tokens and current_parts:
            chunks.append(" ".join(current_parts))
            # Overlap: keep last N tokens worth of sentences
            overlap_parts: list[str] = []
            overlap_count = 0
            for part in reversed(current_parts):
                pt = approx_token_count(part)
                if overlap_count + pt > overlap_tokens:
                    break
                overlap_parts.insert(0, part)
                overlap_count += pt
            current_parts = overlap_parts
            current_tokens = overlap_count

        current_parts.append(sentence)
        current_tokens += sent_tokens

    if current_parts:
        chunks.append(" ".join(current_parts))

    return chunks


@dataclass(frozen=True)
class _PageUnit:
    """A canonical piece of page text used exactly once by the chunker."""

    page_number: int
    text: str
    section_id: str
    section_title: str


def _section_for_page(
    page_number: int,
    sections: list[DocumentSection],
) -> tuple[str, str]:
    """Return stable section metadata for a page.

    Legacy parsed data can contain overlapping sections. The most recently
    started section owns the page, while headings that start on that same page
    are retained as metadata.
    """
    matching = [
        section
        for section in sections
        if section.page_start <= page_number <= section.page_end
    ]
    if not matching:
        return "", "Full Document"

    latest_start = max(section.page_start for section in matching)
    owners = [section for section in matching if section.page_start == latest_start]
    titles = list(dict.fromkeys(section.title for section in owners if section.title))
    section_ids = list(dict.fromkeys(section.section_id for section in owners))
    return "+".join(section_ids), " / ".join(titles) or "Full Document"


def _build_page_units(
    pages: list[DocumentPage],
    sections: list[DocumentSection],
    target_tokens: int,
    overlap_tokens: int,
) -> list[_PageUnit]:
    units: list[_PageUnit] = []
    for page in pages:
        text = page.text.strip()
        if not text:
            continue
        section_id, section_title = _section_for_page(page.page_number, sections)
        pieces = (
            _build_token_splits(text, target_tokens, overlap_tokens)
            if approx_token_count(text) > target_tokens
            else [text]
        )
        units.extend(
            _PageUnit(
                page_number=page.page_number,
                text=piece.strip(),
                section_id=section_id,
                section_title=section_title,
            )
            for piece in pieces
            if piece.strip()
        )
    return units


def chunk_document(
    document_id: str,
    pages: list[DocumentPage],
    sections: list[DocumentSection],
    *,
    target_tokens: int = 3500,
    overlap_tokens: int = 250,
) -> list[DocumentChunk]:
    """Create canonical page-first chunks without duplicating page text.

    Section headings are metadata and optional grouping boundaries. Page text
    is never copied into every section that happens to overlap that page.
    Overlap is used only when a single page exceeds the token budget.

    Args:
        document_id: Document identifier.
        pages: Parsed pages.
        sections: Detected sections.
        target_tokens: Target chunk size in approximate tokens.
        overlap_tokens: Overlap for forced token splits.

    Returns:
        List of DocumentChunk in document order.
    """
    if not pages:
        return []

    units = _build_page_units(
        pages, sections, target_tokens, overlap_tokens
    )
    groups: list[list[_PageUnit]] = []
    current: list[_PageUnit] = []
    current_tokens = 0

    for unit in units:
        marked_text = f"[PAGE {unit.page_number}]\n{unit.text}"
        unit_tokens = approx_token_count(marked_text)
        budget_exceeded = bool(
            current and current_tokens + unit_tokens > target_tokens
        )
        if budget_exceeded:
            groups.append(current)
            current = []
            current_tokens = 0
        current.append(unit)
        current_tokens += unit_tokens
    if current:
        groups.append(current)

    chunks: list[DocumentChunk] = []
    for chunk_index, group in enumerate(groups):
        text = "\n\n".join(
            f"[PAGE {unit.page_number}]\n{unit.text}"
            for unit in group
        )
        section_titles = list(dict.fromkeys(
            unit.section_title
            for unit in group
            if unit.section_title
        ))
        section_ids = list(dict.fromkeys(
            unit.section_id
            for unit in group
            if unit.section_id
        ))
        chunk = DocumentChunk(
            document_id=document_id,
            chunk_id=(
                f"{document_id}:c{chunk_index:05d}:"
                f"p{group[0].page_number:04d}-{group[-1].page_number:04d}"
            ),
            chunk_index=chunk_index,
            section_id="+".join(section_ids),
            section_title=section_titles[0] if section_titles else "Full Document",
            section_titles=section_titles,
            page_start=group[0].page_number,
            page_end=group[-1].page_number,
            text=text,
        )
        chunk.compute_hash()
        chunks.append(chunk)

    for index, chunk in enumerate(chunks):
        if index > 0:
            chunk.previous_context_hint = chunks[index - 1].text[-200:].strip()
        if index + 1 < len(chunks):
            chunk.next_context_hint = chunks[index + 1].text[:200].strip()

    logger.info(
        "Created %d canonical chunks from %d pages and %d sections (target=%d tokens)",
        len(chunks),
        len(pages),
        len(sections),
        target_tokens,
    )
    return chunks


def generate_chunk_manifest(chunks: list[DocumentChunk]) -> dict:
    """Generate a coverage manifest for the chunks.

    Returns a dict with summary stats and per-chunk info.
    """
    manifest = {
        "total_chunks": len(chunks),
        "total_tokens_approx": sum(approx_token_count(c.text) for c in chunks),
        "page_range": (
            (min(c.page_start for c in chunks), max(c.page_end for c in chunks))
            if chunks
            else (0, 0)
        ),
        "chunks": [
            {
                "chunk_id": c.chunk_id,
                "chunk_index": c.chunk_index,
                "section_title": c.section_title,
                "section_titles": c.section_titles,
                "page_start": c.page_start,
                "page_end": c.page_end,
                "tokens_approx": approx_token_count(c.text),
                "content_hash": c.content_hash,
            }
            for c in chunks
        ],
    }
    return manifest
