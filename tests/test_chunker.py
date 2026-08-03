"""Tests for the structural chunker."""

from __future__ import annotations

from app.chunking.structural_chunker import (
    approx_token_count,
    chunk_document,
    generate_chunk_manifest,
)
from app.schemas import DocumentPage, DocumentSection


def _make_pages(texts: list[str]) -> list[DocumentPage]:
    """Create DocumentPage objects from text strings."""
    return [
        DocumentPage(page_number=i + 1, text=t)
        for i, t in enumerate(texts)
    ]


def _make_section(title: str, page_start: int, page_end: int) -> DocumentSection:
    return DocumentSection(
        title=title, level=1, page_start=page_start, page_end=page_end
    )


class TestApproxTokenCount:
    def test_empty(self):
        assert approx_token_count("") == 1  # min 1

    def test_short(self):
        assert approx_token_count("hello") == 1

    def test_typical(self):
        # 400 chars -> ~100 tokens
        text = "word " * 80  # 400 chars
        assert 90 <= approx_token_count(text) <= 110


class TestChunkDocument:
    def test_single_small_section(self):
        pages = _make_pages(["This is a small document."])
        sections = [_make_section("Intro", 1, 1)]
        chunks = chunk_document("doc1", pages, sections, target_tokens=3500)

        assert len(chunks) == 1
        assert chunks[0].section_title == "Intro"
        assert chunks[0].page_start == 1
        assert chunks[0].page_end == 1
        assert chunks[0].chunk_index == 0
        assert chunks[0].document_id == "doc1"
        assert chunks[0].content_hash != ""

    def test_chunk_ordering_preserved(self):
        pages = _make_pages(["Page one.", "Page two.", "Page three."])
        sections = [
            _make_section("A", 1, 1),
            _make_section("B", 2, 2),
            _make_section("C", 3, 3),
        ]
        chunks = chunk_document("doc1", pages, sections, target_tokens=3)

        assert len(chunks) == 3
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i

    def test_page_numbers_preserved(self):
        pages = _make_pages(["First page.", "Second page.", "Third page."])
        sections = [_make_section("All", 1, 3)]
        chunks = chunk_document("doc1", pages, sections, target_tokens=3500)

        # All pages should be within the chunk's range
        for chunk in chunks:
            assert chunk.page_start >= 1
            assert chunk.page_end <= 3

    def test_large_section_splits(self):
        # Create a section much larger than target
        big_text = "This is a paragraph. " * 500  # ~10500 chars = ~2625 tokens
        pages = _make_pages([big_text])
        sections = [_make_section("Big", 1, 1)]
        chunks = chunk_document(
            "doc1", pages, sections,
            target_tokens=500,  # Force splitting
            overlap_tokens=50,
        )

        assert len(chunks) > 1
        # All chunks should be in order
        for i, chunk in enumerate(chunks):
            assert chunk.chunk_index == i
        # Context hints should be populated for middle chunks
        if len(chunks) > 2:
            assert chunks[1].previous_context_hint != ""
            assert chunks[1].next_context_hint != ""

    def test_empty_pages(self):
        pages = _make_pages(["", "Some text", ""])
        sections = [_make_section("Sparse", 1, 3)]
        chunks = chunk_document("doc1", pages, sections, target_tokens=3500)
        # Should still create a chunk (non-empty text exists)
        assert len(chunks) >= 1

    def test_no_pages(self):
        chunks = chunk_document("doc1", [], [], target_tokens=3500)
        assert chunks == []

    def test_content_hash_unique(self):
        pages = _make_pages(["Text A.", "Text B."])
        sections = [
            _make_section("S1", 1, 1),
            _make_section("S2", 2, 2),
        ]
        chunks = chunk_document("doc1", pages, sections, target_tokens=3)
        hashes = [c.content_hash for c in chunks]
        assert len(hashes) == len(set(hashes)), "Hashes must be unique"

    def test_overlapping_sections_do_not_duplicate_page_text(self):
        pages = _make_pages(["Shared page text.", "Second page text."])
        sections = [
            _make_section("First heading", 1, 1),
            _make_section("Second heading", 1, 2),
        ]

        chunks = chunk_document("doc1", pages, sections, target_tokens=3500)
        combined = "\n".join(chunk.text for chunk in chunks)

        assert combined.count("Shared page text.") == 1
        assert combined.count("Second page text.") == 1

    def test_chunk_ids_are_stable(self):
        pages = _make_pages(["Page one.", "Page two."])
        sections = [_make_section("All", 1, 2)]

        first = chunk_document("doc1", pages, sections, target_tokens=3500)
        second = chunk_document("doc1", pages, sections, target_tokens=3500)

        assert [chunk.chunk_id for chunk in first] == [
            chunk.chunk_id for chunk in second
        ]

    def test_short_sections_share_budget_and_keep_titles(self):
        pages = _make_pages(["First.", "Second.", "Third."])
        sections = [
            _make_section("A", 1, 1),
            _make_section("B", 2, 2),
            _make_section("C", 3, 3),
        ]

        chunks = chunk_document("doc1", pages, sections, target_tokens=100)

        assert len(chunks) == 1
        assert chunks[0].section_titles == ["A", "B", "C"]
        assert "[PAGE 1]" in chunks[0].text
        assert "[PAGE 3]" in chunks[0].text


class TestChunkManifest:
    def test_manifest_structure(self):
        pages = _make_pages(["Page 1", "Page 2"])
        sections = [_make_section("All", 1, 2)]
        chunks = chunk_document("doc1", pages, sections, target_tokens=3500)
        manifest = generate_chunk_manifest(chunks)

        assert manifest["total_chunks"] == len(chunks)
        assert "total_tokens_approx" in manifest
        assert len(manifest["chunks"]) == len(chunks)
        for entry in manifest["chunks"]:
            assert "chunk_id" in entry
            assert "tokens_approx" in entry
