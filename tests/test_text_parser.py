"""Tests for plain-text document parsing."""

from __future__ import annotations

import pytest

from app.exceptions import DocumentParseError
from app.parsing.structure_detector import detect_sections
from app.parsing.text_parser import (
    normalize_text_source,
    parse_text,
    split_virtual_pages,
)


def test_parse_utf8_text(tmp_path):
    path = tmp_path / "sample.txt"
    path.write_text("First paragraph.\n\nSecond paragraph.", encoding="utf-8")

    metadata, pages = parse_text(path)

    assert metadata.filename == "sample.txt"
    assert metadata.page_count == 1
    assert "Second paragraph" in pages[0].text
    assert pages[0].quality is not None


def test_parse_cp949_korean_text(tmp_path):
    path = tmp_path / "korean.txt"
    path.write_bytes("한글 문서를 정상적으로 읽습니다.".encode("cp949"))

    _, pages = parse_text(path)

    assert "한글 문서" in pages[0].text


def test_long_text_splits_without_losing_paragraphs():
    text = "\n\n".join(f"Paragraph {index}: " + "x" * 180 for index in range(12))

    pages = split_virtual_pages(text, target_characters=500)

    assert len(pages) > 1
    assert "\n\n".join(pages) == text


def test_empty_text_is_rejected(tmp_path):
    path = tmp_path / "empty.txt"
    path.write_text(" \n\n ", encoding="utf-8")

    with pytest.raises(DocumentParseError):
        parse_text(path)


def test_parse_escaped_docling_page_markers(tmp_path):
    path = tmp_path / "docling.txt"
    path.write_text(
        r"<!-- page-start-marker-1 -->\n\n## First Park\n\nPage one."
        r"\n\n<!-- page-start-marker-2 -->\n\n## Toolbox\n\nPage two.",
        encoding="utf-8",
    )

    metadata, pages = parse_text(path)
    sections = detect_sections(pages)

    assert metadata.page_count == 2
    assert [page.page_number for page in pages] == [1, 2]
    assert "\n" in pages[0].text
    assert "\\n" not in pages[0].text
    assert sections[0].title == "First Park"


def test_normal_text_backslashes_are_not_decoded():
    text = "A path contains \\new and has one physical\nline break."

    assert normalize_text_source(text) == text
