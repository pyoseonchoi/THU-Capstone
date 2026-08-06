"""A chapter is named by the page that opens it, not by what it lists."""

from __future__ import annotations

from app.schemas import DocumentPage
from app.v3.compiler import (
    _claim_named_openings,
    _cycle_segments,
    _opening_title,
)


def _page(number: int, text: str) -> DocumentPage:
    return DocumentPage(page_number=number, text=text)


def test_an_item_a_section_lists_never_names_the_chapter():
    # Guest houses, walks and dishes are set as headings exactly like a
    # chapter title. Only their position tells them apart: they follow the
    # section that introduces them, and a chapter's name precedes it.
    page = _page(1, "## Stay here...\n\n## Hotel Bellevue\n\nA warm welcome.\n")

    assert _opening_title([page], {"Stay here..."}) == ""


def test_a_name_above_the_furniture_still_names_the_chapter():
    page = _page(1, "Alpha Station\n\nProse.\n\n## Stay here...\n\n## Hotel Bellevue\n")

    assert _opening_title([page], {"Stay here..."}) == "Alpha Station"


def test_a_caption_where_a_title_would_sit_names_nothing():
    # A page can open with a caption running on from the previous spread.
    page = _page(1, "the ridge seems to nudge the clouds above the valley floor.\n")

    assert _opening_title([page], set()) == ""


def test_a_fact_card_label_never_becomes_the_chapter_name():
    # On a page carrying only a card, the first label reads enough like a
    # name to be taken for one.
    page = _page(1, "## Station in numbers\n\n100\nArea covered (sq km)\n")

    assert _opening_title([page], set()) == ""


def test_a_nameless_chapter_starts_on_the_page_inside_it_that_names_one():
    pages = [
        _page(1, "## Alpha Station\n\nProse.\n"),
        _page(2, "More prose continuing the first.\n"),
        _page(3, "## Beta Station\n\nProse.\n"),
        _page(4, "Prose about the second.\n"),
    ]
    # Segmentation cut early, so the second range opens inside the first.
    segments = [(1, 1, "Alpha Station"), (2, 4, "")]

    assert _claim_named_openings({p.page_number: p for p in pages}, set(), segments) == [
        (1, 2, "Alpha Station"),
        (3, 4, "Beta Station"),
    ]


def test_the_chapter_before_keeps_the_page_it_starts_on():
    pages = [
        _page(1, "## Alpha Station\n\nProse.\n"),
        _page(2, "## Beta Station\n\nProse.\n"),
    ]
    segments = [(1, 1, "Alpha Station"), (2, 2, "")]

    # Moving the boundary would leave the first chapter with no pages at all.
    assert _claim_named_openings({p.page_number: p for p in pages}, set(), segments) == segments


def _tied_marker_pages() -> list[DocumentPage]:
    """Three chapters whose furniture repeats two headings equally often.

    "Furniture A" sits on each chapter's opening page and "Furniture B" on the
    page after it. Only one of them cuts the document where the chapters
    actually begin, so which is chosen decides whether the chapters are named.
    """
    pages: list[DocumentPage] = []
    for index, name in enumerate(("Alpha", "Beta", "Gamma")):
        first = index * 2 + 1
        pages.append(_page(first, f"## {name} Station\n\n## Furniture A\n"))
        pages.append(_page(first + 1, f"## Furniture B\n\n## Item {index}\n"))
    return pages


def test_equally_frequent_furniture_is_resolved_by_which_names_more_chapters():
    # Both headings repeat once per chapter, so frequency cannot separate
    # them. Reading them out of a set left the choice to the interpreter and
    # compiled the same document two ways on two runs.
    assert _cycle_segments(_tied_marker_pages()) == [
        (1, 2, "Alpha Station"),
        (3, 4, "Beta Station"),
        (5, 6, "Gamma Station"),
    ]


def test_the_same_document_always_segments_the_same_way():
    pages = _tied_marker_pages()

    assert _cycle_segments(pages) == _cycle_segments(pages)
