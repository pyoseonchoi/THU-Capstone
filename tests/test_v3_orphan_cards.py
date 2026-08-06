"""A chapter's fact card may be printed on the spread that faces its opening."""

from __future__ import annotations

from app.schemas import DocumentPage
from app.v3.compiler import _claim_orphan_fact_cards as _claim_cards
from app.v3.compiler import _page_headings


def _pages(cards: set[int], last: int) -> list[DocumentPage]:
    return [
        DocumentPage(
            page_number=number,
            text=("## Park in numbers\n\n100\nArea covered (sq km)\n"
                  if number in cards else "Prose about the chapter.\n"),
        )
        for number in range(1, last + 1)
    ]


def _claim_orphan_fact_cards(pages, segments, boilerplate=frozenset()):
    """Call the claim with the page lookups the compiler builds for it."""
    return _claim_cards(
        pages,
        {page.page_number: page for page in pages},
        _page_headings(pages),
        set(boilerplate),
        segments,
    )


def test_a_card_no_chapter_reads_goes_to_the_chapter_it_faces():
    # The second chapter's card sits on page 6, the last page of the first
    # chapter's range, so the first reads its own card and the second none.
    pages = _pages(cards={2, 6}, last=10)
    segments = [(1, 6, "First"), (7, 10, "Second")]

    assert _claim_orphan_fact_cards(pages, segments) == [
        (1, 5, "First"),
        (6, 10, "Second"),
    ]


def test_a_chapter_with_its_own_card_is_left_alone():
    pages = _pages(cards={2, 8}, last=10)
    segments = [(1, 6, "First"), (7, 10, "Second")]

    assert _claim_orphan_fact_cards(pages, segments) == segments


def test_the_only_card_in_a_range_is_never_taken_from_it():
    # Moving this card would leave the first chapter with none.
    pages = _pages(cards={6}, last=10)
    segments = [(1, 6, "First"), (7, 10, "Second")]

    assert _claim_orphan_fact_cards(pages, segments) == segments


def test_a_card_far_from_the_opening_page_is_not_claimed():
    # Too far ahead to be the facing page; it belongs where it sits.
    pages = _pages(cards={2, 3}, last=20)
    segments = [(1, 12, "First"), (13, 20, "Second")]

    assert _claim_orphan_fact_cards(pages, segments) == segments


def test_a_chapter_that_moves_onto_its_card_is_renamed_from_that_page():
    # The card faces the text it belongs to, so claiming it usually claims the
    # chapter's opening page. The old name was read from the page the chapter
    # used to start on, which is now the page after it.
    pages = [
        DocumentPage(page_number=1, text="## Park in numbers\n\n100\nArea (sq km)\n"),
        DocumentPage(page_number=2, text="Prose about the first chapter.\n"),
        DocumentPage(
            page_number=3,
            text="## Beta Station\n\n## Park in numbers\n\n200\nArea (sq km)\n",
        ),
        DocumentPage(page_number=4, text="## Hotel Bellevue\n\nA warm welcome.\n"),
        DocumentPage(page_number=5, text="More prose.\n"),
    ]
    segments = [(1, 3, "First"), (4, 5, "Hotel Bellevue")]

    assert _claim_orphan_fact_cards(pages, segments) == [
        (1, 2, "First"),
        (3, 5, "Beta Station"),
    ]


def test_the_number_of_chapters_never_changes():
    pages = _pages(cards={2, 6, 11}, last=16)
    segments = [(1, 6, "A"), (7, 11, "B"), (12, 16, "C")]

    adjusted = _claim_orphan_fact_cards(pages, segments)

    assert len(adjusted) == len(segments)
    assert adjusted[0][0] == 1 and adjusted[-1][1] == 16
    # Ranges stay contiguous and non-overlapping.
    for (_, end, _), (start, _, _) in zip(adjusted, adjusted[1:]):
        assert start == end + 1


def test_a_nameless_chapter_takes_back_the_page_that_names_it():
    """A chapter opening on a title page must not lose it to the chapter before."""
    from app.v3.compiler import _claim_opening_titles

    pages = [
        DocumentPage(page_number=1, text="## First Chapter\n\nProse.\n"),
        DocumentPage(page_number=2, text="More prose about the first.\n"),
        DocumentPage(page_number=3, text="## Second Chapter\n"),
        DocumentPage(page_number=4, text="Prose about the second.\n"),
    ]
    # Segmentation put the boundary after the page that names the second.
    segments = [(1, 3, "First Chapter"), (4, 4, "")]

    assert _claim_opening_titles(pages, segments, set()) == [
        (1, 2, "First Chapter"),
        (3, 4, "Second Chapter"),
    ]


def test_a_chapter_that_already_has_a_name_keeps_its_pages():
    from app.v3.compiler import _claim_opening_titles

    pages = [
        DocumentPage(page_number=1, text="## First Chapter\n\nProse.\n"),
        DocumentPage(page_number=2, text="## Second Chapter\n\nProse.\n"),
    ]
    segments = [(1, 1, "First Chapter"), (2, 2, "Second Chapter")]

    assert _claim_opening_titles(pages, segments, set()) == segments


def test_the_chapter_before_is_never_left_without_a_page():
    from app.v3.compiler import _claim_opening_titles

    pages = [
        DocumentPage(page_number=1, text="## Only Page Of First\n"),
        DocumentPage(page_number=2, text="Prose about the second.\n"),
    ]
    segments = [(1, 1, "Only Page Of First"), (2, 2, "")]

    assert _claim_opening_titles(pages, segments, set()) == segments
