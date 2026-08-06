"""A chapter's fact card may be printed on the spread that faces its opening."""

from __future__ import annotations

from app.schemas import DocumentPage
from app.v3.compiler import _claim_orphan_fact_cards


def _pages(cards: set[int], last: int) -> list[DocumentPage]:
    return [
        DocumentPage(
            page_number=number,
            text=("## Park in numbers\n\n100\nArea covered (sq km)\n"
                  if number in cards else "Prose about the chapter.\n"),
        )
        for number in range(1, last + 1)
    ]


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


def test_the_number_of_chapters_never_changes():
    pages = _pages(cards={2, 6, 11}, last=16)
    segments = [(1, 6, "A"), (7, 11, "B"), (12, 16, "C")]

    adjusted = _claim_orphan_fact_cards(pages, segments)

    assert len(adjusted) == len(segments)
    assert adjusted[0][0] == 1 and adjusted[-1][1] == 16
    # Ranges stay contiguous and non-overlapping.
    for (_, end, _), (start, _, _) in zip(adjusted, adjusted[1:]):
        assert start == end + 1
