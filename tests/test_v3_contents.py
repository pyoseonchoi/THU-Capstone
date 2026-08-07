"""A contents list heads its groups and numbers its entries its own way."""

from __future__ import annotations

from app.schemas import DocumentPage
from app.v3.table_compiler import compile_contents


def _contents(body: str) -> list:
    _text, _pages, entries, _trusted = compile_contents(
        [DocumentPage(page_number=1, text=body)]
    )
    return entries


def test_a_group_headed_in_words_opens_its_list():
    # The same section is headed "T A B L E S" in one publication and "List of
    # tables" in another. Reading only the first leaves the list unparsed, and
    # a contents index with no entries answers no counting question at all.
    entries = _contents(
        "Contents\n"
        "\n"
        "List of tables\n"
        "\n"
        "3.1  Alpha coverage by region   68\n"
        "4.1  Beta reforms by site   87\n"
    )

    assert [(entry.category, entry.identifier) for entry in entries] == [
        ("tables", "3.1"),
        ("tables", "4.1"),
    ]


def test_the_same_phrase_inside_a_sentence_is_not_a_heading():
    # "...see the list of tables below" names no section. Reading it as one
    # would cut the list it sits in and file the entries after it wrongly.
    entries = _contents(
        "Contents\n"
        "\n"
        "List of figures\n"
        "\n"
        "1.1  Alpha trend   12\n"
        "For definitions see the list of tables in the annex   13\n"
        "1.2  Beta trend   14\n"
    )

    assert {entry.category for entry in entries} == {"figures"}


def test_an_annex_numbers_its_entries_after_the_annex_it_belongs_to():
    # An annex labels its items with its own name — "A2.1" sits in the same
    # list as "4.6" and is as much an entry as it is.
    entries = _contents(
        "Contents\n"
        "\n"
        "List of tables\n"
        "\n"
        "4.6  Alpha methods   182\n"
        "A2.1  Beta groupings   240\n"
        "A3.2  Gamma requirements   256\n"
    )

    assert [entry.identifier for entry in entries] == ["4.6", "A2.1", "A3.2"]


def test_a_capitalised_heading_still_opens_its_list():
    entries = _contents(
        "Contents\n"
        "\n"
        "TABLES\n"
        "\n"
        "3.1  Alpha coverage by region   68\n"
        "4.1  Beta reforms by site   87\n"
    )

    assert [entry.identifier for entry in entries] == ["3.1", "4.1"]
