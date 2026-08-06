"""A record's group comes from the document's own label, not from a word list.

The compiler keeps a list of country names to fall back on, but a document
may file its records under groups no such list could hold — a region, a
division, an invented place. Whatever the document labels them with has to
work on its own.
"""

from __future__ import annotations

from app.v3.compiler import _find_country


def test_a_label_names_a_group_the_word_list_has_never_heard_of():
    assert _find_country("COUNTRY: Uzbekistan\nA remote highland reserve.") == "Uzbekistan"
    assert _find_country("COUNTRY: Kanto Region\nA coastal facility.") == "Kanto Region"


def test_a_label_stops_at_the_end_of_its_own_line():
    # Running past the line break makes the value swallow the first word of
    # whatever follows it.
    assert _find_country("Country: Norway\nAbisko sits above the Arctic Circle.") == "Norway"
    assert _find_country("Country: Chile\nEvery visitor needs a permit.") == "Chile"


def test_a_multi_word_group_on_one_line_is_kept_whole():
    assert _find_country("Country: New South Wales\nEstablished 1994.") == "New South Wales"


def test_a_label_with_no_value_names_no_group():
    assert _find_country("COUNTRY:\nA facility with no group named.") == ""


def test_prose_that_merely_contains_the_word_is_not_a_label():
    # "country: over millennia" would be read as a group if the value were
    # matched case-insensitively.
    assert _find_country("Shaped by this country: over millennia of erosion.") == ""
