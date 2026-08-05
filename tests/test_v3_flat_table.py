"""Reading tables that conversion flattened into running text."""

from __future__ import annotations

from app.v3.flat_table import parse_flat_table

# The shape a PDF leaves behind: caption, headings and cells on one line, with
# spaces standing in for thousands separators and a dash for a missing value.
OBSERVATIONS = (
    "Number of observations per country Country Nb. of employees Nb. of companies "
    "Belgium 2 321 410 Canada 2 165 411 France 2 184 406 Germany 2 300 408 "
    "Italy* - 403 Japan 2 096 413 Korea 2 016 409 Total 31 762 6 112 "
    "Note: Italy is not covered in the employee-level survey."
)


def test_space_thousands_are_grouped_by_the_column_count():
    columns, rows = parse_flat_table(OBSERVATIONS)

    assert columns == ["country", "nb_of_employees", "nb_of_companies"]
    values = dict(rows)
    # "2 321 410" is two cells, not three: the three-digit group continues the
    # number before it only as far as the column count allows.
    assert values["Belgium"] == [2321.0, 410.0]
    assert values["Germany"] == [2300.0, 408.0]
    assert values["Korea"] == [2016.0, 409.0]
    assert values["Total"] == [31762.0, 6112.0]


def test_a_missing_cell_keeps_the_row_aligned():
    _, rows = parse_flat_table(OBSERVATIONS)

    assert dict(rows)["Italy*"] == [None, 403.0]


def test_the_note_is_not_read_as_rows():
    _, rows = parse_flat_table(OBSERVATIONS)

    assert all("Italy is not covered" not in label for label, _ in rows)


def test_a_list_of_tables_is_not_a_table():
    # A contents page pairs each table's title with the page it starts on.
    listing = (
        "OLS regression estimates of regional accessibility 107 "
        "Table 5.1. Number of observations per country 257 "
        "Table 5.2. Main conditions under which clauses are upheld 270 "
        "Table 6.1. The design of the indicator of protection 339"
    )

    assert parse_flat_table(listing) == ([], [])


def test_prose_containing_numbers_is_not_a_table():
    prose = (
        "Employment rose in 2019 and again in 2023, while participation among "
        "workers aged 55 to 64 reached 61 percent in most member countries."
    )

    assert parse_flat_table(prose) == ([], [])


def test_a_contents_list_is_found_whatever_precedes_its_heading():
    """The heading opens the page; it need not be its first characters."""
    from app.v3.table_compiler import _CONTENTS_HEADING_RE

    assert _CONTENTS_HEADING_RE.match("Contents Foreword v")
    # A running folio, a page number, and "Table of" all come first here.
    assert _CONTENTS_HEADING_RE.match("9 Table of contents Foreword 3")
    # The word inside a sentence does not open a contents list.
    assert not _CONTENTS_HEADING_RE.match("The park contents are varied")


def test_a_contents_entry_may_name_its_kind_and_close_its_number():
    """One list, written two ways: "1.1 Title 21" and "Figure 1.1. Title 21"."""
    from app.v3.table_compiler import _CONTENTS_ENTRY_RE

    plain = "1.1 GDP growth remained resilient 21 1.2 Unemployment stays low 22"
    captioned = (
        "Figure 1.1. GDP growth remained resilient 21 "
        "Figure 1.2. Unemployment stays low 22"
    )

    for body in (plain, captioned):
        found = [
            (match.group("identifier"), int(match.group("page")))
            for match in _CONTENTS_ENTRY_RE.finditer(body)
        ]
        assert found == [("1.1", 21), ("1.2", 22)]


def test_the_word_in_a_sentence_does_not_start_a_contents_section():
    """A section heading is set in capitals; a sentence's word is not."""
    from app.v3.table_compiler import _CONTENTS_GROUP_RE

    prose = "Annex 5.B. Additional figures and tables 308"
    heading = "FIGURES Figure 1.1. GDP growth remained resilient 21"

    assert not list(_CONTENTS_GROUP_RE.finditer(prose))
    assert [m.group("group") for m in _CONTENTS_GROUP_RE.finditer(heading)] == ["FIGURES"]


def test_a_cell_may_carry_the_unit_it_is_printed_with():
    """Percentages are cells; the sign belongs to the column, not the number."""
    body = (
        "Efficiency of weighting Country Efficiency "
        "Belgium 76% Canada 88% France 84% Germany 90% "
        "New Zealand 45% Switzerland 44% United Kingdom 89%"
    )

    columns, rows = parse_flat_table(body)

    assert columns == ["country", "efficiency"]
    assert dict(rows)["Switzerland"] == [44.0]
    assert min(rows, key=lambda item: item[1][0])[0] == "Switzerland"
