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
