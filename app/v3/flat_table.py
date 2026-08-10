"""Reconstruct tables that PDF extraction flattened into running text.

A table that survives conversion as Markdown pipes parses itself. A table that
does not arrives as one long line — caption, column headings and every cell
run together — and nothing downstream can count or compare over it. The
questions that need it are exactly the ones Python should answer: totals,
maxima, and figures cited twice in different places.

Two things make the flattened form parseable without knowing the subject.
Rows repeat one shape, a label followed by the same number of cells, so the
cell count can be recovered by trying each width and keeping the one that
parses the most rows. And where a number was printed with spaces for
thousands, only some groupings read as whole numbers, so the width that was
recovered also fixes where each number begins and ends.
"""

from __future__ import annotations

import re

# A cell may carry the unit it is printed with, most often a percent sign.
_CELL = r"(?:(?:\d[\d ]*\d|\d)(?:\.\d+)?%?|-|\.\.|n/a|na)"
_TERMINATORS = re.compile(
    r"\b(?:Note|Notes|Source|Sources|StatLink)\s*:",
    re.IGNORECASE,
)
# A table's own body never announces another table. Where one follows, the
# text is a list of tables rather than a table, and its rows would be titles
# paired with the page each appears on.
_NEXT_CAPTION = re.compile(
    r"(?:Annex\s+)?(?:Table|Figure|Box)\s+\d+(?:\.[A-Za-z0-9]+)*\.\s",
)
_MAX_CELLS = 8
# Three rows of numbers occur in prose by accident; a table has more.
_MIN_ROWS = 4
# A row label names one entity. A phrase this long is a sentence or a title.
_MAX_LABEL = 48
# Share of the body that must belong to rows for the text to be a table.
_MIN_ROW_COVERAGE = 0.6
# Share of row labels that must read as names rather than clause fragments.
_MIN_NAMED_LABELS = 0.8


def _thousands_groupings(tokens: list[str], width: int) -> list[float] | None:
    """Split numeric tokens into `width` numbers, honouring space thousands.

    "2 321 410" is two numbers, 2321 and 410, because a group of exactly three
    digits continues the number before it. Only the grouping that yields the
    expected number of cells is a reading of the row.
    """
    if not tokens or width <= 0:
        return None

    def spans(index: int) -> list[tuple[str, int]]:
        """Every reading of the number starting here, longest first.

        A group of exactly three digits continues the number before it, so
        "2 321" is one number and "2 321 410" could be one or two. Preferring
        the longest reading that still leaves a number for every remaining
        column is what makes "2 321 410" two cells rather than three.
        """
        readings = [(tokens[index], index + 1)]
        offset = index + 1
        while offset < len(tokens) and len(tokens[offset]) == 3:
            readings.append((f"{readings[-1][0]}{tokens[offset]}", offset + 1))
            offset += 1
        return list(reversed(readings))

    def walk(index: int, remaining: int) -> list[float] | None:
        if remaining == 0:
            return [] if index == len(tokens) else None
        if index >= len(tokens):
            return None
        for span, offset in spans(index):
            value = _as_number(span)
            if value is None:
                continue
            tail = walk(offset, remaining - 1)
            if tail is not None:
                return [value, *tail]
        return None

    return walk(0, width)


def _as_number(digits: str) -> float | None:
    try:
        return float(digits)
    except ValueError:
        return None


def _row_candidates(body: str) -> list[tuple[str, list[str], int]]:
    """Split a flattened table body into label and raw-cell runs."""
    rows: list[tuple[str, list[str], int]] = []
    pattern = re.compile(
        rf"(?P<label>(?:[^\d\s][^\d]*?))\s+(?P<cells>{_CELL}(?:\s+{_CELL})*)(?=\s+[^\d\s]|\s*$)"
    )
    for match in pattern.finditer(body):
        label = re.sub(r"\s+", " ", match.group("label")).strip(" ,;")
        # A printed unit belongs to the column, not to the number.
        cells = [cell.rstrip("%") for cell in match.group("cells").split()]
        if label and cells:
            rows.append((label, cells, match.end() - match.start()))
    return rows


def _blank_cells(cells: list[str], width: int) -> list[float | None] | None:
    """Read a row that writes a missing value as a dash or ellipsis."""
    marks = [cell for cell in cells if not cell[0].isdigit()]
    if not marks:
        return None
    numeric = [cell for cell in cells if cell[0].isdigit()]
    values = _thousands_groupings(numeric, width - len(marks))
    if values is None:
        return None
    reading: list[float | None] = []
    numbers = iter(values)
    for cell in cells:
        reading.append(None if not cell[0].isdigit() else next(numbers))
    # Merged thousands mean fewer readings than raw cells; pad from the left.
    return reading[:width] if len(reading) >= width else None


def _read_row(cells: list[str], width: int) -> list[float | None] | None:
    if any(not cell[0].isdigit() for cell in cells):
        return _blank_cells(cells, width)
    values = _thousands_groupings(cells, width)
    return list(values) if values is not None else None


def _best_width(candidates: list[tuple[str, list[str], int]]) -> int:
    """Return the cell count that reads the most rows consistently."""
    best_width, best_score = 0, 0
    for width in range(1, _MAX_CELLS + 1):
        score = sum(
            1 for _, cells, _span in candidates if _read_row(cells, width) is not None
        )
        if score > best_score:
            best_width, best_score = width, score
    return best_width if best_score >= _MIN_ROWS else 0


def _column_names(header: str, width: int) -> list[str]:
    """Name the columns from the headings that precede the first row.

    The caption's title, the column headings and the first row's own label all
    arrive as one run of text. Each of those starts a new capitalised phrase,
    so dropping the trailing row label leaves the headings at the end and the
    title before them.
    """
    parts = _capitalised_parts(header)[:-1]
    headings = parts[-(width + 1):] if len(parts) > width else []
    if len(headings) != width + 1:
        return ["label", *(f"column_{index}" for index in range(1, width + 1))]
    return [_slug(name) for name in headings]


def _capitalised_parts(header: str) -> list[str]:
    return [
        part.strip(" .,")
        for part in re.split(r"(?<=[a-z.)*])\s+(?=[A-Z])", header)
        if part.strip(" .,")
    ]


def _slug(name: str) -> str:
    text = re.sub(r"[^a-z0-9]+", "_", name.casefold()).strip("_")
    return text[:60] or "column"


def parse_flat_table(body: str) -> tuple[list[str], list[tuple[str, list[float | None]]]]:
    """Return column names and rows for one flattened table body.

    An empty result means the text did not read as a table, which is the
    honest outcome for prose that merely contains numbers.
    """
    body = _TERMINATORS.split(body)[0]
    body = _NEXT_CAPTION.split(body)[0]
    # The first row's label arrives fused to the caption and the column
    # headings, so judge every label by the entity name at the end of it.
    candidates = [
        item
        for item in _row_candidates(body)
        if len(_trailing_label(item[0])) <= _MAX_LABEL
    ]
    width = _best_width(candidates)
    if not width:
        return [], []
    rows: list[tuple[str, list[float | None]]] = []
    header = ""
    consumed = 0
    for label, cells, span in candidates:
        values = _read_row(cells, width)
        if values is None:
            continue
        consumed += span
        if not rows:
            # Everything before the first readable row is caption and headings.
            header = label
            label = _trailing_label(label)
        rows.append((label, values))
    if len(rows) < _MIN_ROWS:
        return [], []
    # In a table almost every character belongs to a row. In prose that merely
    # contains numbers, the matches are scattered fragments, and reading those
    # as a table invents figures nobody tabulated.
    if consumed < len(body.strip()) * _MIN_ROW_COVERAGE:
        return [], []
    if len({label for label, _ in rows}) < len(rows) * 0.8:
        return [], []
    # A row label names something. Prose broken at its numbers leaves fragments
    # that trail off mid-clause — "rose in", "aged" — and reading those as a
    # table would tabulate figures nobody put in a table.
    named = sum(1 for label, _ in rows if label[:1].isupper())
    if named < len(rows) * _MIN_NAMED_LABELS:
        return [], []
    return _column_names(header, width), rows


def _trailing_label(header: str) -> str:
    """Take the first row's own label back off the end of the header run."""
    parts = _capitalised_parts(header)
    return parts[-1] if parts else header.strip()
