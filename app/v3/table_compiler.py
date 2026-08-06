"""Question-independent compilation of contents lists and flattened tables."""

from __future__ import annotations

import re
from collections import Counter, defaultdict

from app.schemas import DocumentPage
from app.v3.models import CompiledTable, CompiledTableRow, ContentsEntry

_TABLE_MARKER_RE = re.compile(r"\bT\s*A\s*B\s*L\s*E\s+(?P<number>\d{1,3})\b", re.I)
_CONTENTS_GROUP_RE = re.compile(
    r"\b(?P<group>B\s*O\s*X\s*E\s*S|S\s*P\s*O\s*T\s*L\s*I\s*G\s*H\s*T\s*S|"
    r"F\s*I\s*G\s*U\s*R\s*E\s*S|T\s*A\s*B\s*L\s*E\s*S)\b",
    re.I,
)
_CONTENTS_ENTRY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<identifier>(?:[OS]\.)?S?\d+(?:\.\d+){1,3})\.?\s+"
    r"(?P<title>.+?)\s+(?P<page>\d{1,3})(?=\s+(?:(?:[OS]\.)?S?\d+(?:\.\d+){1,3})\s+|$)",
    re.I | re.S,
)
_CONTENTS_IDENTIFIER = (
    r"(?:[OS]\.\d+(?:\.\d+){0,2}|\d+\.[A-Z](?:\.\d+){1,2}|S?\d+(?:\.\d+){1,3})"
)
_BODY_CAPTION_RE = re.compile(
    rf"\b(?P<group>Box|Spotlight|Figure|Table)\s+"
    rf"(?P<identifier>{_CONTENTS_IDENTIFIER})\b",
    re.I,
)
_HDI_ROW_START_RE = re.compile(
    r"(?<![\d.,])(?P<rank>[1-9]\d{0,2})\s+"
    r"(?P<label>[A-Z][^\d]+?)\s+(?P<hdi>0\.\d{3})(?=\s)",
)
_FOOTNOTE_RE = r"(?:\s+[a-z]+(?:,\s*[a-z]+)*)?"
_HDI_COMPONENT_RE = re.compile(
    rf"^\s*(?P<life>\d{{2,3}}\.\d|\.\.){_FOOTNOTE_RE}\s+"
    rf"(?P<expected>\d{{1,2}}\.\d|\.\.){_FOOTNOTE_RE}\s+"
    rf"(?P<mean>\d{{1,2}}\.\d|\.\.){_FOOTNOTE_RE}\s+"
    rf"(?P<gni>\d{{1,3}}(?:,\d{{3}})+|\.\.){_FOOTNOTE_RE}\b",
    re.I,
)
_HDI_GROUPS = (
    "Very high human development",
    "High human development",
    "Medium human development",
    "Low human development",
)
_CONTENTS_HEADING_RE = re.compile(
    r"^[^A-Za-z]{0,25}(?:table\s+of\s+)?contents\b",
    re.I,
)


def _compact_heading(text: str) -> str:
    return re.sub(r"\s+", "", text).casefold()


def _table_number(text: str) -> int | None:
    match = _TABLE_MARKER_RE.search(text[:180])
    return int(match.group("number")) if match else None


def _table_title(text: str, number: int) -> str:
    match = _TABLE_MARKER_RE.search(text[:500])
    if not match:
        return f"Table {number}"
    tail = text[match.end() : match.end() + 240].strip()
    stop = re.search(
        r"\b(?:SDG|HDI RANK|Human Development Life expectancy|Change in Human)",
        tail,
        re.I,
    )
    title = tail[: stop.start()].strip(" /:-") if stop else tail[:100].strip(" /:-")
    return f"Table {number} {title}".strip()


def _optional_number(value: str) -> float | None:
    if not value or value == "..":
        return None
    return float(value.replace(",", ""))


def _group_before(text: str, offset: int, current: str) -> str:
    prefix = text[:offset].casefold()
    pattern = re.compile(
        r"\b(very high human development|(?<!very )high human development|"
        r"medium human development|low human development)\b",
        re.I,
    )
    canonical = {group.casefold(): group for group in _HDI_GROUPS}
    positions = [
        (match.start(), canonical[match.group(1).casefold()])
        for match in pattern.finditer(prefix)
    ]
    return max(positions, default=(-1, current))[1]


def _parse_hdi_component_rows(
    table_id: str,
    pages: list[DocumentPage],
) -> list[CompiledTableRow]:
    """Parse the common HDI component table after PDF columns were flattened."""
    rows: list[CompiledTableRow] = []
    current_group = ""
    seen_labels: set[str] = set()
    for page in pages:
        matches = list(_HDI_ROW_START_RE.finditer(page.text))
        for index, match in enumerate(matches):
            label = re.sub(r"\s+", " ", match.group("label")).strip(" ,")
            if not label or label.casefold() in seen_labels:
                continue
            current_group = _group_before(page.text, match.start(), current_group)
            end = matches[index + 1].start() if index + 1 < len(matches) else len(page.text)
            tail = page.text[match.end() : end]
            tail = re.split(
                r"\b(?:Other countries or territories|Human development groups|"
                r"Developing countries|Regions|Continued)\b",
                tail,
                maxsplit=1,
                flags=re.I,
            )[0]
            components = _HDI_COMPONENT_RE.match(tail)
            values: dict[str, float | None] = {
                "hdi_2023": float(match.group("hdi")),
                "life_expectancy_2023": None,
                "expected_schooling_years_2023": None,
                "mean_schooling_years_2023": None,
                "gni_per_capita_2023": None,
            }
            if components:
                values.update({
                    "life_expectancy_2023": _optional_number(components.group("life")),
                    "expected_schooling_years_2023": _optional_number(
                        components.group("expected")
                    ),
                    "mean_schooling_years_2023": _optional_number(components.group("mean")),
                    "gni_per_capita_2023": _optional_number(components.group("gni")),
                })
            quote_end = components.end() if components else min(len(tail), 220)
            quote = re.sub(
                r"\s+",
                " ",
                page.text[match.start() : match.end() + quote_end],
            ).strip()
            rank = int(match.group("rank"))
            rows.append(CompiledTableRow(
                row_id=f"{table_id}-row-{len(rows) + 1:03d}",
                label=label,
                page=page.page_number,
                rank=rank,
                group=current_group,
                values=values,
                quote=quote,
            ))
            seen_labels.add(label.casefold())
    return rows


def _parse_markdown_rows(
    table_id: str,
    pages: list[DocumentPage],
) -> tuple[list[str], list[CompiledTableRow]]:
    columns: list[str] = []
    rows: list[CompiledTableRow] = []
    for page in pages:
        lines = [line.strip() for line in page.text.splitlines() if "|" in line]
        for index, line in enumerate(lines):
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2 or all(re.fullmatch(r":?-+:?", cell) for cell in cells):
                continue
            if not columns and index + 1 < len(lines) and re.search(r"\|\s*:?-+", lines[index + 1]):
                columns = [re.sub(r"\W+", "_", cell.casefold()).strip("_") for cell in cells]
                continue
            if not columns or len(cells) != len(columns):
                continue
            values = dict(zip(columns[1:], cells[1:], strict=True))
            rows.append(CompiledTableRow(
                row_id=f"{table_id}-row-{len(rows) + 1:03d}",
                label=cells[0],
                page=page.page_number,
                values=values,
                quote=line,
            ))
    return columns, rows


_LABELED_DATE_RE = re.compile(r"^\d{1,2}-[A-Za-z]{3}-\d{2,4}$")
_LABELED_NUMBER_RE = re.compile(r"^\d{1,3}(?:,\d{3})+(?:\.\d+)?%?$|^\d+(?:\.\d+)?%?$")
_LABELED_STOP_WORDS = {"total", "totals", "note", "notes", "source"}
_LABELED_MAX_LABEL_WORDS = 6


def _is_labeled_value(token: str) -> bool:
    """Whether a whitespace-delimited token is a data value (number, percent,
    or short date) rather than part of a row's label."""
    stripped = token.rstrip(",;").replace(" ", "")
    return bool(
        _LABELED_DATE_RE.fullmatch(token.rstrip(","))
        or _LABELED_NUMBER_RE.fullmatch(stripped)
    )


def _merge_grouped_number_tokens(tokens: list[str]) -> list[str]:
    """Rejoin a bare 1-3 digit token immediately followed by an exact 3-digit
    token into one number, undoing a space-as-thousands-separator split
    (e.g. "2 321" -> one value, not two)."""
    merged: list[str] = []
    index = 0
    while index < len(tokens):
        if (
            index + 1 < len(tokens)
            and re.fullmatch(r"\d{1,3}", tokens[index])
            and re.fullmatch(r"\d{3}", tokens[index + 1])
        ):
            merged.append(f"{tokens[index]} {tokens[index + 1]}")
            index += 2
            continue
        merged.append(tokens[index])
        index += 1
    return merged


def _lenient_label_value_runs(tokens: list[str]) -> list[tuple[int, int, int]]:
    """Split tokens into alternating (label-run, value-run) pairs, keeping
    each run's token-index bounds. Column count and label shape are not yet
    known, so value-runs may vary in length here -- callers reconcile that
    against the stable shape of the runs that follow the header."""
    runs: list[tuple[int, int, int]] = []
    index, total = 0, len(tokens)
    while index < total:
        label_start = index
        while index < total and not _is_labeled_value(tokens[index]):
            index += 1
        if index == label_start:
            index += 1
            continue
        value_start = index
        while index < total and _is_labeled_value(tokens[index]):
            index += 1
        runs.append((label_start, value_start, index))
    return runs


def _parse_labeled_rows(
    table_id: str,
    page: DocumentPage,
    start: int,
) -> tuple[list[str], list[CompiledTableRow]]:
    """Reconstruct rows from a table whose entries read as a flat
    "Label value value..." run with no markdown or newline structure, e.g.
    a country-by-country or company-by-company breakdown flattened by PDF
    text extraction. Row boundaries are inferred purely from the repeating
    label/value(s) rhythm -- there is no fixed vocabulary of valid labels,
    so this applies to any entity type (countries, companies, regions...)
    rather than one specific document's facts.

    The header line (e.g. "Country Response rate") is indistinguishable
    from a row's own label using shape alone, since both are runs of
    capitalized words. It is resolved by learning the typical label length
    from the rows after the first (unambiguous once past the header) and
    picking whichever split of the first row's label span comes closest to
    that length, rather than guessing from capitalization alone.
    """
    text = page.text[start:start + 4000]
    tokens = _merge_grouped_number_tokens([m.group(0) for m in re.finditer(r"\S+", text)])
    runs = _lenient_label_value_runs(tokens)
    if len(runs) < 3:
        return [], []

    # A footnote/source line after the real rows ("Note: ... 90% of firms
    # ...") often contains its own scattered numbers, which would otherwise
    # keep generating bogus "rows" out of ordinary prose. Cut everything
    # from the first such marker onward rather than filtering it row by
    # row, so it can't contaminate the label-length statistics below.
    stop_index = next(
        (
            index
            for index, (label_start, value_start, _) in enumerate(runs)
            if index > 0
            and value_start > label_start
            and tokens[label_start].strip(" *").casefold().rstrip(":") in _LABELED_STOP_WORDS
        ),
        len(runs),
    )
    runs = runs[:stop_index]
    if len(runs) < 4:
        # A page-footer boilerplate line (a StatLink URL, a copyright
        # notice) can coincidentally repeat a short "label + number"
        # rhythm two or three times; real entity-indexed tables in
        # practice run much longer than that, so a low row count is
        # treated as a sign this isn't a real table rather than a small
        # one, and the whole page is rejected.
        return [], []

    later = runs[1:]
    value_counts = [end - value_start for _, value_start, end in later]
    counter = Counter(value_counts)
    num_columns = counter.most_common(1)[0][0]
    trustworthy_later = [run for run in later if (run[2] - run[1]) == num_columns]
    if len(trustworthy_later) < 2:
        return [], []
    label_lengths = sorted(
        value_start - label_start for label_start, value_start, _ in trustworthy_later
    )
    target_label_len = label_lengths[len(label_lengths) // 2]
    if target_label_len > _LABELED_MAX_LABEL_WORDS or any(
        length > _LABELED_MAX_LABEL_WORDS for length in label_lengths
    ):
        # Real row labels (country/company/region/product names) are short
        # identifying phrases. A run this long is prose that happened to
        # contain scattered numbers -- not an entity-indexed table.
        return [], []

    rows: list[CompiledTableRow] = []
    header_words: list[str] = []
    first_label_start, first_value_start, first_value_end = runs[0]
    if first_value_end - first_value_start == num_columns:
        span = first_value_start - first_label_start
        best_k = min(range(1, span + 1), key=lambda k: abs(k - target_label_len))
        header_words = tokens[first_label_start:first_value_start - best_k]
        label = " ".join(tokens[first_value_start - best_k:first_value_start]).strip(" *")
        if label.casefold().rstrip(":") not in _LABELED_STOP_WORDS:
            rows.append((label, tokens[first_value_start:first_value_end]))

    for label_start, value_start, value_end in trustworthy_later:
        label = " ".join(tokens[label_start:value_start]).strip(" *")
        if label.casefold().rstrip(":") in _LABELED_STOP_WORDS:
            continue
        rows.append((label, tokens[value_start:value_end]))

    unique_labels = len({label.casefold() for label, _ in rows}) == len(rows)
    if len(rows) < 4 or not unique_labels:
        return [], []

    # The header phrase preceding the row-index column's own name (e.g.
    # "Country") splits at each new capitalized word into that many column
    # names ("Nb. of employees", "Nb. of companies", ...). Keep only the
    # last num_columns pieces, discarding the row-index column's own label;
    # fall back to positional names if the header shape is irregular.
    header_text = " ".join(header_words).strip()
    split_columns = [part.strip() for part in re.split(r"\s+(?=[A-Z])", header_text) if part.strip()]
    if len(split_columns) >= num_columns:
        columns = split_columns[-num_columns:]
    else:
        columns = [f"col_{index + 1}" for index in range(num_columns)]
    table_rows = [
        CompiledTableRow(
            row_id=f"{table_id}-row-{index + 1:03d}",
            label=label,
            page=page.page_number,
            values=dict(zip(columns, values, strict=True)),
            quote=f"{label} {' '.join(values)}",
        )
        for index, (label, values) in enumerate(rows)
    ]
    return columns, table_rows


def compile_tables(pages: list[DocumentPage]) -> list[CompiledTable]:
    """Find consecutive table pages and reconstruct supported row schemas."""
    page_runs: list[list[DocumentPage]] = []
    current: list[DocumentPage] = []
    current_number: int | None = None
    for page in pages:
        number = _table_number(page.text)
        consecutive = current and page.page_number == current[-1].page_number + 1
        if number is None:
            if current:
                page_runs.append(current)
                current, current_number = [], None
            continue
        if current and (not consecutive or number != current_number):
            page_runs.append(current)
            current = []
        current.append(page)
        current_number = number
    if current:
        page_runs.append(current)

    tables: list[CompiledTable] = []
    occurrences: defaultdict[int, int] = defaultdict(int)
    for run in page_runs:
        number = _table_number(run[0].text)
        if number is None:
            continue
        raw_text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}" for page in run
        )
        table_signal = _compact_heading(raw_text)
        occurrences[number] += 1
        suffix = f"-{occurrences[number]}" if occurrences[number] > 1 else ""
        table_id = f"table-{number}{suffix}"
        title = _table_title(run[0].text, number)
        columns, rows = _parse_markdown_rows(table_id, run)
        if (
            "humandevelopmentindex" in table_signal
            and "lifeexpectancy" in table_signal
            and "grossnationalincome" in table_signal
        ):
            rows = _parse_hdi_component_rows(table_id, run)
            columns = [
                "rank",
                "label",
                "group",
                "hdi_2023",
                "life_expectancy_2023",
                "expected_schooling_years_2023",
                "mean_schooling_years_2023",
                "gni_per_capita_2023",
            ]
        if len(run) == 1 and not rows:
            continue
        identifier = ""
        caption_match = _BODY_CAPTION_RE.search(run[0].text[:200])
        if caption_match and caption_match.group("group").casefold() == "table":
            identifier = caption_match.group("identifier")
        ranks = [row.rank for row in rows if row.rank is not None]
        unique_labels = len({row.label.casefold() for row in rows}) == len(rows)
        rank_consistent = bool(ranks) and abs(len(ranks) - max(ranks)) <= max(
            2, round(len(ranks) * 0.05)
        )
        trusted = bool(rows) and unique_labels and (not ranks or rank_consistent)
        warnings: list[str] = []
        if rows and not trusted:
            warnings.append("Table rows failed a uniqueness or rank continuity check")
        if not rows:
            warnings.append("Table range detected but no deterministic row schema matched")
        tables.append(CompiledTable(
            table_id=table_id,
            number=number,
            identifier=identifier,
            title=title,
            page_start=run[0].page_number,
            page_end=run[-1].page_number,
            columns=columns,
            rows=rows,
            raw_text=raw_text,
            trusted=trusted,
            warnings=warnings,
        ))

    # A page's real table caption isn't always within the narrow window
    # _table_number checks, so scan every page independently (not just the
    # ones the number-based run grouping above already picked up) for a
    # flat "label, then values, repeated" schema. This runs after, and
    # never touches, the existing HDI/markdown run-grouping logic, so it
    # can't disturb tables that mechanism already resolves correctly.
    found_identifiers = {table.identifier for table in tables if table.trusted and table.identifier}
    table_number = 0
    for page in pages:
        for match in _BODY_CAPTION_RE.finditer(page.text):
            if match.group("group").casefold() != "table":
                continue
            identifier = match.group("identifier")
            if identifier in found_identifiers:
                continue
            table_number += 1
            columns, rows = _parse_labeled_rows(
                f"labeled-table-{table_number}", page, match.end()
            )
            if not rows:
                continue
            found_identifiers.add(identifier)
            leading_number_match = re.match(r"\d+", identifier)
            blurb = page.text[match.end():match.end() + 80].strip()
            tables.append(CompiledTable(
                table_id=f"labeled-table-{table_number}",
                number=int(leading_number_match.group(0)) if leading_number_match else None,
                identifier=identifier,
                title=f"Table {identifier} {blurb}".strip(),
                page_start=page.page_number,
                page_end=page.page_number,
                columns=columns,
                rows=rows,
                raw_text=page.text,
                trusted=True,
                warnings=[],
            ))
    return tables


def compile_contents(
    pages: list[DocumentPage],
) -> tuple[str, list[int], list[ContentsEntry], bool]:
    """Preserve the contents range and parse straightforward single-column lists."""
    start_index = next(
        (
            index
            for index, page in enumerate(pages)
            if _CONTENTS_HEADING_RE.match(page.text)
        ),
        None,
    )
    if start_index is None:
        return "", [], [], False
    selected: list[DocumentPage] = []
    for page in pages[start_index : start_index + 10]:
        leading = _compact_heading(page.text[:180])
        if selected and leading.startswith(("overview", "introduction", "chapter1")):
            break
        selected.append(page)
    text = "\n\n".join(f"[Page {page.page_number}]\n{page.text}" for page in selected)
    headings = list(_CONTENTS_GROUP_RE.finditer(text))
    entries: list[ContentsEntry] = []
    aliases = {
        "boxes": "boxes",
        "spotlights": "spotlights",
        "figures": "figures",
        "tables": "tables",
    }
    for index, heading in enumerate(headings):
        key = aliases.get(_compact_heading(heading.group("group")))
        if not key:
            continue
        end = headings[index + 1].start() if index + 1 < len(headings) else len(text)
        body = text[heading.end() : end]
        for match in _CONTENTS_ENTRY_RE.finditer(body):
            entries.append(ContentsEntry(
                category=key,
                identifier=match.group("identifier"),
                title=re.sub(r"\s+", " ", match.group("title")).strip(),
                page=int(match.group("page")),
            ))
    contents_pages = [page.page_number for page in selected]
    group_names = {
        "box": "boxes",
        "spotlight": "spotlights",
        "figure": "figures",
        "table": "tables",
    }
    caption_entries: list[ContentsEntry] = []
    seen_captions: set[tuple[str, str]] = set()
    for page in pages:
        if page.page_number in contents_pages:
            continue
        for match in _BODY_CAPTION_RE.finditer(page.text):
            category = group_names[match.group("group").casefold()]
            identifier = match.group("identifier")
            key = category, identifier.casefold()
            if key in seen_captions or not re.search(
                rf"(?<![A-Za-z0-9]){re.escape(identifier)}(?![A-Za-z0-9])",
                text,
                re.I,
            ):
                continue
            seen_captions.add(key)
            caption_entries.append(ContentsEntry(
                category=category,
                identifier=identifier,
                page=page.page_number,
            ))

    heading_groups = {
        _compact_heading(match.group("group")) for match in _CONTENTS_GROUP_RE.finditer(text)
    }
    caption_groups = {entry.category for entry in caption_entries}
    caption_trusted = bool(heading_groups) and heading_groups <= caption_groups
    if caption_trusted:
        entries = caption_entries
    return text, contents_pages, entries, caption_trusted
