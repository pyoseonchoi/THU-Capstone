"""Question-independent compilation of contents lists and flattened tables."""

from __future__ import annotations

import re
from collections import defaultdict

from app.schemas import DocumentPage
from app.v3.models import CompiledTable, CompiledTableRow, ContentsEntry

_TABLE_MARKER_RE = re.compile(r"\bT\s*A\s*B\s*L\s*E\s+(?P<number>\d{1,3})\b", re.I)
_CONTENTS_GROUP_RE = re.compile(
    r"\b(?P<group>B\s*O\s*X\s*E\s*S|S\s*P\s*O\s*T\s*L\s*I\s*G\s*H\s*T\s*S|"
    r"F\s*I\s*G\s*U\s*R\s*E\s*S|T\s*A\s*B\s*L\s*E\s*S)\b",
    re.I,
)
_CONTENTS_ENTRY_RE = re.compile(
    r"(?<![A-Za-z0-9])(?P<identifier>(?:[OS]\.)?S?\d+(?:\.[A-Z])?(?:\.\d+){1,3})\s+"
    r"(?P<title>.+?)\s+(?P<page>\d{1,3})"
    r"(?=\s+(?:(?:[OS]\.)?S?\d+(?:\.[A-Z])?(?:\.\d+){1,3})\s+|$)",
    re.I | re.S,
)
_CONTENTS_IDENTIFIER = (
    r"(?:[OS]\.\d+(?:\.\d+){0,2}|S?\d+(?:\.[A-Z])?(?:\.\d+){1,3})"
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
_HDI_COMPONENT_RE = re.compile(
    r"^\s*(?P<life>\d{2,3}\.\d|\.\.)\s+"
    r"(?:[a-z]\s+)?"
    r"(?P<expected>\d{1,2}\.\d|\.\.)(?:\s+[a-z])?\s+"
    r"(?P<mean>\d{1,2}\.\d|\.\.)(?:\s+[a-z])?\s+"
    r"(?P<gni>\d{1,3}(?:,\d{3})+|\.\.)(?:\s+[a-z])?\b",
    re.I,
)
_HDI_GROUPS = (
    "Very high human development",
    "High human development",
    "Medium human development",
    "Low human development",
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
            if not columns or len(cells) < 2:
                continue
            values = dict(zip(columns[1:], cells[1:]))
            rows.append(CompiledTableRow(
                row_id=f"{table_id}-row-{len(rows) + 1:03d}",
                label=cells[0],
                page=page.page_number,
                values=values,
                quote=line,
            ))
    return columns, rows


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
        if len(run) == 1 and "hdirank" not in table_signal and "|" not in raw_text:
            continue
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
            title=title,
            page_start=run[0].page_number,
            page_end=run[-1].page_number,
            columns=columns,
            rows=rows,
            raw_text=raw_text,
            trusted=trusted,
            warnings=warnings,
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
            if re.search(r"\b(?:contents|table\s+of\s+contents)\b", page.text[:240], re.I)
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
