"""Question-independent compilation of repeated Markdown records and facts."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

from app.reduction.normalizer import parse_number
from app.schemas import DocumentPage
from app.v3.models import CompiledDocument, CompiledRecord, NumberFact

_RECORD_MARKER_RE = re.compile(r"(?i)\bpark\s+in\s+numbers\b")
_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")
_NUMBER_PREFIX_RE = re.compile(
    r"^(?P<raw>[<>]?\s*-?\d[\d,.]*(?:\.\d+)?)\s*(?P<label>.*)$"
)
_GENERIC_TITLE_TERMS = (
    "park in numbers",
    "camping",
    "hotel",
    "car park",
    "gateway centre",
)
_BACK_MATTER_RE = re.compile(
    r"^#{1,6}\s*(?:index|references|bibliography|acknowledgements?|credits|glossary)\b",
    re.IGNORECASE,
)

_COUNTRIES = {
    "albania": "Albania",
    "austria": "Austria",
    "bulgaria": "Bulgaria",
    "croatia": "Croatia",
    "denmark": "Denmark",
    "england": "England",
    "estonia": "Estonia",
    "finland": "Finland",
    "france": "France",
    "germany": "Germany",
    "greece": "Greece",
    "hungary": "Hungary",
    "iceland": "Iceland",
    "ireland": "Ireland",
    "italy": "Italy",
    "latvia": "Latvia",
    "lithuania": "Lithuania",
    "montenegro": "Montenegro",
    "norway": "Norway",
    "poland": "Poland",
    "portugal": "Portugal",
    "romania": "Romania",
    "scotland": "Scotland",
    "slovakia": "Slovakia",
    "slovenia": "Slovenia",
    "spain": "Spain",
    "sweden": "Sweden",
    "switzerland": "Switzerland",
    "ukraine": "Ukraine",
    "wales": "Wales",
}

_COUNTRY_ALIASES = {
    "Albania": ("albania", "albanian"),
    "Austria": ("austria", "austrian"),
    "Bulgaria": ("bulgaria", "bulgarian"),
    "Croatia": ("croatia", "croatian"),
    "Denmark": ("denmark", "danish"),
    "England": ("england", "english"),
    "Estonia": ("estonia", "estonian"),
    "Finland": ("finland", "finnish"),
    "France": ("france", "french"),
    "Germany": ("germany", "german"),
    "Greece": ("greece", "greek"),
    "Hungary": ("hungary", "hungarian"),
    "Iceland": ("iceland", "icelandic"),
    "Ireland": ("ireland", "irish"),
    "Italy": ("italy", "italian"),
    "Latvia": ("latvia", "latvian"),
    "Lithuania": ("lithuania", "lithuanian"),
    "Montenegro": ("montenegro", "montenegrin"),
    "Norway": ("norway", "norwegian"),
    "Poland": ("poland", "polish"),
    "Portugal": ("portugal", "portuguese"),
    "Romania": ("romania", "romanian"),
    "Scotland": ("scotland", "scottish", "scots"),
    "Slovakia": ("slovakia", "slovakian", "slovak"),
    "Slovenia": ("slovenia", "slovenian"),
    "Spain": ("spain", "spanish"),
    "Sweden": ("sweden", "swedish"),
    "Switzerland": ("switzerland", "swiss"),
    "Ukraine": ("ukraine", "ukrainian"),
    "Wales": ("wales", "welsh"),
}


def _clean_line(line: str) -> str:
    return html.unescape(_HEADING_PREFIX_RE.sub("", line.strip())).strip()


def _is_record_title(line: str) -> bool:
    text = _clean_line(line)
    folded = text.casefold()
    if not 3 <= len(text) <= 100 or len(text.split()) > 14:
        return False
    if text.endswith((".", ",", ":", ";")):
        return False
    if any(term in folded for term in _GENERIC_TITLE_TERMS):
        return False
    return (
        "park" in folded
        or folded.startswith("parco ")
        or folded in {"picos de europa", "slovenský raj", "slovensky raj"}
    )


def _find_country(page_text: str) -> str:
    for raw_line in page_text[:1600].splitlines():
        line = _clean_line(raw_line)
        line = re.sub(r"\d+", "", line)
        compact = re.sub(r"[^a-z]", "", line.casefold())
        if compact in _COUNTRIES:
            return _COUNTRIES[compact]

    folded = page_text.casefold()
    introduction = folded[:800]
    scores: dict[str, float] = {}
    first_positions: dict[str, int] = {}
    for country, aliases in _COUNTRY_ALIASES.items():
        score = 0.0
        positions: list[int] = []
        for alias_index, alias in enumerate(aliases):
            matches = list(re.finditer(rf"\b{re.escape(alias)}\b", folded))
            positions.extend(match.start() for match in matches)
            score += len(matches) * (2.0 if alias_index == 0 else 1.5)
            possessive = rf"\b{re.escape(alias)}(?:'s|s')\b"
            if re.search(possessive, introduction):
                score += 20.0
        if score:
            scores[country] = score
            first_positions[country] = min(positions)
    if not scores:
        return ""
    return max(
        scores,
        key=lambda country: (scores[country], -first_positions[country]),
    )


def _slug_label(label: str) -> str:
    folded = html.unescape(label).casefold()
    if "area covered" in folded:
        return "area"
    if "highest point" in folded or "highest peak" in folded:
        return "highest_point"
    if "visitor" in folded and ("annual" in folded or "per year" in folded):
        return "annual_visitors"
    if "estimated age" in folded and "tree" in folded:
        return "oldest_tree_age"
    if "first recorded eruption" in folded:
        return "first_recorded_eruption_year"
    normalized = re.sub(r"[^a-z0-9]+", "_", folded).strip("_")
    return normalized[:80] or "number"


def _extract_unit(label: str) -> str:
    folded = html.unescape(label).casefold()
    match = re.search(r"\(([^)]+)\)", folded)
    unit = match.group(1).strip() if match else ""
    aliases = {
        "m": "m",
        "sq km": "sq km",
        "sq miles": "sq miles",
        "years": "years",
        "km/h": "km/h",
    }
    if unit in aliases:
        return aliases[unit]
    if "million" in folded:
        return "visitors/year"
    if "(bc)" in folded:
        return "year_bc"
    return unit


def _extract_subject(label: str) -> str:
    text = html.unescape(label)
    if ":" in text:
        subject = text.split(":", 1)[1]
        subject = re.sub(r"\([^)]*\)", "", subject).strip()
        return subject
    match = re.search(r"tree,\s+the\s+(.+?)\s*\(years\)", text, re.IGNORECASE)
    return match.group(1).strip() if match else ""


def _normalize_card_value(raw: str, label: str) -> float | None:
    value = parse_number(raw.replace(">", "").replace("<", ""))
    if value is None:
        return None
    folded = label.casefold()
    if "million" in folded:
        value *= 1_000_000
    if "(bc)" in folded or re.search(r"\b(?:bc|bce)\b", folded):
        value = -abs(value)
    return value


def _card_lines(page_text: str) -> list[str]:
    match = _RECORD_MARKER_RE.search(page_text)
    if not match:
        return []
    tail = page_text[match.end():]
    lines: list[str] = []
    for raw in tail.splitlines():
        stripped = raw.strip()
        if stripped.startswith("##") and lines:
            break
        cleaned = _clean_line(stripped)
        if cleaned:
            lines.append(cleaned)
        if len(lines) >= 30:
            break
    return lines


def parse_number_facts(
    record_id: str,
    page_number: int,
    page_text: str,
) -> list[NumberFact]:
    """Bind each card number to the label that follows it in reading order."""
    lines = _card_lines(page_text)
    facts: list[NumberFact] = []
    pending_raw = ""
    for line in lines:
        match = _NUMBER_PREFIX_RE.match(line)
        if match:
            raw = re.sub(r"\s+", "", match.group("raw"))
            inline_label = match.group("label").strip()
            if inline_label:
                value = _normalize_card_value(raw, inline_label)
                if value is not None:
                    facts.append(NumberFact(
                        record_id=record_id,
                        field=_slug_label(inline_label),
                        label=inline_label,
                        value=value,
                        raw_value=raw,
                        unit=_extract_unit(inline_label),
                        subject=_extract_subject(inline_label),
                        page=page_number,
                        quote=f"{raw} {inline_label}",
                    ))
                pending_raw = ""
            else:
                pending_raw = raw
            continue
        if pending_raw:
            value = _normalize_card_value(pending_raw, line)
            if value is not None:
                facts.append(NumberFact(
                    record_id=record_id,
                    field=_slug_label(line),
                    label=line,
                    value=value,
                    raw_value=pending_raw,
                    unit=_extract_unit(line),
                    subject=_extract_subject(line),
                    page=page_number,
                    quote=f"{pending_raw} {line}",
                ))
            pending_raw = ""
    return facts


def _marker_pages(pages: Iterable[DocumentPage]) -> list[int]:
    return [
        page.page_number
        for page in pages
        if _RECORD_MARKER_RE.search(page.text)
    ]


def _back_matter_start(pages: list[DocumentPage], after_page: int) -> int | None:
    for page in pages:
        if page.page_number <= after_page:
            continue
        first_lines = [line.strip() for line in page.text.splitlines() if line.strip()][:8]
        if any(_BACK_MATTER_RE.match(line) for line in first_lines):
            return page.page_number
    return None


def _segment_pages(
    document_id: str,
    pages: list[DocumentPage],
    *,
    prefix: str = "segment",
    target_characters: int = 14_000,
) -> list[CompiledRecord]:
    """Build exhaustive, consecutive segments for documents without records."""
    records: list[CompiledRecord] = []
    current: list[DocumentPage] = []
    current_size = 0

    def flush() -> None:
        nonlocal current, current_size
        if not current:
            return
        first, last = current[0], current[-1]
        text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}" for page in current
        )
        heading = next(
            (
                _clean_line(line)
                for line in first.text.splitlines()
                if line.strip().startswith("#") and _clean_line(line)
            ),
            f"Pages {first.page_number}-{last.page_number}",
        )
        ordinal = len(records) + 1
        records.append(CompiledRecord(
            record_id=f"{prefix}-{ordinal:03d}",
            ordinal=ordinal,
            title=heading[:100],
            country=_find_country(text),
            page_start=first.page_number,
            page_end=last.page_number,
            anchor_page=first.page_number,
            text=text,
        ))
        current = []
        current_size = 0

    for page in pages:
        size = len(page.text)
        is_page_gap = current and page.page_number != current[-1].page_number + 1
        if current and (is_page_gap or current_size + size > target_characters):
            flush()
        current.append(page)
        current_size += size
    flush()
    return records


def all_mapping_records(document: CompiledDocument) -> list[CompiledRecord]:
    """Return every terminal text unit, including non-catalog front/back matter."""
    return [*document.records, *document.supplementary_records]


def compile_document(
    document_id: str,
    pages: list[DocumentPage],
) -> CompiledDocument:
    """Compile repeated records and numeric facts without question-time retrieval."""
    anchors = _marker_pages(pages)
    if len(anchors) < 3:
        records = _segment_pages(document_id, pages)
        return CompiledDocument(
            document_id=document_id,
            record_kind="segments",
            records=records,
            page_count=len(pages),
            warnings=["No repeated numeric-card structure detected"],
        )

    page_by_number = {page.page_number: page for page in pages}
    provisional_bounds: list[tuple[int, int]] = []
    for index, anchor in enumerate(anchors):
        start = (
            max(pages[0].page_number, anchor - 3)
            if index == 0
            else (anchors[index - 1] + anchor) // 2 + 1
        )
        end = (
            pages[-1].page_number
            if index == len(anchors) - 1
            else (anchor + anchors[index + 1]) // 2
        )
        provisional_bounds.append((start, end))

    titles: list[tuple[int, str]] = []
    warnings: list[str] = []
    for ordinal, (anchor, (start, end)) in enumerate(
        zip(anchors, provisional_bounds, strict=True),
        start=1,
    ):
        candidates: list[tuple[int, int, str]] = []
        for page_number in range(start, end + 1):
            page = page_by_number.get(page_number)
            if not page:
                continue
            for line_index, line in enumerate(page.text.splitlines()):
                if _is_record_title(line):
                    candidates.append((page_number, line_index, _clean_line(line)))
        if candidates:
            title_page, _, title = candidates[0]
        else:
            title_page, title = anchor, f"Record {ordinal}"
            warnings.append(f"Could not identify title for record {ordinal}")
        titles.append((title_page, title))

    records: list[CompiledRecord] = []
    back_matter_start = _back_matter_start(pages, anchors[-1])
    for index, ((title_page, title), anchor) in enumerate(
        zip(titles, anchors, strict=True)
    ):
        page_start = title_page
        page_end = (
            titles[index + 1][0] - 1
            if index + 1 < len(titles)
            else (
                back_matter_start - 1
                if back_matter_start is not None
                else pages[-1].page_number
            )
        )
        selected_pages = [
            page_by_number[number]
            for number in range(page_start, page_end + 1)
            if number in page_by_number
        ]
        record_id = f"record-{index + 1:03d}"
        anchor_text = page_by_number[anchor].text
        facts = parse_number_facts(record_id, anchor, anchor_text)
        text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}" for page in selected_pages
        )
        records.append(CompiledRecord(
            record_id=record_id,
            ordinal=index + 1,
            title=title,
            country=_find_country(text),
            page_start=page_start,
            page_end=page_end,
            anchor_page=anchor,
            text=text,
            number_facts=facts,
        ))

    assigned = {
        page
        for record in records
        for page in range(record.page_start, record.page_end + 1)
    }
    unassigned = "\n\n".join(
        f"[Page {page.page_number}]\n{page.text}"
        for page in pages
        if page.page_number not in assigned
    )
    unassigned_pages = [
        page for page in pages if page.page_number not in assigned
    ]
    supplementary = _segment_pages(
        document_id,
        unassigned_pages,
        prefix="supplement",
    )
    return CompiledDocument(
        document_id=document_id,
        record_kind="repeated_entity",
        records=records,
        supplementary_records=supplementary,
        unassigned_text=unassigned,
        page_count=len(pages),
        warnings=warnings,
    )
