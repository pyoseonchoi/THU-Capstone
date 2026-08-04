"""Question-independent compilation of repeated Markdown entities and facts."""

from __future__ import annotations

import html
import re
from collections.abc import Iterable

from app.reduction.normalizer import parse_number
from app.schemas import DocumentPage
from app.v3.models import CompiledDocument, CompiledRecord, NumberFact

COMPILER_VERSION = "v3.1"

_HEADING_PREFIX_RE = re.compile(r"^#{1,6}\s*")
_HEADING_RE = re.compile(r"^(?P<marks>#{1,6})\s+(?P<title>.+?)\s*$")
_NUMBER_PREFIX_RE = re.compile(
    r"^(?P<raw>[<>]?\s*-?\d[\d,.]*(?:\.\d+)?)\s*(?P<label>.*)$"
)
_LABEL_VALUE_RE = re.compile(
    r"^(?P<label>[^:]{2,100}):\s*"
    r"(?P<raw>[<>]?\s*-?\d[\d,.]*(?:\.\d+)?)\s*(?P<unit>.*)$"
)
_BACK_MATTER_RE = re.compile(
    r"^#{1,6}\s*(?:index|references|bibliography|acknowledgements?|credits|glossary)\b",
    re.IGNORECASE,
)
_TITLE_EXCLUSIONS = {
    "contents",
    "introduction",
    "research programme",
    "field notes",
    "interpreting the figures",
    "key facts",
    "facts and figures",
    "at a glance",
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


def _heading(line: str) -> tuple[int, str] | None:
    match = _HEADING_RE.match(line.strip())
    if not match:
        return None
    return len(match.group("marks")), html.unescape(match.group("title")).strip()


def _is_fact_section_title(title: str) -> bool:
    folded = re.sub(r"\s+", " ", html.unescape(title).strip().casefold())
    return (
        folded.endswith(" in numbers")
        or folded in {"key facts", "facts and figures", "at a glance"}
    )


def _fact_marker(page_text: str) -> tuple[int, str, int] | None:
    """Return line index, title, and heading level for a fact section."""
    for index, raw_line in enumerate(page_text.splitlines()):
        parsed = _heading(raw_line)
        if parsed and _is_fact_section_title(parsed[1]):
            return index, parsed[1], parsed[0]
        cleaned = _clean_line(raw_line)
        if raw_line.strip() == cleaned and _is_fact_section_title(cleaned):
            return index, cleaned, 7
    return None


def _is_record_title(line: str) -> bool:
    text = _clean_line(line)
    folded = text.casefold()
    if not 3 <= len(text) <= 120 or len(text.split()) > 16:
        return False
    if text.endswith((".", ",", ":", ";")):
        return False
    if folded in _TITLE_EXCLUSIONS or _is_fact_section_title(text):
        return False
    return bool(_heading(line)) or any(
        term in folded
        for term in (
            "park",
            "station",
            "observatory",
            "laboratory",
            "centre",
            "center",
            "institute",
        )
    )


def _find_country(record_text: str) -> str:
    explicit = re.search(
        r"(?im)^\s*(?:\*\*|__)?country\s*:\s*(.+?)(?:\*\*|__)?\s*$",
        record_text,
    )
    if explicit:
        value = re.sub(r"[*_`#]", "", explicit.group(1)).strip(" .")
        if value:
            return value

    for raw_line in record_text[:1600].splitlines():
        line = re.sub(r"\d+", "", _clean_line(raw_line))
        compact = re.sub(r"[^a-z]", "", line.casefold())
        for country in _COUNTRY_ALIASES:
            if compact == re.sub(r"[^a-z]", "", country.casefold()):
                return country

    folded = record_text.casefold()
    introduction = folded[:900]
    scores: dict[str, float] = {}
    positions: dict[str, int] = {}
    for country, aliases in _COUNTRY_ALIASES.items():
        country_positions: list[int] = []
        score = 0.0
        for alias_index, alias in enumerate(aliases):
            matches = list(re.finditer(rf"\b{re.escape(alias)}\b", folded))
            country_positions.extend(match.start() for match in matches)
            score += len(matches) * (2.0 if alias_index == 0 else 1.5)
            if re.search(rf"\b{re.escape(alias)}(?:'s|s')\b", introduction):
                score += 20.0
        if score:
            scores[country] = score
            positions[country] = min(country_positions)
    if not scores:
        return ""
    return max(scores, key=lambda name: (scores[name], -positions[name]))


def _slug_label(label: str) -> str:
    folded = html.unescape(label).casefold()
    if any(term in folded for term in ("area covered", "area monitored", "monitored area")):
        return "area"
    if any(
        term in folded
        for term in (
            "highest point",
            "highest peak",
            "highest operating point",
            "operating elevation",
        )
    ):
        return "highest_point"
    if (
        ("visitor" in folded or "visiting researcher" in folded)
        and ("annual" in folded or "per year" in folded)
    ):
        return "annual_visitors"
    if "year established" in folded or folded.startswith("established"):
        return "establishment_year"
    if "estimated age" in folded and ("tree" in folded or "cedar" in folded):
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
        "sq mile": "sq miles",
        "years": "years",
        "km/h": "km/h",
    }
    if unit in aliases:
        return aliases[unit]
    if re.search(r"\bsq\s*km\b", folded):
        return "sq km"
    if re.search(r"\bsq\s*miles?\b", folded):
        return "sq miles"
    if re.search(r"\bmetres?\b|\bmeters?\b|\(m\)", folded):
        return "m"
    if "million" in folded:
        return "visitors/year"
    if re.search(r"\b(?:bc|bce)\b", folded):
        return "year_bc"
    return unit


def _extract_subject(label: str) -> str:
    text = html.unescape(label)
    if ":" in text:
        subject = re.sub(r"\([^)]*\)", "", text.split(":", 1)[1]).strip()
        if not re.fullmatch(r"[\d,.]+(?:\s*\w+)?", subject):
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
    if re.search(r"\b(?:bc|bce)\b", folded):
        value = -abs(value)
    return value


def _card_lines(page_text: str) -> list[str]:
    marker = _fact_marker(page_text)
    if not marker:
        return []
    marker_index, _, _ = marker
    lines: list[str] = []
    for raw in page_text.splitlines()[marker_index + 1 :]:
        stripped = raw.strip()
        if _heading(stripped) and lines:
            break
        cleaned = _clean_line(stripped)
        if cleaned:
            lines.append(cleaned)
        if len(lines) >= 40:
            break
    return lines


def parse_number_facts(
    record_id: str,
    page_number: int,
    page_text: str,
) -> list[NumberFact]:
    """Bind card numbers and labels in either reading-order direction."""
    lines = _card_lines(page_text)
    facts: list[NumberFact] = []
    pending_raw = ""
    pending_label = ""

    def append_fact(raw: str, label: str, unit_hint: str = "") -> None:
        normalized_label = f"{label} ({unit_hint})" if unit_hint else label
        value = _normalize_card_value(raw, normalized_label)
        if value is None:
            return
        facts.append(NumberFact(
            record_id=record_id,
            field=_slug_label(label),
            label=label,
            value=value,
            raw_value=raw,
            unit=_extract_unit(normalized_label),
            subject=_extract_subject(label),
            page=page_number,
            quote=f"{label}: {raw}{f' {unit_hint}' if unit_hint else ''}",
        ))

    for line in lines:
        label_value = _LABEL_VALUE_RE.match(line)
        if label_value:
            append_fact(
                re.sub(r"\s+", "", label_value.group("raw")),
                label_value.group("label").strip(),
                label_value.group("unit").strip(),
            )
            pending_raw = ""
            pending_label = ""
            continue

        number = _NUMBER_PREFIX_RE.match(line)
        if number:
            raw = re.sub(r"\s+", "", number.group("raw"))
            inline_label = number.group("label").strip()
            if inline_label:
                append_fact(raw, inline_label)
                pending_raw = ""
                pending_label = ""
            elif pending_label:
                append_fact(raw, pending_label)
                pending_raw = ""
                pending_label = ""
            else:
                pending_raw = raw
            continue

        if pending_raw:
            append_fact(pending_raw, line)
            pending_raw = ""
            pending_label = ""
        else:
            pending_label = line
    return facts


def _marker_pages(pages: Iterable[DocumentPage]) -> list[int]:
    return [page.page_number for page in pages if _fact_marker(page.text)]


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
    """Build exhaustive consecutive segments when no repeated structure exists."""
    del document_id
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
        title = next(
            (
                parsed[1]
                for line in first.text.splitlines()
                if (parsed := _heading(line))
            ),
            f"Pages {first.page_number}-{last.page_number}",
        )
        ordinal = len(records) + 1
        records.append(CompiledRecord(
            record_id=f"{prefix}-{ordinal:03d}",
            ordinal=ordinal,
            title=title[:120],
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
        page_gap = current and page.page_number != current[-1].page_number + 1
        if current and (page_gap or current_size + size > target_characters):
            flush()
        current.append(page)
        current_size += size
    flush()
    return records


def all_mapping_records(document: CompiledDocument) -> list[CompiledRecord]:
    """Return every terminal text unit, including front and back matter."""
    return [*document.records, *document.supplementary_records]


def _title_for_anchor(
    pages: list[DocumentPage],
    anchor: int,
    previous_anchor: int | None,
) -> tuple[int, str] | None:
    page_by_number = {page.page_number: page for page in pages}
    marker = _fact_marker(page_by_number[anchor].text)
    marker_level = marker[2] if marker else 7
    start = max(pages[0].page_number, (previous_anchor or anchor - 8) + 1)
    candidates: list[tuple[int, int, str]] = []
    fallbacks: list[tuple[int, int, str]] = []
    for page_number in range(start, anchor + 1):
        page = page_by_number.get(page_number)
        if page is None:
            continue
        for line_index, line in enumerate(page.text.splitlines()):
            parsed = _heading(line)
            if parsed and parsed[0] < marker_level and _is_record_title(line):
                candidates.append((page_number, line_index, parsed[1]))
            elif _is_record_title(line):
                fallbacks.append((page_number, line_index, _clean_line(line)))
    selected = candidates[-1] if candidates else (fallbacks[-1] if fallbacks else None)
    if selected is None:
        return None
    return selected[0], selected[2]


def _toc_count(pages: list[DocumentPage], before_page: int) -> int:
    text = "\n".join(page.text for page in pages if page.page_number < before_page)
    entries = re.findall(r"(?m)^\s*\d{1,3}[.)]\s+\S.+$", text)
    return len(entries) if len(entries) >= 3 else 0


def _entity_label(marker_title: str) -> str:
    folded = marker_title.casefold().strip()
    if folded.endswith(" in numbers"):
        return marker_title[: -len(" in numbers")].strip().casefold() or "entity"
    return "entity"


def compile_document(
    document_id: str,
    pages: list[DocumentPage],
) -> CompiledDocument:
    """Compile a trusted repeated-entity registry or exhaustive fallback segments."""
    anchors = _marker_pages(pages)
    if len(anchors) < 3:
        records = _segment_pages(document_id, pages)
        return CompiledDocument(
            document_id=document_id,
            compiler_version=COMPILER_VERSION,
            record_kind="segments",
            records=records,
            page_count=len(pages),
            registry_signals={"fact_sections": len(anchors)},
            warnings=["No repeated fact-section structure detected"],
        )

    page_by_number = {page.page_number: page for page in pages}
    warnings: list[str] = []
    titles: list[tuple[int, str]] = []
    for index, anchor in enumerate(anchors):
        title = _title_for_anchor(
            pages,
            anchor,
            anchors[index - 1] if index else None,
        )
        if title is None:
            title = (anchor, f"Record {index + 1}")
            warnings.append(f"Could not identify title for record {index + 1}")
        titles.append(title)

    back_matter_start = _back_matter_start(pages, anchors[-1])
    records: list[CompiledRecord] = []
    for index, ((title_page, title), anchor) in enumerate(zip(titles, anchors, strict=True)):
        page_end = (
            titles[index + 1][0] - 1
            if index + 1 < len(titles)
            else (
                back_matter_start - 1
                if back_matter_start is not None
                else pages[-1].page_number
            )
        )
        selected = [
            page_by_number[number]
            for number in range(title_page, page_end + 1)
            if number in page_by_number
        ]
        record_id = f"record-{index + 1:03d}"
        text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}" for page in selected
        )
        records.append(CompiledRecord(
            record_id=record_id,
            ordinal=index + 1,
            title=title,
            country=_find_country(text),
            page_start=title_page,
            page_end=page_end,
            anchor_page=anchor,
            text=text,
            number_facts=parse_number_facts(record_id, anchor, page_by_number[anchor].text),
        ))

    assigned_pages = {
        number
        for record in records
        for number in range(record.page_start, record.page_end + 1)
    }
    unassigned_pages = [
        page for page in pages if page.page_number not in assigned_pages
    ]
    supplementary = _segment_pages(
        document_id,
        unassigned_pages,
        prefix="supplement",
    )
    unique_titles = len({record.title.casefold() for record in records})
    toc_count = _toc_count(pages, titles[0][0])
    trusted = (
        len(records) == len(anchors)
        and unique_titles == len(records)
        and (not toc_count or toc_count == len(records))
        and not any(record.title.startswith("Record ") for record in records)
    )
    if not trusted:
        warnings.append("Repeated-entity registry failed an integrity check")
    if toc_count and toc_count != len(records):
        warnings.append(
            f"Contents lists {toc_count} entities but compiler found {len(records)}"
        )

    marker = _fact_marker(page_by_number[anchors[0]].text)
    marker_title = marker[1] if marker else "entity"
    unassigned_text = "\n\n".join(
        f"[Page {page.page_number}]\n{page.text}" for page in unassigned_pages
    )
    return CompiledDocument(
        document_id=document_id,
        compiler_version=COMPILER_VERSION,
        record_kind="repeated_entity",
        entity_label=_entity_label(marker_title),
        records=records,
        supplementary_records=supplementary,
        unassigned_text=unassigned_text,
        page_count=len(pages),
        registry_trusted=trusted,
        registry_signals={
            "fact_sections": len(anchors),
            "record_titles": len(records),
            "unique_titles": unique_titles,
            "contents_entries": toc_count,
            "assigned_pages": len(assigned_pages),
        },
        warnings=warnings,
    )
