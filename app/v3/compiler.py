"""Question-independent compilation of repeated Markdown entities and facts."""

from __future__ import annotations

import html
import re
from collections import Counter
from collections.abc import Iterable

from app.reduction.normalizer import parse_number
from app.schemas import DocumentPage
from app.v3.models import CompiledDocument, CompiledRecord, NumberFact
from app.v3.table_compiler import compile_contents, compile_flat_tables, compile_tables

COMPILER_VERSION = "v3.4"

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
    "table of contents",
    "introduction",
    "overview",
    "executive summary",
    "preface",
    "foreword",
    "appendix",
    "key facts",
    "facts and figures",
    "at a glance",
}
_INLINE_PROFILE_RE = re.compile(
    r"\b(?P<label>[A-Z][A-Z _-]{2,30})\s+(?P<ordinal>\d{1,4})\s*:\s*"
    r"(?P<title>[^.\n]{3,140}?)(?=\.\s+(?:COUNTRY|LOCATION|REGION)\s*:|[.\n])"
)
_INLINE_PROFILE_EXCLUSIONS = {
    "box",
    "chapter",
    "figure",
    "page",
    "section",
    "spotlight",
    "table",
}
_INLINE_FACT_RE = re.compile(
    r"\b(?P<kind>[A-Z][A-Z _-]{1,40})\s+IN\s+NUMBERS\s*:\s*"
    r"(?P<body>.*?)(?=\.(?:\s+[A-Z]|\s*\[Page\s+\d+\]|$))",
    re.I | re.S,
)
_INLINE_FACT_ITEM_RE = re.compile(
    r"^(?P<label>[A-Za-z][^\d;]{0,120}?)\s+"
    r"(?P<raw>[<>]?\s*-?\d[\d,.]*(?:\.\d+)?)\s*(?P<unit>.*)$"
)
# Image-caption arrows never introduce a chapter title.
_CAPTION_PREFIXES = ("↓", "↑", "→", "←")
# A banner such as "12 SPAIN" labels a chapter but is not its title.
_ORDINAL_BANNER_RE = re.compile(r"^\d{1,3}\s+[A-Z][A-Z\s'\-]*$")
# A zero-padded entry such as "03 Omey Island" belongs to a numbered sub-list.
_LIST_ENTRY_RE = re.compile(r"^0\d\s+\S")
# A heading repeated at least this often is per-record furniture, not a title.
_BOILERPLATE_MIN_REPEATS = 3
# How far ahead of a chapter's opening page its fact card may be printed.
_ORPHAN_CARD_REACH = 3

_COUNTRY_ALIASES = {
    "Albania": ("albania", "albanian"),
    "Argentina": ("argentina", "argentinian", "argentine"),
    "Australia": ("australia", "australian"),
    "Austria": ("austria", "austrian"),
    "Brazil": ("brazil", "brazilian"),
    "Bulgaria": ("bulgaria", "bulgarian"),
    "Canada": ("canada", "canadian"),
    "Chile": ("chile", "chilean"),
    "China": ("china", "chinese"),
    "Croatia": ("croatia", "croatian"),
    "Denmark": ("denmark", "danish"),
    "Egypt": ("egypt", "egyptian"),
    "England": ("england", "english"),
    "Estonia": ("estonia", "estonian"),
    "Finland": ("finland", "finnish"),
    "France": ("france", "french"),
    "Germany": ("germany", "german"),
    "Greece": ("greece", "greek"),
    "Hungary": ("hungary", "hungarian"),
    "Iceland": ("iceland", "icelandic"),
    "India": ("india", "indian"),
    "Ireland": ("ireland", "irish"),
    "Italy": ("italy", "italian"),
    "Japan": ("japan", "japanese"),
    "Kenya": ("kenya", "kenyan"),
    "Korea": ("korea", "korean"),
    "Latvia": ("latvia", "latvian"),
    "Lithuania": ("lithuania", "lithuanian"),
    "Mexico": ("mexico", "mexican"),
    "Montenegro": ("montenegro", "montenegrin"),
    "New Zealand": ("new zealand", "new zealanders", "kiwi"),
    "Nigeria": ("nigeria", "nigerian"),
    "Norway": ("norway", "norwegian"),
    "Poland": ("poland", "polish"),
    "Portugal": ("portugal", "portuguese"),
    "Romania": ("romania", "romanian"),
    "Scotland": ("scotland", "scottish", "scots"),
    "Slovakia": ("slovakia", "slovakian", "slovak"),
    "Slovenia": ("slovenia", "slovenian"),
    "South Africa": ("south africa", "south african"),
    "Spain": ("spain", "spanish"),
    "Sweden": ("sweden", "swedish"),
    "Switzerland": ("switzerland", "swiss"),
    "Ukraine": ("ukraine", "ukrainian"),
    "United States": ("united states", "usa", "us", "american"),
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
    # A title that is not marked up as a heading is recognised by how it is
    # set rather than by what it is called: names are capitalised and prose is
    # not, which holds whatever the document is about.
    return bool(_heading(line)) or _is_name_cased(text)


def _is_name_cased(text: str) -> bool:
    """Report whether a line is capitalised the way a name is, not a sentence."""
    words = [word for word in re.findall(r"[^\W\d_]+", text, re.UNICODE) if len(word) > 2]
    if not words:
        return False
    capitalised = sum(1 for word in words if word[:1].isupper())
    # One lowercase particle ("of", "del", "i") is normal inside a name.
    return capitalised >= max(1, len(words) - 1)


def _find_country(record_text: str) -> str:
    # Only the label is case-insensitive. Folding the value too would let the
    # capitalisation requirement through, so a sentence reading "... country:
    # over millennia ..." would be read as a group name.
    # A label names its value on the line it appears on. Letting the value run
    # over a line break makes it swallow the first word of whatever follows,
    # so "Country: Uzbekistan" above a sentence becomes "Uzbekistan A".
    explicit = re.search(
        r"(?:^|\b)(?:\*\*|__)?(?i:country)[^\S\r\n]*:[^\S\r\n]*"
        r"(?P<country>[A-Z][A-Za-z'\-]*(?:[^\S\r\n]+[A-Z][A-Za-z'\-]*){0,4})",
        record_text,
    )
    if explicit:
        value = re.sub(r"[*_`#]", "", explicit.group("country")).strip(" .")
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


def field_id(label: str) -> str:
    """Reduce a metric's wording to the identifier facts are grouped under.

    Every stage that binds a number to a field must route through here. Two
    stages deriving their own identifiers would split one measurement across
    two fields, and a field split in half never reaches the full coverage
    that deterministic reduction requires.
    """
    folded = html.unescape(label).casefold()
    # A label names the measurement before any colon; what follows is the
    # specific thing measured, and a parenthetical carries the unit. Neither
    # belongs to the field's identity, or "Highest point: Mount A" and
    # "Highest point: Mount B" would be two fields instead of one.
    text = re.sub(r"\([^)]*\)", " ", folded.split(":", 1)[0])
    normalized = re.sub(r"[^a-z0-9]+", "_", text).strip("_")
    return normalized[:80] or "number"


# Words that say how many, how precisely, or how often — never what.
_QUANTITY_WORDS = frozenset({
    "about", "annual", "annually", "approximate", "approximately", "approx",
    "around", "average", "billion", "each", "estimated", "every", "figure",
    "hundred", "million", "monthly", "number", "numbers", "peak", "population",
    "recorded", "thousand", "total", "value", "week", "weekly", "year",
    "yearly", "years",
})
_FIELD_STOPWORDS = frozenset({
    "a", "an", "and", "at", "been", "by", "for", "have", "in", "is",
    "its", "of", "on", "or", "per", "s", "that", "the", "to", "was", "with",
})


def _field_terms(field: str) -> frozenset[str]:
    return frozenset(
        term for term in field.split("_") if term and term not in _FIELD_STOPWORDS
    )


def _measured_terms(field: str) -> frozenset[str]:
    """Return what a field measures, without how much or how often.

    A document writes one metric two ways: "approximate number of visitors
    annually" beside "million visitors per year". Everything but the thing
    measured differs, and the words that differ all say how many or how often
    rather than what. Removing those leaves the metric itself.
    """
    return frozenset(_field_terms(field) - _QUANTITY_WORDS)


def _drop_label_titles(records: list[CompiledRecord]) -> None:
    """Unname a record whose title is one of the document's own metric labels.

    A chapter opening on its fact card can leave the first line of that card
    standing where the title should be, and an answer then reports "Area
    covered (sq km)" as the name of a place. The document says which lines are
    labels — it uses them as labels elsewhere — so a title that is one of them
    names nothing, and leaving it numbered lets profiling find the real name.
    """
    labels = {
        re.sub(r"\s+", " ", fact.label).strip().casefold()
        for record in records
        for fact in record.number_facts
        if fact.label
    }
    for record in records:
        folded = re.sub(r"\s+", " ", record.title).strip().casefold()
        if folded and folded in labels:
            record.title = f"Record {record.ordinal}"


def merge_synonym_fields(records: list[CompiledRecord]) -> dict[str, str]:
    """Fold fields that are one measurement written two ways.

    A document does not always spell a metric the same way twice — a fact card
    may read "Year established" in one chapter and "Established" in the next —
    and a metric split across two fields never reaches the coverage that
    deterministic reduction requires. Two signals identify a split without any
    knowledge of the subject: one field's terms are contained in the other's,
    and no record reports both, because a record states each measurement once.

    Returns the applied mapping from folded field to surviving field.
    """
    holders: dict[str, set[str]] = {}
    for record in records:
        for fact in record.number_facts:
            holders.setdefault(fact.field, set()).add(record.record_id)
    terms = {field: _field_terms(field) for field in holders}
    measured = {field: _measured_terms(field) for field in holders}
    # Terms contained by many fields describe a shape common to the document
    # ("number of ...", "length of ...") rather than one specific measurement.
    containers = {
        field: sum(
            1
            for other in holders
            if other != field and terms[field] and terms[field] <= terms[other]
        )
        for field in holders
    }
    order = sorted(holders, key=lambda field: (-len(holders[field]), field))
    mapping: dict[str, str] = {}
    for index, survivor in enumerate(order):
        if survivor in mapping:
            continue
        for folded in order[index + 1:]:
            if folded in mapping or not terms[folded]:
                continue
            # One field's terms sit inside the other's, or the two name the
            # same measurement once the quantity words are set aside.
            contained = terms[folded] < terms[survivor]
            same_metric = bool(
                measured[folded]
                and measured[folded] == measured[survivor]
                and terms[folded] != terms[survivor]
            )
            if not contained and not same_metric:
                continue
            if contained and containers[folded] > 2:
                continue
            if not holders[folded].isdisjoint(holders[survivor]):
                continue
            mapping[folded] = survivor
            holders[survivor] |= holders[folded]
    if mapping:
        for record in records:
            for fact in record.number_facts:
                fact.field = mapping.get(fact.field, fact.field)
    return mapping


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
    if "gigawatt" in folded:
        return "GWh"
    if "ppp" in folded and "$" in folded:
        return "2021 PPP $"
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
            field=field_id(label),
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


def _page_at_offset(text: str, offset: int, default: int) -> int:
    page = default
    for marker in re.finditer(r"\[Page (\d+)\]", text[:offset]):
        page = int(marker.group(1))
    return page


def parse_inline_number_facts(
    record_id: str,
    default_page: int,
    record_text: str,
) -> list[NumberFact]:
    """Parse semicolon-delimited ``X IN NUMBERS`` statements inside prose."""
    facts: list[NumberFact] = []
    scan_text = re.sub(
        r"([A-Za-z])\s*\[Page \d+\]\s*([A-Za-z])",
        r"\1\2",
        record_text,
    )
    scan_text = re.sub(r"\s*\[Page \d+\]\s*", " ", scan_text)
    for marker in _INLINE_FACT_RE.finditer(scan_text):
        page = default_page
        for raw_item in marker.group("body").split(";"):
            item = re.sub(r"\s+", " ", raw_item).strip(" .")
            match = _INLINE_FACT_ITEM_RE.match(item)
            if not match:
                continue
            label = match.group("label").strip(" :-")
            raw = re.sub(r"\s+", "", match.group("raw"))
            unit = match.group("unit").strip(" .")
            normalized_label = f"{label} ({unit})" if unit else label
            value = _normalize_card_value(raw, normalized_label)
            if value is None:
                continue
            facts.append(NumberFact(
                record_id=record_id,
                field=field_id(label),
                label=label,
                value=value,
                raw_value=raw,
                unit=_extract_unit(normalized_label),
                subject=_extract_subject(label),
                page=page,
                quote=f"{label}: {raw}{f' {unit}' if unit else ''}",
            ))
    return facts


def _paged_source(pages: list[DocumentPage]) -> str:
    return "\n\n".join(f"[Page {page.page_number}]\n{page.text}" for page in pages)


def _inline_profile_matches(text: str) -> tuple[str, list[re.Match[str]]]:
    grouped: dict[str, list[re.Match[str]]] = {}
    for match in _INLINE_PROFILE_RE.finditer(text):
        label = re.sub(r"\s+", " ", match.group("label")).strip().casefold()
        if label in _INLINE_PROFILE_EXCLUSIONS:
            continue
        grouped.setdefault(label, []).append(match)
    candidates: list[tuple[int, int, str, list[re.Match[str]]]] = []
    for label, matches in grouped.items():
        ordinals = [int(match.group("ordinal")) for match in matches]
        unique = sorted(set(ordinals))
        if len(unique) < 3:
            continue
        expected = unique[-1] - unique[0] + 1
        continuity = len(unique) / expected if expected else 0
        candidates.append((round(continuity * 1000), len(unique), label, matches))
    if not candidates:
        return "", []
    _, _, label, matches = max(candidates, key=lambda item: (item[0], item[1]))
    return label, matches


def _compile_inline_profiles(
    document_id: str,
    pages: list[DocumentPage],
) -> tuple[list[CompiledRecord], list[CompiledRecord], str, bool, dict[str, int]] | None:
    source = _paged_source(pages)
    label, matches = _inline_profile_matches(source)
    if not matches:
        return None
    records: list[CompiledRecord] = []
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(source)
        text = source[match.start() : end].strip()
        ordinal = int(match.group("ordinal"))
        record_id = f"record-{ordinal:03d}"
        page_start = _page_at_offset(source, match.start(), pages[0].page_number)
        page_end = _page_at_offset(source, max(match.start(), end - 1), page_start)
        title = re.sub(r"\s+", " ", match.group("title")).strip(" .")
        records.append(CompiledRecord(
            record_id=record_id,
            ordinal=ordinal,
            title=title,
            country=_find_country(text),
            page_start=page_start,
            page_end=page_end,
            anchor_page=page_start,
            text=text,
            number_facts=parse_inline_number_facts(record_id, page_start, text),
        ))

    supplementary: list[CompiledRecord] = []
    prefix = source[: matches[0].start()].strip()
    if prefix:
        supplement_end = _page_at_offset(source, matches[0].start(), pages[0].page_number)
        supplementary.append(CompiledRecord(
            record_id="supplement-001",
            ordinal=len(records) + 1,
            title="Front matter",
            page_start=pages[0].page_number,
            page_end=supplement_end,
            anchor_page=pages[0].page_number,
            text=prefix,
        ))

    ordinals = [record.ordinal for record in records]
    expected = set(range(min(ordinals), max(ordinals) + 1))
    missing = expected - set(ordinals)
    titles_unique = len({record.title.casefold() for record in records}) == len(records)
    countries_complete = all(record.country for record in records)
    facts_complete = all(record.number_facts for record in records)
    trusted = not missing and titles_unique and countries_complete and facts_complete
    facts = Counter(fact.field for record in records for fact in record.number_facts)
    first_kind = next(
        (match.group("kind") for match in _INLINE_FACT_RE.finditer(records[0].text)),
        label,
    )
    signals = {
        "profile_markers": len(matches),
        "record_titles": len(records),
        "unique_titles": len({record.title.casefold() for record in records}),
        "missing_ordinals": len(missing),
        "records_with_country": sum(bool(record.country) for record in records),
        "records_with_facts": sum(bool(record.number_facts) for record in records),
        "distinct_fact_fields": len(facts),
    }
    return records, supplementary, first_kind.casefold(), trusted, signals


def _marker_pages(pages: Iterable[DocumentPage]) -> list[int]:
    return [page.page_number for page in pages if _fact_marker(page.text)]


def _page_headings(pages: list[DocumentPage]) -> dict[int, list[str]]:
    """Return every Markdown heading title found on each page."""
    headings: dict[int, list[str]] = {}
    for page in pages:
        titles: list[str] = []
        for line in page.text.splitlines():
            parsed = _heading(line)
            if parsed:
                titles.append(parsed[1].strip())
        headings[page.page_number] = titles
    return headings


def _is_title_like(line: str, boilerplate: set[str]) -> bool:
    """Reject chapter furniture, captions, credits, and list entries."""
    text = _clean_line(line)
    if not text or text.startswith(_CAPTION_PREFIXES):
        return False
    if not 3 <= len(text) <= 120 or not re.search(r"[A-Za-z]{3}", text):
        return False
    if text.casefold() in _TITLE_EXCLUSIONS or _is_fact_section_title(text):
        return False
    # Photo credits, standalone category labels, the "12 SPAIN" ordinal banner,
    # and numbered trail entries all sit beside the title but are never it.
    if "|" in text or _ORDINAL_BANNER_RE.match(text) or _LIST_ENTRY_RE.match(text):
        return False
    if text == text.upper():
        return False
    folded = text.casefold()
    return not any(
        folded == item.casefold() or folded.startswith(item.casefold())
        for item in boilerplate
    )


def _lines_before_furniture(page: DocumentPage, boilerplate: set[str]) -> list[str]:
    """Return the lines a page carries ahead of its recurring chapter furniture.

    A chapter names itself above the sections it repeats. Past the first of
    those sections the page is inside the chapter, and the headings there name
    the items a section lists — a guest house, a walk, a dish — never the
    chapter. Reading one of those as the chapter's name misnames it, and a
    chapter misnamed after something it merely mentions is worse than one left
    nameless: a nameless chapter can still be given the page that names it.
    """
    lines: list[str] = []
    for raw_line in page.text.splitlines():
        parsed = _heading(raw_line)
        if parsed and parsed[1].strip() in boilerplate:
            break
        # The fact card is furniture too, and its first label reads enough
        # like a name to be mistaken for one on a page that carries no title.
        if _is_fact_section_title(parsed[1] if parsed else _clean_line(raw_line)):
            break
        lines.append(raw_line)
    return lines


def _opening_title(opening: list[DocumentPage], boilerplate: set[str]) -> str:
    """Pick the entity title from the pages that open a chapter.

    A chapter can start on a full-bleed photo page whose only text is a
    caption, so the search continues onto the following pages until the
    recurring chapter furniture begins.
    """
    for page in opening:
        lines = _lines_before_furniture(page, boilerplate)
        for raw_line in lines:
            parsed = _heading(raw_line)
            if parsed and _names_an_entity(parsed[1], boilerplate):
                return parsed[1].strip()[:120]
        for raw_line in lines:
            if _names_an_entity(raw_line, boilerplate):
                return _clean_line(raw_line)
    return ""


def _names_an_entity(line: str, boilerplate: set[str]) -> bool:
    """Report whether a line both sits where a title sits and reads as one.

    Position alone is not enough. A page can open with a photo credit, a
    caption run on from the previous spread, or the tail of a sentence, and
    any of those would otherwise be taken as the chapter's name. Requiring the
    line to be set like a name as well is the same test the other segmentation
    path applies, so both paths agree on what a title looks like.
    """
    return _is_title_like(line, boilerplate) and _is_record_title(line)


def _cycle_segments(
    pages: list[DocumentPage],
) -> list[tuple[int, int, str]] | None:
    """Segment repeated chapters by their recurring boilerplate headings.

    Publication chapters repeat the same furniture headings ("Stay here...",
    "Research programme"). The most frequent one occurs exactly once per
    record, so its occurrences calibrate both the record count and the
    chapter boundaries without relying on domain vocabulary.

    Several headings can repeat equally often, and which one calibrates the
    boundaries changes where each chapter is judged to start. Picking from an
    unordered set left that choice to the interpreter, so the same document
    compiled two ways in two runs. Every equally frequent heading is tried
    instead, and the one that names the most chapters wins.
    """
    per_page = _page_headings(pages)
    frequency = Counter(title for titles in per_page.values() for title in titles)
    boilerplate = {
        title
        for title, count in frequency.items()
        if count >= _BOILERPLATE_MIN_REPEATS
    }
    if not boilerplate:
        return None
    occurrences = max(frequency[title] for title in boilerplate)
    if occurrences < 3:
        return None
    candidates = sorted(
        title for title in boilerplate if frequency[title] == occurrences
    )
    scored = [
        (sum(1 for _start, _end, title in segments if title), index, segments)
        for index, segments in enumerate(
            _segments_for_marker(pages, per_page, boilerplate, marker)
            for marker in candidates
        )
    ]
    # Ties fall back to the candidate order, which is sorted, so the result
    # never depends on how a set happened to be iterated.
    return max(scored, key=lambda item: (item[0], -item[1]))[2]


def _segments_for_marker(
    pages: list[DocumentPage],
    per_page: dict[int, list[str]],
    boilerplate: set[str],
    marker: str,
) -> list[tuple[int, int, str]]:
    """Cut the document into chapters using one recurring heading."""
    marker_pages = [
        number for number, titles in sorted(per_page.items()) if marker in titles
    ]

    numbers = sorted(per_page)
    position = {number: index for index, number in enumerate(numbers)}
    starts: list[int] = []
    for order, marker_page in enumerate(marker_pages):
        floor = marker_pages[order - 1] if order else numbers[0] - 1
        start = marker_page
        walker = position[marker_page]
        while walker - 1 >= 0:
            candidate = numbers[walker - 1]
            if candidate <= floor:
                break
            titles = per_page[candidate]
            if titles and all(title in boilerplate for title in titles):
                walker -= 1
                continue
            start = candidate
            break
        starts.append(start)

    by_number = {page.page_number: page for page in pages}
    segments = [
        (
            start,
            starts[order + 1] - 1 if order + 1 < len(starts) else numbers[-1],
            "",
        )
        for order, start in enumerate(starts)
    ]
    segments = [
        (start, end, _segment_title(by_number, per_page, boilerplate, start, end))
        for start, end, _title in segments
    ]
    # Both corrections belong to segmentation, and their order matters: a card
    # is claimed first, then the page that names the chapter it opens.
    segments = _claim_orphan_fact_cards(
        pages, by_number, per_page, boilerplate, segments
    )
    segments = _claim_opening_titles(pages, segments, boilerplate)
    return _claim_named_openings(by_number, boilerplate, segments)


def _segment_title(
    by_number: dict[int, DocumentPage],
    per_page: dict[int, list[str]],
    boilerplate: set[str],
    start: int,
    end: int,
) -> str:
    """Read a chapter's name from the pages that open its range."""
    opening: list[DocumentPage] = []
    for number in range(start, end + 1):
        page = by_number.get(number)
        if page is None:
            continue
        if opening and any(title in boilerplate for title in per_page.get(number, [])):
            break
        opening.append(page)
    return _opening_title(opening, boilerplate)


def _claim_orphan_fact_cards(
    pages: list[DocumentPage],
    by_number: dict[int, DocumentPage],
    per_page: dict[int, list[str]],
    boilerplate: set[str],
    segments: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Give a fact card that no chapter reads to the chapter it introduces.

    A publication may print a chapter's card on the spread facing its opening
    page. The card then falls at the end of the previous chapter's range,
    where nothing reads it because that chapter already has one, and the
    chapter it describes is left with none. Moving the boundary back to the
    card puts each card with the chapter whose figures it states, without
    changing how many chapters there are.
    """
    marked = {page.page_number for page in pages if _fact_marker(page.text)}
    adjusted = list(segments)
    for index in range(1, len(adjusted)):
        start, end, title = adjusted[index]
        if any(number in marked for number in range(start, end + 1)):
            continue
        previous_start, previous_end, previous_title = adjusted[index - 1]
        cards = [
            number
            for number in range(previous_start, previous_end + 1)
            if number in marked
        ]
        # The first card in a range is the one that chapter reads. Only a
        # later one is spare, and only if it sits within reach of the opening
        # page it faces.
        candidate = next(
            (
                number
                for number in reversed(cards[1:])
                if 0 < start - number <= _ORPHAN_CARD_REACH
            ),
            None,
        )
        if candidate is None:
            continue
        adjusted[index - 1] = (previous_start, candidate - 1, previous_title)
        # The name was read from the page this chapter used to start on. Now
        # that it starts on the card's page, that page names it — usually the
        # chapter's own opening page, since a card faces the text it belongs
        # to. Keeping the old name would label the chapter with whatever item
        # happened to head the page after it.
        renamed = _segment_title(by_number, per_page, boilerplate, candidate, end)
        adjusted[index] = (candidate, end, renamed or title)
    return adjusted


def _claim_opening_titles(
    pages: list[DocumentPage],
    segments: list[tuple[int, int, str]],
    boilerplate: set[str],
) -> list[tuple[int, int, str]]:
    """Start a nameless chapter on the page that names it.

    A chapter opens on a spread whose first page carries nothing but its
    title, and where segmentation put the boundary after that page the chapter
    is left with no name of its own while the page sits unread at the end of
    the chapter before. Taking it back names the chapter, and the chapter it
    came from does not want it: the page names something else.
    """
    by_number = {page.page_number: page for page in pages}
    adjusted = list(segments)
    for index in range(1, len(adjusted)):
        start, end, title = adjusted[index]
        if title:
            continue
        previous_start, previous_end, previous_title = adjusted[index - 1]
        # Only a page that the chapter before can spare, and only the page
        # immediately ahead of this one.
        if previous_end != start - 1 or previous_end <= previous_start:
            continue
        page = by_number.get(previous_end)
        if page is None:
            continue
        found = _opening_title([page], boilerplate)
        if not found or found == previous_title:
            continue
        adjusted[index - 1] = (previous_start, previous_end - 1, previous_title)
        adjusted[index] = (previous_end, end, found)
    return adjusted


def _claim_named_openings(
    by_number: dict[int, DocumentPage],
    boilerplate: set[str],
    segments: list[tuple[int, int, str]],
) -> list[tuple[int, int, str]]:
    """Start a nameless chapter on the page inside it that carries a name.

    The heading that calibrates the boundaries can repeat before a chapter's
    opening page rather than after it, and the cut then lands inside the
    chapter before: the pages ahead of the name are that chapter's tail, and
    the nameless range that follows begins too early. Moving the boundary
    forward to the page that names a chapter hands the name to the chapter it
    opens and the pages before it back to the one they continue.
    """
    adjusted = list(segments)
    for index in range(1, len(adjusted)):
        start, end, title = adjusted[index]
        if title:
            continue
        previous_start, previous_end, previous_title = adjusted[index - 1]
        for number in range(start + 1, end + 1):
            page = by_number.get(number)
            if page is None:
                continue
            found = _opening_title([page], boilerplate)
            if not found or found == previous_title:
                continue
            # The chapter before takes the pages this one gives up, so it must
            # still be left with the page it starts on.
            if number - 1 < previous_start:
                break
            adjusted[index - 1] = (previous_start, number - 1, previous_title)
            adjusted[index] = (number, end, found)
            break
    return adjusted


def _cycle_records(
    pages: list[DocumentPage],
    segments: list[tuple[int, int, str]],
) -> list[CompiledRecord]:
    """Build compiled records from boilerplate-cycle chapter boundaries."""
    by_number = {page.page_number: page for page in pages}
    records: list[CompiledRecord] = []
    for index, (start, end, title) in enumerate(segments):
        selected = [
            by_number[number]
            for number in range(start, end + 1)
            if number in by_number
        ]
        if not selected:
            continue
        text = "\n\n".join(
            f"[Page {page.page_number}]\n{page.text}" for page in selected
        )
        anchor = next(
            (page.page_number for page in selected if _fact_marker(page.text)),
            start,
        )
        record_id = f"record-{index + 1:03d}"
        facts = parse_number_facts(record_id, anchor, by_number[anchor].text)
        if not facts:
            facts = parse_inline_number_facts(record_id, start, text)
        records.append(CompiledRecord(
            record_id=record_id,
            ordinal=index + 1,
            title=title or f"Record {index + 1}",
            country=_find_country(text),
            page_start=start,
            page_end=end,
            anchor_page=anchor,
            text=text,
            number_facts=facts,
        ))
    return records


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


def _field_catalog(
    records: list[CompiledRecord],
    tables,
) -> list[str]:
    fields = {fact.field for record in records for fact in record.number_facts}
    for table in tables:
        fields.update(table.columns)
        for row in table.rows:
            fields.update(row.values)
    return sorted(field for field in fields if field)


def _finalize_fields(records: list[CompiledRecord], tables) -> list[str]:
    """Fold wording variants together, then report what fields remain."""
    merge_synonym_fields(records)
    _drop_label_titles(records)
    return _field_catalog(records, tables)


def compile_document(
    document_id: str,
    pages: list[DocumentPage],
) -> CompiledDocument:
    """Compile a trusted repeated-entity registry or exhaustive fallback segments."""
    tables = compile_tables(pages)
    # A report numbers tables within a chapter and again within its annexes,
    # so "Table 5.1" and "Annex Table 5.A.7" share a leading number while
    # being different tables. Only the pages a table was read from tell them
    # apart, and dropping one because the other exists loses its figures.
    known = {
        page
        for table in tables
        if table.rows
        for page in range(table.page_start, table.page_end + 1)
    }
    tables += [
        table
        for table in compile_flat_tables(pages)
        if table.page_start not in known
    ]
    contents_text, contents_pages, contents_entries, contents_trusted = compile_contents(pages)
    inline = _compile_inline_profiles(document_id, pages)
    if inline is not None:
        records, supplementary, entity_label, trusted, signals = inline
        warnings = [] if trusted else [
            "Inline repeated-entity registry failed a completeness check"
        ]
        warnings.extend(
            warning
            for table in tables
            for warning in table.warnings
            if table.rows
        )
        return CompiledDocument(
            document_id=document_id,
            compiler_version=COMPILER_VERSION,
            record_kind="repeated_entity",
            entity_label=entity_label,
            records=records,
            supplementary_records=supplementary,
            tables=tables,
            contents_entries=contents_entries,
            contents_text=contents_text,
            contents_pages=contents_pages,
            contents_trusted=contents_trusted,
            field_catalog=_finalize_fields(records, tables),
            unassigned_text=supplementary[0].text if supplementary else "",
            page_count=len(pages),
            registry_trusted=trusted,
            registry_signals=signals,
            warnings=warnings,
        )

    anchors = _marker_pages(pages)
    if len(anchors) < 3:
        records = _segment_pages(document_id, pages)
        return CompiledDocument(
            document_id=document_id,
            compiler_version=COMPILER_VERSION,
            record_kind="segments",
            records=records,
            tables=tables,
            contents_entries=contents_entries,
            contents_text=contents_text,
            contents_pages=contents_pages,
            contents_trusted=contents_trusted,
            field_catalog=_finalize_fields(records, tables),
            page_count=len(pages),
            registry_signals={
                "fact_sections": len(anchors),
                "tables": len(tables),
                "trusted_tables": sum(table.trusted for table in tables),
            },
            warnings=[
                "No repeated fact-section structure detected",
                *[
                    warning
                    for table in tables
                    for warning in table.warnings
                    if table.rows
                ],
            ],
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

    toc_count = _toc_count(pages, titles[0][0])

    def integrity(candidate: list[CompiledRecord], expected: int) -> tuple[bool, int]:
        distinct = len({record.title.casefold() for record in candidate})
        return (
            len(candidate) == expected
            and distinct == len(candidate)
            and (not toc_count or toc_count == len(candidate))
            and not any(record.title.startswith("Record ") for record in candidate)
        ), distinct

    trusted, unique_titles = integrity(records, len(anchors))
    segmentation = "fact_sections"
    cycle_markers = 0
    if not trusted:
        # Fact-section anchors merge chapters whose fact box was lost in
        # conversion. Recurring chapter furniture calibrates the true record
        # count without relying on any domain vocabulary.
        segments = _cycle_segments(pages)
        if segments:

            candidate = _cycle_records(pages, segments)
            candidate_trusted, candidate_titles = integrity(candidate, len(segments))
            if candidate and candidate_titles > unique_titles:
                records = candidate
                trusted, unique_titles = candidate_trusted, candidate_titles
                segmentation = "boilerplate_cycle"
                # The marker repeats once per chapter, so its occurrences are
                # independent evidence for how many records the document has.
                cycle_markers = len(segments)
    if not trusted:
        warnings.append("Repeated-entity registry failed an integrity check")
    if toc_count and toc_count != len(records):
        warnings.append(
            f"Contents lists {toc_count} entities but compiler found {len(records)}"
        )

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
        tables=tables,
        contents_entries=contents_entries,
        contents_text=contents_text,
        contents_pages=contents_pages,
        contents_trusted=contents_trusted,
        field_catalog=_finalize_fields(records, tables),
        unassigned_text=unassigned_text,
        page_count=len(pages),
        registry_trusted=trusted,
        registry_signals={
            "fact_sections": len(anchors),
            "record_titles": len(records),
            "unique_titles": unique_titles,
            "contents_entries": toc_count,
            "assigned_pages": len(assigned_pages),
            "segmentation": segmentation,
            "cycle_markers": cycle_markers,
        },
        warnings=warnings,
    )
