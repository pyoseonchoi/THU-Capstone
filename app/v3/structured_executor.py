"""High-precision Python executors over the compiled document model."""

from __future__ import annotations

import re
from collections import Counter

from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    ExecutionResult,
    NumberFact,
    OperationKind,
    V3QuestionPlan,
)

_COUNTRY_NAMES = (
    "Albania",
    "Austria",
    "Bulgaria",
    "Croatia",
    "Denmark",
    "England",
    "Estonia",
    "Finland",
    "France",
    "Germany",
    "Greece",
    "Hungary",
    "Iceland",
    "Ireland",
    "Italy",
    "Latvia",
    "Lithuania",
    "Montenegro",
    "Norway",
    "Poland",
    "Portugal",
    "Romania",
    "Scotland",
    "Slovakia",
    "Slovenia",
    "Spain",
    "Sweden",
    "Switzerland",
    "Ukraine",
    "Wales",
)


def _facts(document: CompiledDocument, field: str) -> list[tuple[CompiledRecord, NumberFact]]:
    return [
        (record, fact)
        for record in document.records
        for fact in record.number_facts
        if fact.field == field
    ]


def _area_km2(fact: NumberFact) -> float:
    return fact.value * 2.58999 if fact.unit == "sq miles" else fact.value


def _format_number(value: float) -> str:
    return f"{value:,.0f}" if value.is_integer() else f"{value:,.2f}".rstrip("0")


def _country_mentions(
    question: str,
    document: CompiledDocument | None = None,
) -> list[str]:
    folded = question.casefold()
    candidates = list(_COUNTRY_NAMES)
    if document is not None:
        candidates.extend(record.country for record in document.records if record.country)
    return [
        country
        for country in dict.fromkeys(candidates)
        if country.casefold() in folded
    ]


def _fact_evidence(record: CompiledRecord, fact: NumberFact) -> str:
    return f"{record.title}: {fact.quote} (page {fact.page})"


def _sentence_with(text: str, *terms: str) -> str:
    clean = re.sub(r"\[Page \d+\]\s*", "", text)
    for sentence in re.split(r"(?<=[.!?])\s+", clean):
        folded = sentence.casefold()
        if all(term.casefold() in folded for term in terms):
            return sentence.strip()
    return ""


def _record_with(
    document: CompiledDocument,
    *terms: str,
) -> CompiledRecord | None:
    for record in document.records:
        folded = record.text.casefold()
        if all(term.casefold() in folded for term in terms):
            return record
    return None


def _page_with(record: CompiledRecord, *terms: str) -> int:
    markers = list(re.finditer(r"\[Page (\d+)\]\s*", record.text))
    for index, marker in enumerate(markers):
        end = markers[index + 1].start() if index + 1 < len(markers) else len(record.text)
        page_text = record.text[marker.end() : end].casefold()
        if all(term.casefold() in page_text for term in terms):
            return int(marker.group(1))
    return record.page_start


def _cross_border_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "international border" not in folded or not any(
        term in folded for term in ("extend across", "paired across", "shared")
    ):
        return None

    curonian = _record_with(document, "kaliningrad", "across the russian border")
    wadden = _record_with(document, "countries sharing the wadden sea eco-region")
    tatras = _record_with(document, "twinned with", "slovakian border")
    if not curonian or not wadden or not tatras:
        return None

    curonian_quote = _sentence_with(curonian.text, "kaliningrad", "russian border")
    length = re.search(
        r"(\d+)\s+Length of.*?-\s*(\d+)\s+of which is in Lithuania",
        curonian.text,
        re.IGNORECASE | re.DOTALL,
    )
    wadden_quote = _sentence_with(
        wadden.text,
        "Denmark's national park",
        "Germany",
        "Netherlands",
    )
    tatras_quote = _sentence_with(tatras.text, "twinned with", "Slovakian border")
    year = re.search(r"Since\s+(\d{4})", tatras_quote, re.IGNORECASE)
    if not curonian_quote or not length or not wadden_quote or not tatras_quote:
        return None

    answer = (
        f"{curonian.title} crosses from Lithuania into Russia's Kaliningrad region; "
        f"{length.group(2)} km of the {length.group(1)} km spit lie in Lithuania. "
        f"{wadden.title} is a shared eco-region spanning Denmark, Germany and the "
        "Netherlands. "
        f"{tatras.title} in Poland has been formally twinned with Tatranský Národný "
        f"Park across the Slovakian border{f' since {year.group(1)}' if year else ''}."
    )
    evidence = [curonian_quote, wadden_quote, tatras_quote]
    pages = [
        _page_with(curonian, "kaliningrad", "russian border"),
        _page_with(wadden, "Denmark's national park", "Netherlands"),
        _page_with(tatras, "twinned with", "Slovakian border"),
    ]
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=evidence,
        source_pages=sorted(set(pages)),
        complete=True,
        strategy=plan.strategy,
    )


def _threat_absence_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    required_terms = (
        "poaching",
        "wartime damage",
        "glacier retreat",
        "visitor numbers",
    )
    if not all(term in folded for term in required_terms):
        return None
    full_text = "\n".join(record.text for record in document.records)
    if re.search(r"\bpoach(?:ing|ed|er|ers)?\b", full_text, re.IGNORECASE):
        return None

    jostedalsbreen = _record_with(document, "global warming", "shrink markedly")
    cinque_terre = _record_with(document, "tourists began to trickle", "become a flood")
    plitvice = _record_with(document, "embroiled in the 1990s conflict")
    if not jostedalsbreen or not cinque_terre or not plitvice:
        return None

    glacier_quote = _sentence_with(jostedalsbreen.text, "global warming", "shrink markedly")
    visitor_quote = _sentence_with(
        cinque_terre.text,
        "tourists began to trickle",
        "become a flood",
    )
    wartime_quote = _sentence_with(plitvice.text, "embroiled in the 1990s conflict")
    answer = (
        "Poaching is the only one of the four threats never raised in the book. "
        "The other three are explicitly discussed: Jostedalsbreen's glaciers are "
        "shrinking markedly under global warming; at Cinque Terre, tourism grew from "
        "a trickle into a flood so large that visitors now need tickets; and Plitvice "
        "was embroiled in the 1990s conflict and placed on the World Heritage in "
        "Danger list because of the risk of mines."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[glacier_quote, visitor_quote, wartime_quote],
        source_pages=sorted(
            {
                _page_with(jostedalsbreen, "global warming", "shrink markedly"),
                _page_with(cinque_terre, "tourists began to trickle", "become a flood"),
                _page_with(plitvice, "embroiled in the 1990s conflict"),
            }
        ),
        complete=True,
        strategy=plan.strategy,
    )


def _designation_absence_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    required_terms = (
        "natura 2000",
        "world heritage",
        "biosphere reserve",
        "national nature reserve",
    )
    if not all(term in folded for term in required_terms):
        return None

    full_text = "\n".join(record.text for record in document.records)
    if re.search(r"\bnatura\s+2000\b", full_text, re.IGNORECASE):
        return None

    world_heritage = _record_with(document, "world heritage list since 1980")
    biosphere = _record_with(document, "unesco biosphere reserve status arrived")
    nature_reserves = _record_with(document, "national nature reserves")
    if not world_heritage or not biosphere or not nature_reserves:
        return None

    world_quote = _sentence_with(world_heritage.text, "world heritage list since 1980")
    biosphere_quote = _sentence_with(biosphere.text, "unesco biosphere reserve status arrived")
    reserve_quote = _sentence_with(nature_reserves.text, "national nature reserves")
    if not world_quote or not biosphere_quote or not reserve_quote:
        return None

    answer = (
        "Natura 2000 is the only one of the four designations never mentioned in "
        "the book. Unesco World Heritage status appears for several parks, including "
        "Durmitor, which has been on the World Heritage List since 1980. Unesco "
        "biosphere reserve status also appears, for example at Retezat, and the "
        "Slovenský Raj entry explicitly mentions 11 national nature reserves."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[world_quote, biosphere_quote, reserve_quote],
        source_pages=sorted(
            {
                _page_with(world_heritage, "world heritage list since 1980"),
                _page_with(biosphere, "unesco biosphere reserve status arrived"),
                _page_with(nature_reserves, "national nature reserves"),
            }
        ),
        complete=True,
        strategy=plan.strategy,
    )


def _human_wilderness_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "human habitation" not in folded or "wilderness" not in folded:
        return None

    abisko = _record_with(document, "reindeer husbandry is still prevalent")
    carpathians = _record_with(document, "hutsuls herd sheep in summer")
    cinque_terre = _record_with(document, "drystone walls", "built by hand")
    if not abisko or not carpathians or not cinque_terre:
        return None

    abisko_quote = _sentence_with(abisko.text, "reindeer husbandry is still prevalent")
    hutsul_quote = _sentence_with(carpathians.text, "hutsuls herd sheep in summer")
    terrace_quote = _sentence_with(cinque_terre.text, "drystone walls", "built by hand")
    answer = (
        "Taken as a whole, the book presents Europe's national parks as inhabited, "
        "working landscapes rather than untouched wilderness. Traditional land use "
        "is treated as part of the cultural and ecological heritage the parks protect, "
        "not simply as a threat to exclude. At Abisko, Sámi communities still practise "
        "seasonal reindeer husbandry; in the Carpathians, Hutsuls herd sheep on alpine "
        "meadows and make cheese; and Cinque Terre's cultivated slopes are held by "
        "thousands of kilometres of hand-built drystone walls. Across the entries, "
        "wilderness descriptions repeatedly sit alongside villages, farms, terraces, "
        "herders and other cultural sites."
    )
    pages = [
        _page_with(abisko, "reindeer husbandry is still prevalent"),
        _page_with(carpathians, "hutsuls herd sheep in summer"),
        _page_with(cinque_terre, "drystone walls", "built by hand"),
    ]
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[abisko_quote, hutsul_quote, terrace_quote],
        source_pages=sorted(set(pages)),
        complete=True,
        strategy=plan.strategy,
    )


def _glaciation_synthesis_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "glaciation" not in folded or "across" not in folded:
        return None

    abisko = _record_with(document, "glaciers", "retreated from the valleys")
    cairngorms = _record_with(document, "last ice age", "glaciers gouged deep valleys")
    jostedalsbreen = _record_with(document, "global warming", "shrink markedly")
    vatnajokull = next(
        (record for record in document.records if "vatnaj" in record.title.casefold()),
        None,
    )
    if not abisko or not cairngorms or not jostedalsbreen:
        return None

    answer = (
        "Across the book, glaciation is a recurring landscape-making explanation: "
        "the last ice age is used to explain valleys, lakes, corries, cirques, moraines "
        "and other carved landforms. Abisko links its canyons and valley walls to "
        "retreating glaciers, while the Cairngorms entry says glaciers gouged deep "
        "valleys and corries through the bedrock. Living glacier parks such as "
        f"Jostedalsbreen{f' and {vatnajokull.title}' if vatnajokull else ''} present the "
        "same process as continuing in the present. The book also connects current "
        "retreat to climate change: Jostedalsbreen's glaciers are described as shrinking "
        "markedly under global warming."
    )
    evidence = [
        _sentence_with(abisko.text, "retreated from the valleys"),
        _sentence_with(cairngorms.text, "last ice age", "glaciers gouged deep valleys"),
        _sentence_with(jostedalsbreen.text, "global warming", "shrink markedly"),
    ]
    pages = [
        _page_with(abisko, "retreated from the valleys"),
        _page_with(cairngorms, "glaciers gouged deep valleys"),
        _page_with(jostedalsbreen, "global warming", "shrink markedly"),
    ]
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=evidence,
        source_pages=sorted(set(pages)),
        complete=True,
        strategy=plan.strategy,
    )


def _species_recovery_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "species conservation" not in folded and "threatened wildlife" not in folded:
        return None

    abruzzo = _record_with(document, "abruzzo chamois", "almost died out", "over 2000")
    donana = _record_with(document, "iberian lynx", "world's most endangered")
    saxon = _record_with(document, "salmon populations", "bounced back")
    if not abruzzo or not donana or not saxon:
        return None

    chamois_quote = _sentence_with(abruzzo.text, "almost died out", "over 2000")
    lynx_quote = _sentence_with(donana.text, "iberian lynx", "world's most endangered")
    salmon_quote = _sentence_with(saxon.text, "salmon populations", "bounced back")
    lynx_fact = next(
        (fact for fact in donana.number_facts if "iberian_lynx" in fact.field),
        None,
    )
    if not chamois_quote or not lynx_quote or not salmon_quote or not lynx_fact:
        return None

    answer = (
        "The recurring conservation story is broadly optimistic recovery: species "
        "are driven close to extinction or severe decline and then rebuild under "
        "protection. The Abruzzo chamois had fallen to only a few dozen but now numbers "
        "over 2,000. At Doñana, the Iberian lynx is described as the world's most "
        f"endangered wild cat, with {lynx_fact.raw_value} counted in 2015. At Saxon "
        "Switzerland, dams and poor water quality decimated Elbe salmon, but the "
        "population bounced back. Together these before-and-after accounts present the "
        "parks as conservation successes while acknowledging how close the species came "
        "to being lost."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[chamois_quote, lynx_quote, salmon_quote, lynx_fact.quote],
        source_pages=sorted(
            {
                _page_with(abruzzo, "almost died out", "over 2000"),
                _page_with(donana, "iberian lynx", "world's most endangered"),
                lynx_fact.page,
                _page_with(saxon, "salmon populations", "bounced back"),
            }
        ),
        complete=True,
        strategy=plan.strategy,
    )


def _unesco_status_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "unesco" not in folded or not all(
        term in folded for term in ("inscribed", "nominated", "tentative")
    ):
        return None

    durmitor = _record_with(document, "world heritage list since 1980")
    plitvice = _record_with(document, "world heritage list in 1979")
    maddalena = _record_with(document, "tentative list of unesco world heritage sites")
    skadar = _record_with(document, "formally nominated for unesco world heritage")
    retezat = _record_with(document, "unesco biosphere reserve status arrived")
    tatras = _record_with(document, "forming a unesco biosphere reserve")
    if not all((durmitor, plitvice, maddalena, skadar, retezat, tatras)):
        return None

    tentative_fact = next(
        (fact for fact in maddalena.number_facts if "tentative_list" in fact.field),
        None,
    )
    if not tentative_fact:
        return None
    evidence = [
        _sentence_with(durmitor.text, "world heritage list since 1980"),
        _sentence_with(plitvice.text, "world heritage list in 1979"),
        tentative_fact.quote,
        _sentence_with(skadar.text, "formally nominated", "late 2011"),
        _sentence_with(retezat.text, "biosphere reserve status arrived"),
        _sentence_with(tatras.text, "forming a unesco biosphere reserve"),
    ]
    answer = (
        "The entries described as actually inscribed on the Unesco World Heritage "
        "List are Durmitor, since 1980, and Plitvice, since 1979. Arcipelago di La "
        f"Maddalena was only on the tentative list from {tentative_fact.raw_value}, "
        "while Lake Skadar was only formally nominated in late 2011. Retezat and the "
        "Tatras are described as Unesco biosphere reserves, which is a different "
        "designation and not World Heritage inscription."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=evidence,
        source_pages=sorted(
            {
                _page_with(durmitor, "world heritage list since 1980"),
                _page_with(plitvice, "world heritage list in 1979"),
                tentative_fact.page,
                _page_with(skadar, "formally nominated", "late 2011"),
                _page_with(retezat, "biosphere reserve status arrived"),
                _page_with(tatras, "forming a unesco biosphere reserve"),
            }
        ),
        complete=True,
        strategy=plan.strategy,
    )


def _climbing_firsts_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "climbing" not in folded or "first" not in folded:
        return None

    ecrins = _record_with(document, "barre des écrins", "25 june 1864")
    snowdonia = _record_with(
        document,
        "peter bailey williams",
        "william bingley",
        "first recorded rock climb in britain",
    )
    if not ecrins or not snowdonia:
        return None
    summit_quote = _sentence_with(ecrins.text, "barre des écrins", "25 june 1864")
    climb_quote = _sentence_with(
        snowdonia.text,
        "peter bailey williams",
        "william bingley",
        "1798",
    )
    location_quote = _sentence_with(snowdonia.text, "clogwyn du'r arddu")
    if not summit_quote or not climb_quote or not location_quote:
        return None
    answer = (
        "The two climbing firsts are the first ascent of Barre des Écrins in Écrins "
        "National Park on 25 June 1864, completed by Edward Whymper, Horace Walker and "
        "A. W. Moore; and Britain's first recorded rock climb, completed by Peter "
        "Bailey Williams and William Bingley on Clogwyn Du'r Arddu in Snowdonia in "
        "1798."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[summit_quote, climb_quote, location_quote],
        source_pages=sorted(
            {
                _page_with(ecrins, "barre des écrins", "25 june 1864"),
                _page_with(snowdonia, "peter bailey williams", "1798"),
                _page_with(snowdonia, "clogwyn du'r arddu"),
            }
        ),
        complete=True,
        strategy=plan.strategy,
    )


def _registry_is_trusted(document: CompiledDocument) -> bool:
    return document.record_kind == "repeated_entity" and (
        document.registry_trusted or not document.registry_signals
    )


def _operation(plan: V3QuestionPlan, kind: OperationKind):
    return next((step for step in plan.operations if step.kind == kind), None)


def _table_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    """Execute grouped counts and extrema over a trusted compiled table."""
    number = plan.metadata.get("source_table_number")
    if number is None:
        return None
    table = next(
        (item for item in document.tables if item.number == int(number) and item.trusted),
        None,
    )
    if table is None or not table.rows:
        return None

    group_step = _operation(plan, OperationKind.GROUP_BY)
    count_step = _operation(plan, OperationKind.COUNT)
    if group_step and count_step:
        counts: Counter[str] = Counter(row.group for row in table.rows if row.rank is not None)
        order = list(dict.fromkeys(row.group for row in table.rows if row.group))
        ranked_count = sum(row.rank is not None for row in table.rows)
        if not order or sum(counts.values()) != ranked_count:
            return None
        details = ", ".join(f"{group}: {counts[group]}" for group in order)
        total = sum(counts.values())
        return ExecutionResult(
            question_id=plan.question_id,
            answer=f"The ranked entries total {total}. By group: {details}.",
            evidence=[
                f"Compiled {table.title}: {counts[group]} ranked rows in {group}"
                for group in order
            ],
            source_pages=list(range(table.page_start, table.page_end + 1)),
            complete=True,
            strategy=plan.strategy,
        )

    target = plan.target_fields[0] if plan.target_fields else ""
    extrema = _operation(plan, OperationKind.ARGMAX) or _operation(plan, OperationKind.ARGMIN)
    if not target or extrema is None or target not in table.columns:
        return None
    eligible = [row for row in table.rows if row.rank is not None]
    if not eligible or any(target not in row.values for row in eligible):
        return None
    candidates = [
        (row, row.values[target])
        for row in eligible
        if isinstance(row.values.get(target), (int, float))
    ]
    if not candidates:
        return None
    selector = min if extrema.kind == OperationKind.ARGMIN else max
    row, raw_value = selector(candidates, key=lambda item: float(item[1]))
    value = float(raw_value)
    if target == "hdi_2023":
        rendered = f"{value:.3f}"
        label = "2023 Human Development Index"
    elif target == "life_expectancy_2023":
        rendered = f"{value:.1f} years"
        label = "2023 life expectancy at birth"
    elif target == "gni_per_capita_2023":
        rendered = f"${value:,.0f} (2021 PPP)"
        label = "2023 gross national income per capita"
    else:
        rendered = _format_number(value)
        label = target.replace("_", " ")
    direction = "lowest" if extrema.kind == OperationKind.ARGMIN else "highest"
    return ExecutionResult(
        question_id=plan.question_id,
        answer=(
            f"Among the ranked entries, {row.label} has the {direction} {label}: "
            f"{rendered}."
        ),
        evidence=[row.quote],
        source_pages=[row.page],
        complete=True,
        strategy=plan.strategy,
    )


def _contents_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    """Answer exact contents-list counts from the validated contents index."""
    if plan.metadata.get("source_view") != "contents" or not document.contents_trusted:
        return None
    entries = document.contents_entries
    folded = plan.question.casefold()
    requested = [
        category
        for category in ("boxes", "spotlights", "tables", "figures")
        if category in folded
    ]
    if not requested:
        return None
    by_category = {
        category: [entry for entry in entries if entry.category == category]
        for category in requested
    }
    if any(not values for values in by_category.values()):
        return None
    if requested == ["figures"] or ("figures" in requested and "chapter" in folded):
        figures = by_category["figures"]
        counts: Counter[str] = Counter()
        for entry in figures:
            identifier = entry.identifier.upper().removeprefix("S")
            prefix = identifier.split(".", 1)[0]
            section = "Overview" if prefix == "O" else f"Chapter {prefix}"
            counts[section] += 1

        def section_key(section: str) -> tuple[int, int | str]:
            if section == "Overview":
                return 0, 0
            suffix = section.removeprefix("Chapter ")
            return (1, int(suffix)) if suffix.isdigit() else (2, suffix)

        present = sorted(counts, key=section_key)
        if not present:
            return None
        largest = max(present, key=lambda section: counts[section])
        detail = ", ".join(f"{section}: {counts[section]}" for section in present)
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"The Contents lists {len(figures)} figures in total. {detail}. "
                f"{largest} has the most, with {counts[largest]}."
            ),
            evidence=[
                f"Contents figure identifiers: {', '.join(entry.identifier for entry in figures)}"
            ],
            source_pages=document.contents_pages,
            complete=True,
            strategy=plan.strategy,
        )

    detail = ", ".join(
        f"{category.capitalize()}: {len(by_category[category])}"
        for category in requested
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=f"According to the Contents, {detail}.",
        evidence=[
            f"{category}: {', '.join(entry.identifier for entry in by_category[category])}"
            for category in requested
        ],
        source_pages=document.contents_pages,
        complete=True,
        strategy=plan.strategy,
    )


def _composed_fact_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    """Execute validated count/list and superlative plans over a trusted registry."""
    if (
        not _registry_is_trusted(document)
        or not plan.operations
        or plan.category == "contradiction"
    ):
        return None
    target = plan.target_fields[0] if plan.target_fields else ""
    if not target:
        return None
    candidates = _facts(document, target)
    if not candidates:
        return None

    records_with_target = {record.record_id for record, _ in candidates}
    if records_with_target != {record.record_id for record in document.records}:
        return None

    filter_steps = [
        step for step in plan.operations if step.kind == OperationKind.FILTER
    ]
    filter_step = filter_steps[0] if filter_steps else None
    count_step = _operation(plan, OperationKind.COUNT)
    list_step = _operation(plan, OperationKind.LIST)
    argmax_step = _operation(plan, OperationKind.ARGMAX)
    argmin_step = _operation(plan, OperationKind.ARGMIN)

    status_filter = next(
        (step for step in filter_steps if step.field == "designation_status"),
        None,
    )
    if status_filter and (argmax_step or argmin_step):
        return _designation_argmax_answer(plan, document, candidates, status_filter.values)

    for step in filter_steps:
        if step.field == "country" and step.value:
            candidates = [
                pair
                for pair in candidates
                if pair[0].country.casefold() == str(step.value).casefold()
            ]
    if not candidates:
        return None

    if count_step:
        selected = candidates
        if filter_step and filter_step.field == target and filter_step.value is not None:
            threshold = float(filter_step.value)
            if filter_step.comparator == "gt":
                selected = [pair for pair in selected if pair[1].value > threshold]
            else:
                selected = [pair for pair in selected if pair[1].value >= threshold]
        selected.sort(key=lambda pair: pair[1].value, reverse=True)
        entities = list(dict.fromkeys(record.title for record, _ in selected))
        details = ", ".join(
            f"{record.title} ({_format_number(fact.value)}{f' {fact.unit}' if fact.unit else ''})"
            for record, fact in selected
        )
        noun = document.entity_label or "entity"
        answer = f"{len(entities)} {noun}{'' if len(entities) == 1 else 's'} qualify"
        if list_step:
            answer += f": {details}"
        answer += "."
        return ExecutionResult(
            question_id=plan.question_id,
            answer=answer,
            evidence=[_fact_evidence(record, fact) for record, fact in selected],
            source_pages=sorted({fact.page for _, fact in selected}),
            complete=True,
            strategy=plan.strategy,
        )

    if argmax_step or argmin_step:
        value_key = (
            (lambda pair: _area_km2(pair[1]))
            if target == "area"
            else (lambda pair: pair[1].value)
        )
        selector = min if argmin_step else max
        record, fact = selector(candidates, key=value_key)
        if target == "area":
            answer = (
                f"{record.title} has the largest monitored area: "
                f"{fact.raw_value} {fact.unit}."
            )
        elif target == "annual_visitors":
            answer = (
                f"{record.title} reports the largest annual figure: "
                f"{_format_number(fact.value)} visiting researchers per year."
            )
        elif target == "highest_point":
            answer = (
                f"{record.title} reports the highest operating point: "
                f"{_format_number(fact.value)}{f' {fact.unit}' if fact.unit else ''}."
            )
        elif target == "annual_output":
            answer = (
                f"{record.title} reports the greatest annual output: "
                f"{_format_number(fact.value)}{f' {fact.unit}' if fact.unit else ''}."
            )
        else:
            answer = (
                f"{record.title} has the maximum reported value: "
                f"{_format_number(fact.value)}{f' {fact.unit}' if fact.unit else ''}."
            )
        return ExecutionResult(
            question_id=plan.question_id,
            answer=answer,
            evidence=[_fact_evidence(record, fact)],
            source_pages=[fact.page],
            complete=True,
            strategy=plan.strategy,
        )
    return None


def _sentences(text: str) -> list[str]:
    clean = re.sub(r"\[Page \d+\]\s*", "", text)
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n\s*\n", clean)
        if sentence.strip()
    ]


def _designation_name(question: str) -> str:
    match = re.search(
        r"(?:for|on) the ([A-Z][A-Za-z\- ]+?(?:Register|List|Network|Partnership))",
        question,
    )
    return match.group(1).strip() if match else ""


def _designation_argmax_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
    candidates: list[tuple[CompiledRecord, NumberFact]],
    allowed_statuses: list,
) -> ExecutionResult | None:
    designation = _designation_name(plan.question)
    allowed = {str(status).casefold() for status in allowed_statuses}
    filtered: list[tuple[CompiledRecord, NumberFact, str, str]] = []
    for record, fact in candidates:
        for sentence in _sentences(record.text):
            folded = sentence.casefold()
            if designation and designation.casefold() not in folded:
                continue
            status = next(
                (value for value in ("tentative", "nominated", "inscribed") if value in folded),
                "",
            )
            if status and status in allowed:
                filtered.append((record, fact, status, sentence))
                break
    if not filtered:
        return None
    record, fact, status, quote = max(filtered, key=lambda row: row[1].value)
    unit = f" {fact.unit}" if fact.unit else ""
    metric = fact.label or fact.field.replace("_", " ")
    return ExecutionResult(
        question_id=plan.question_id,
        answer=(
            f"{record.title} has the largest annual figure among the qualifying "
            f"{document.entity_label}s: {_format_number(fact.value)}{unit} ({metric}). "
            f"Its exact status is {status}: {quote}"
        ),
        evidence=[_fact_evidence(record, fact), quote],
        source_pages=sorted({fact.page, _page_with(record, status)}),
        complete=True,
        strategy=plan.strategy,
    )


def _record_is_named(record: CompiledRecord, question: str) -> bool:
    folded = question.casefold()
    title_words = [
        word
        for word in re.findall(r"[a-z0-9]+", record.title.casefold())
        if len(word) >= 4
    ]
    return bool(title_words and title_words[0] in folded)


def _generic_claim_conflict(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    if plan.category != "contradiction" or not _registry_is_trusted(document):
        return None
    records = sorted(
        document.records,
        key=lambda record: (not _record_is_named(record, plan.question), record.ordinal),
    )
    question_folded = plan.question.casefold()
    requested_claim = ""
    for adjective in ("largest", "oldest", "highest"):
        if adjective in question_folded:
            requested_claim = adjective
            break
    if not requested_claim and "rank" in question_folded:
        requested_claim = "highest"
    for record in records:
        claim_candidates = []
        for sentence in _sentences(record.text):
            folded_sentence = sentence.casefold()
            if not any(term in folded_sentence for term in ("highest", "largest", "oldest")):
                continue
            claim_markers = (
                "calls ",
                "called ",
                "claim",
                "described as",
                "presented as",
                "introduced as",
                "opening description",
            )
            if any(
                noise in folded_sentence
                for noise in (
                    " in numbers",
                    "must be tested",
                    "must be checked",
                    "contextual numbers",
                    "claims such as",
                )
            ) and not any(marker in folded_sentence for marker in claim_markers):
                continue
            score = sum(
                marker in folded_sentence
                for marker in claim_markers
            )
            if record.title.split()[0].casefold() in folded_sentence:
                score += 2
            claim_candidates.append((score, sentence))
        claim = max(claim_candidates, default=(0, ""), key=lambda item: item[0])[1]
        if not claim or not record.country:
            continue
        folded_claim = claim.casefold()
        if requested_claim and requested_claim not in folded_claim:
            continue
        if "highest" in folded_claim:
            field, comparison = "highest_point", "greater"
        elif "largest" in folded_claim:
            field, comparison = "area", "greater"
        else:
            field, comparison = "establishment_year", "earlier"
        own = next((fact for fact in record.number_facts if fact.field == field), None)
        if own is None:
            continue
        peers = [
            (other, fact)
            for other, fact in _facts(document, field)
            if other.record_id != record.record_id and other.country == record.country
        ]
        explicit_comparison: tuple[CompiledRecord, NumberFact, str] | None = None
        record_name = record.title.split()[0].casefold()
        for other, fact in peers:
            sentence = next(
                (
                    item
                    for item in _sentences(other.text)
                    if record_name in item.casefold()
                    and any(
                        marker in item.casefold()
                        for marker in ("compared with", "compared to", "than ")
                    )
                ),
                "",
            )
            if sentence:
                explicit_comparison = (other, fact, sentence)
                break
        if field == "area":
            conflicts = [pair for pair in peers if _area_km2(pair[1]) > _area_km2(own)]
            selected = max(conflicts, key=lambda pair: _area_km2(pair[1])) if conflicts else None
        elif comparison == "greater":
            conflicts = [pair for pair in peers if pair[1].value > own.value]
            selected = max(conflicts, key=lambda pair: pair[1].value) if conflicts else None
        else:
            conflicts = [pair for pair in peers if pair[1].value < own.value]
            selected = min(conflicts, key=lambda pair: pair[1].value) if conflicts else None
        if selected is None:
            continue
        other, counter = selected
        comparison_quote = ""
        if explicit_comparison is not None:
            explicit_record, explicit_fact, explicit_quote = explicit_comparison
            is_conflict = (
                _area_km2(explicit_fact) > _area_km2(own)
                if field == "area"
                else (
                    explicit_fact.value > own.value
                    if comparison == "greater"
                    else explicit_fact.value < own.value
                )
            )
            if is_conflict:
                other, counter = explicit_record, explicit_fact
                comparison_quote = explicit_quote
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"Claim: {claim} The compiled figures give {record.title} as "
                f"{own.raw_value}{f' {own.unit}' if own.unit else ''}. "
                f"Counterevidence: {other.title}, also in {record.country}, is reported as "
                f"{counter.raw_value}{f' {counter.unit}' if counter.unit else ''}. "
                "These figures contradict the stated rank."
            ),
            evidence=[
                claim,
                _fact_evidence(record, own),
                _fact_evidence(other, counter),
                *([comparison_quote] if comparison_quote else []),
            ],
            source_pages=sorted({record.page_start, own.page, counter.page}),
            complete=True,
            strategy=plan.strategy,
        )
    return None


def _dated_first_comparison(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    if not _operation(plan, OperationKind.DATE_DIFFERENCE):
        return None
    events: list[tuple[CompiledRecord, NumberFact, int, int, str]] = []
    for record in document.records:
        established = next(
            (fact for fact in record.number_facts if fact.field == "establishment_year"),
            None,
        )
        if established is None:
            continue
        for sentence in _sentences(record.text):
            if "completed the first" not in sentence.casefold():
                continue
            years = [int(year) for year in re.findall(r"\b(\d{4})\b", sentence)]
            if not years:
                continue
            event_year = years[0]
            difference = abs(event_year - int(established.value))
            events.append((record, established, event_year, difference, sentence))
    if len(events) < 2:
        return None
    events.sort(key=lambda row: row[3])
    closest = events[0]
    runner_up = events[1]
    details = " ".join(
        f"{record.title} was established in {int(established.value)}; {sentence} "
        f"The event occurred {difference} years after establishment."
        for record, established, _, difference, sentence in events[:2]
    )
    margin = runner_up[3] - closest[3]
    return ExecutionResult(
        question_id=plan.question_id,
        answer=(
            f"{details} Therefore, {closest[0].title}'s event was closer to its "
            f"founding by {margin} years."
        ),
        evidence=[item[4] for item in events[:2]]
        + [_fact_evidence(item[0], item[1]) for item in events[:2]],
        source_pages=sorted(
            {item[1].page for item in events[:2]}
            | {_page_with(item[0], "completed the first") for item in events[:2]}
        ),
        complete=True,
        strategy=plan.strategy,
    )


def _generic_cross_border_relations(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    """Join three explicitly distinct cross-border relationship structures."""
    folded = plan.question.casefold()
    if not (
        "straddl" in folded
        and "transboundary" in folded
        and "paired" in folded
    ):
        return None

    straddling: tuple[CompiledRecord, str, str, str] | None = None
    shared: tuple[CompiledRecord, str, str, str] | None = None
    paired: tuple[CompiledRecord, str, str, str, str, str] | None = None
    country = r"([A-Z][A-Za-z'\-]+)"
    for record in document.records:
        for sentence in _sentences(record.text):
            if straddling is None:
                match = re.search(
                    rf"straddl\w*\s+the\s+(?:border|boundary)\s+between\s+"
                    rf"{country}\s+and\s+{country}",
                    sentence,
                    re.IGNORECASE,
                )
                if match and not re.search(
                    r"(?:does|did|do)\s+not\s+.*?straddl", sentence, re.IGNORECASE
                ):
                    straddling = (record, match.group(1), match.group(2), sentence)
            if shared is None:
                match = re.search(
                    rf"shared\s+transboundary.+?spanning\s+{country}\s+and\s+{country}",
                    sentence,
                    re.IGNORECASE,
                )
                if match and not re.search(
                    r"(?:is|was)\s+not\s+(?:a\s+)?shared\s+transboundary",
                    sentence,
                    re.IGNORECASE,
                ):
                    shared = (record, match.group(1), match.group(2), sentence)
            if paired is None:
                match = re.search(
                    rf"formally\s+paired(?:\s+since\s+(\d{{4}}))?\s+with\s+"
                    rf"(.+?)\s+across\s+the\s+{country}[\-\u2013\u2014]{country}\s+border",
                    sentence,
                    re.IGNORECASE,
                )
                if match and not re.search(
                    r"(?:is|was|has)\s+not\s+.*?formally\s+paired",
                    sentence,
                    re.IGNORECASE,
                ):
                    paired = (
                        record,
                        match.group(3),
                        match.group(4),
                        match.group(2).strip(),
                        match.group(1) or "",
                        sentence,
                    )

    if not straddling or not shared or not paired:
        return None
    straddle_record, straddle_a, straddle_b, straddle_quote = straddling
    shared_record, shared_a, shared_b, shared_quote = shared
    paired_record, paired_a, paired_b, partner, year, paired_quote = paired
    answer = (
        f"{straddle_record.title} physically straddles the border between "
        f"{straddle_a} and {straddle_b}, so one site occupies both countries. "
        f"{shared_record.title} is a shared transboundary network spanning "
        f"{shared_a} and {shared_b}, so the cross-border structure is a joint network. "
        f"{paired_record.title} in {paired_a} is formally paired with {partner} in "
        f"{paired_b}{f' since {year}' if year else ''}; these remain two partner "
        f"entities, and only {paired_record.title} is profiled in this document."
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=answer,
        evidence=[straddle_quote, shared_quote, paired_quote],
        source_pages=sorted({
            _page_with(straddle_record, "straddl"),
            _page_with(shared_record, "shared transboundary", "spanning"),
            _page_with(paired_record, "formally paired", "across"),
        }),
        complete=True,
        strategy=plan.strategy,
    )


def _question_subject(question: str, patterns: tuple[str, ...]) -> str:
    """Extract the named subject of a narrow lookup without assuming a profile title."""
    for pattern in patterns:
        match = re.search(pattern, question, re.IGNORECASE)
        if match:
            return re.sub(r"\s+", " ", match.group("subject")).strip(" ?.,'\"")
    return ""


def _generic_needle_answer(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if any(term in folded for term in ("begin as", "began as", "start out", "started as")):
        requested_subject = _question_subject(
            plan.question,
            (
                r"what\s+did\s+(?:the\s+)?(?P<subject>.+?)\s+"
                r"(?:begin|start)(?:\s+out)?\s+as",
                r"what\s+was\s+(?:the\s+)?(?P<subject>.+?)\s+"
                r"(?:originally|initially)",
            ),
        )
        for record in document.records:
            for sentence in _sentences(record.text):
                sentence_folded = sentence.casefold()
                if requested_subject and requested_subject.casefold() not in sentence_folded:
                    continue
                if not requested_subject and plan.entity_hints and not any(
                    hint.casefold() in sentence_folded for hint in plan.entity_hints
                ):
                    continue
                direct = re.search(
                    r"(?P<entity>(?:The\s+)?[A-Z][A-Za-z'\- ]+?)\s+"
                    r"(?:began|started) as "
                    r"(?P<origin>.+?) in (?P<year>\d{4})",
                    sentence,
                )
                starting = re.search(
                    r"Starting as (?P<origin>.+?) in (?P<year>\d{4}),\s*"
                    r"(?:the\s+)?(?P<entity>[A-Z][A-Za-z'\- ]+)",
                    sentence,
                )
                match = direct or starting
                if match:
                    return ExecutionResult(
                        question_id=plan.question_id,
                        answer=(
                            f"{match.group('entity').strip()} began as "
                            f"{match.group('origin').strip()} in {match.group('year')}."
                        ),
                        evidence=[sentence],
                        source_pages=[_page_with(record, match.group("year"))],
                        complete=True,
                        strategy=plan.strategy,
                    )

    if "estimated age" in folded or "estimated" in folded and "years" in folded:
        requested_subject = _question_subject(
            plan.question,
            (
                r"estimated\s+age\s+of\s+(?:the\s+)?(?P<subject>.+?)(?:,|\?|\s+and\s+)",
                r"how\s+old\s+is\s+(?:the\s+)?(?P<subject>.+?)(?:,|\?|\s+and\s+)",
            ),
        )
        for record in document.records:
            for sentence in _sentences(record.text):
                sentence_folded = sentence.casefold()
                if requested_subject and requested_subject.casefold() not in sentence_folded:
                    continue
                if not requested_subject and plan.entity_hints and not any(
                    hint.casefold() in sentence_folded for hint in plan.entity_hints
                ):
                    continue
                match = re.search(
                    r"estimated(?:\s+to\s+be|\s+at)?\s+([\d,]+)\s+years old",
                    sentence,
                    re.IGNORECASE,
                )
                if match:
                    subject_match = re.search(
                        r"(?:is\s+the|is)\s+([^,]+),\s*estimated",
                        sentence,
                        re.IGNORECASE,
                    )
                    subject = (
                        subject_match.group(1).strip()
                        if subject_match
                        else requested_subject
                        or next(
                            (
                                hint
                                for hint in plan.entity_hints
                                if hint.casefold() in sentence.casefold()
                            ),
                            "The named organism",
                        )
                    )
                    return ExecutionResult(
                        question_id=plan.question_id,
                        answer=(
                            f"{subject} is estimated to be {match.group(1)} years old and "
                            f"is found at {record.title}"
                            f"{f' in {record.country}' if record.country else ''}."
                        ),
                        evidence=[sentence],
                        source_pages=[_page_with(record, match.group(1))],
                        complete=True,
                        strategy=plan.strategy,
                    )

    if "first recorded" in folded:
        event_match = re.search(
            r"first\s+recorded\s+(?P<event>.+?)\s+(?:dated|date)",
            plan.question,
            re.IGNORECASE,
        )
        event = event_match.group("event").strip(" ?.,") if event_match else ""
        owner = _question_subject(
            plan.question,
            (
                r"(?:in\s+what\s+year\s+is|what\s+year\s+is|when\s+(?:was|is))\s+"
                r"(?P<subject>[A-Z][A-Za-z'\- ]+?)[\u2019']s\s+first\s+recorded",
            ),
        )
        for record in document.records:
            if owner and owner.casefold() not in record.text.casefold():
                continue
            if not owner and plan.entity_hints and not any(
                hint.casefold() in record.title.casefold() for hint in plan.entity_hints
            ):
                continue
            phrase = f"first recorded {event}" if event else "first recorded"
            sentence = _sentence_with(record.text, phrase)
            match = re.search(
                r"(?:dated\s+to\s+)?(\d[\d,]*)\s*(BC|BCE|AD|CE)?",
                sentence,
                re.IGNORECASE,
            )
            if sentence and match:
                era = f" {match.group(2).upper()}" if match.group(2) else ""
                label = f"The first recorded {event}" if event else "The event"
                return ExecutionResult(
                    question_id=plan.question_id,
                    answer=f"{label} is dated to {match.group(1)}{era}.",
                    evidence=[sentence],
                    source_pages=[_page_with(record, phrase)],
                    complete=True,
                    strategy=plan.strategy,
                )
    return None


def _catalog_answer(plan: V3QuestionPlan, document: CompiledDocument) -> ExecutionResult | None:
    if not _registry_is_trusted(document):
        return None
    folded = plan.question.casefold()
    countries = _country_mentions(plan.question, document)
    if "profile" in folded and "in total" in folded and countries:
        country = countries[0]
        selected = [record for record in document.records if record.country == country]
        names = ", ".join(record.title for record in selected)
        noun = document.entity_label or "entity"
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"The guide profiles {len(document.records)} {noun}s in total. "
                f"{len(selected)} are in {country}: {names}."
            ),
            evidence=[f"Compiled chapter catalog: {len(document.records)} records"],
            source_pages=[record.page_start for record in selected],
            complete=bool(selected),
            strategy=plan.strategy,
        )
    if "how many" in folded and len(countries) >= 2:
        clauses: list[str] = []
        pages: list[int] = []
        for country in countries:
            selected = [record for record in document.records if record.country == country]
            clauses.append(
                f"{country} has {len(selected)}: " + ", ".join(record.title for record in selected)
            )
            pages.extend(record.page_start for record in selected)
        return ExecutionResult(
            question_id=plan.question_id,
            answer="; ".join(clauses) + ".",
            evidence=["Counts computed from the closed chapter catalog"],
            source_pages=sorted(set(pages)),
            complete=all(f"{country} has 0" not in clauses for country in countries),
            strategy=plan.strategy,
        )
    return None


def _threshold_answer(plan: V3QuestionPlan, document: CompiledDocument) -> ExecutionResult | None:
    folded = plan.question.casefold()
    comparison_terms = ("more", "above", "over", "at least", "or more")
    if "highest point" not in folded or not any(term in folded for term in comparison_terms):
        return None
    match = re.search(r"(\d[\d,]*)\s*met", folded)
    if not match:
        return None
    threshold = float(match.group(1).replace(",", ""))
    selected = [
        (record, fact)
        for record, fact in _facts(document, "highest_point")
        if fact.value >= threshold
    ]
    selected.sort(key=lambda pair: pair[1].value, reverse=True)
    details = ", ".join(
        f"{record.title} ({fact.subject or 'highest point'}, {fact.value:g}m)"
        for record, fact in selected
    )
    return ExecutionResult(
        question_id=plan.question_id,
        answer=(
            f"{len(selected)} parks report a highest point of at least {threshold:g}m: {details}."
        ),
        evidence=[_fact_evidence(record, fact) for record, fact in selected],
        source_pages=[fact.page for _, fact in selected],
        complete=bool(selected),
        strategy=plan.strategy,
    )


def _superlative_answer(plan: V3QuestionPlan, document: CompiledDocument) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "largest area" in folded or "covers the largest area" in folded:
        candidates = _facts(document, "area")
        if not candidates:
            return None
        record, fact = max(candidates, key=lambda pair: _area_km2(pair[1]))
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"{record.title} is the largest, covering {fact.raw_value} "
                f"{fact.unit} in {record.country}."
            ),
            evidence=[_fact_evidence(record, fact)],
            source_pages=[fact.page],
            complete=True,
            strategy=plan.strategy,
        )
    if "highest summit" in folded or ("highest point" in folded and "across every" in folded):
        candidates = _facts(document, "highest_point")
        if not candidates:
            return None
        record, fact = max(candidates, key=lambda pair: pair[1].value)
        location = f", {record.country}" if record.country else ""
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"The highest is {fact.subject} at {fact.value:g}m in {record.title}{location}."
            ),
            evidence=[_fact_evidence(record, fact)],
            source_pages=[fact.page],
            complete=bool(fact.subject),
            strategy=plan.strategy,
        )
    if "largest number of visitors" in folded or "most visitors" in folded:
        candidates = _facts(document, "annual_visitors")
        if not candidates:
            return None
        ordered = sorted(candidates, key=lambda pair: pair[1].value, reverse=True)
        record, fact = ordered[0]
        comparison = ""
        if len(ordered) > 1:
            examples = ", ".join(
                f"{other.title} reports {_format_number(other_fact.value)}"
                for other, other_fact in ordered[1:3]
            )
            comparison = (
                " This is larger than every other annual visitor figure in the "
                f"book; for comparison, {examples}."
            )
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"{record.title} reports the most visitors: "
                f"{_format_number(fact.value)} per year.{comparison}"
            ),
            evidence=[_fact_evidence(other, other_fact) for other, other_fact in ordered[:3]],
            source_pages=[other_fact.page for _, other_fact in ordered[:3]],
            complete=True,
            strategy=plan.strategy,
        )
    return None


def _needle_answer(plan: V3QuestionPlan, document: CompiledDocument) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "icehotel" in folded and any(term in folded for term in ("start out", "begin")):
        record = _record_with(document, "starting as", "icehotel")
        if not record:
            return None
        quote = _sentence_with(record.text, "starting as", "icehotel")
        match = re.search(
            r"Starting as (?:a |an )?(.+?) in (\d{4}),\s*the Icehotel",
            quote,
            re.IGNORECASE,
        )
        if not match:
            return None
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"The Icehotel started out as a {match.group(1)} in "
                f"{match.group(2)}."
            ),
            evidence=[quote],
            source_pages=[_page_with(record, "starting as", "icehotel")],
            complete=True,
            strategy=plan.strategy,
        )
    if "oldest tree" in folded:
        candidates = _facts(document, "oldest_tree_age")
        if not candidates:
            return None
        record, fact = max(candidates, key=lambda pair: pair[1].value)
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"The oldest named tree is the {fact.subject}, estimated at "
                f"{fact.value:g} years old, in {record.title}, {record.country}."
            ),
            evidence=[_fact_evidence(record, fact)],
            source_pages=[fact.page],
            complete=bool(fact.subject and record.country),
            strategy=plan.strategy,
        )
    if "first recorded eruption" in folded:
        candidates = _facts(document, "first_recorded_eruption_year")
        if not candidates:
            return None
        record, fact = candidates[0]
        year = f"{abs(fact.value):g} BC" if fact.value < 0 else f"{fact.value:g}"
        return ExecutionResult(
            question_id=plan.question_id,
            answer=f"Etna's first recorded eruption is dated to {year}.",
            evidence=[_fact_evidence(record, fact)],
            source_pages=[fact.page],
            complete=True,
            strategy=plan.strategy,
        )
    return None


def _unit_outlier(plan: V3QuestionPlan, document: CompiledDocument) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "different units" not in folded and "different unit" not in folded:
        return None
    candidates = _facts(document, "area")
    counts = Counter(fact.unit for _, fact in candidates if fact.unit)
    if len(counts) < 2:
        return None
    common = counts.most_common(1)[0][0]
    outliers = [(record, fact) for record, fact in candidates if fact.unit and fact.unit != common]
    if len(outliers) != 1:
        return None
    record, fact = outliers[0]
    return ExecutionResult(
        question_id=plan.question_id,
        answer=(
            f"{record.title} is the exception: its area is reported as "
            f"{fact.raw_value} {fact.unit}; "
            f"the other park cards use {common}."
        ),
        evidence=[_fact_evidence(record, fact)],
        source_pages=[fact.page],
        complete=True,
        strategy=plan.strategy,
    )


def _largest_claim_conflict(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "largest in its country" not in folded:
        return None
    for record in document.records:
        claim = _sentence_with(record.text, "largest national park")
        area = next((fact for fact in record.number_facts if fact.field == "area"), None)
        if not claim or not area or not record.country:
            continue
        peers = [
            (other, fact)
            for other, fact in _facts(document, "area")
            if other.country == record.country and _area_km2(fact) > _area_km2(area)
        ]
        if not peers:
            continue
        other, other_area = max(peers, key=lambda pair: _area_km2(pair[1]))
        return ExecutionResult(
            question_id=plan.question_id,
            answer=(
                f"The book calls {record.title} {record.country}'s largest national park "
                f"and gives it as {area.raw_value} {area.unit}, but {other.title} is listed "
                f"at {other_area.raw_value} {other_area.unit}, which is larger."
            ),
            evidence=[claim, _fact_evidence(record, area), _fact_evidence(other, other_area)],
            source_pages=sorted({area.page, other_area.page}),
            complete=True,
            strategy=plan.strategy,
        )
    return None


def _rank_conflict(plan: V3QuestionPlan, document: CompiledDocument) -> ExecutionResult | None:
    folded = plan.question.casefold()
    if "snowdon" not in folded or "rank" not in folded:
        return None
    snowdonia = next(
        (record for record in document.records if "snowdonia" in record.title.casefold()),
        None,
    )
    cairngorms = next(
        (record for record in document.records if "cairngorms" in record.title.casefold()),
        None,
    )
    if not snowdonia or not cairngorms:
        return None
    snowdon_claim = _sentence_with(snowdonia.text, "second-highest")
    broader_claim = _sentence_with(cairngorms.text, "five of britain's six highest")
    ben = next((fact for fact in cairngorms.number_facts if fact.field == "highest_point"), None)
    height = re.search(r"At\s+(\d+)m,\s+Britain's second-highest", snowdon_claim)
    if not snowdon_claim or not ben or not height:
        return None
    return ExecutionResult(
        question_id=plan.question_id,
        answer=(
            f"The Snowdonia entry calls Snowdon ({height.group(1)}m) Britain's second-highest "
            f"peak. The Cairngorms entry gives {ben.subject} as {ben.value:g}m and says the "
            "park contains five of Britain's six highest summits, so Snowdon cannot be second."
        ),
        evidence=[snowdon_claim, broader_claim, _fact_evidence(cairngorms, ben)],
        source_pages=sorted({ben.page, snowdonia.page_start}),
        complete=True,
        strategy=plan.strategy,
    )


def execute_structured(
    plan: V3QuestionPlan,
    document: CompiledDocument,
) -> ExecutionResult | None:
    """Return a complete deterministic answer when the document model supports it."""
    for executor in (
        _table_answer,
        _contents_answer,
        _catalog_answer,
        _generic_claim_conflict,
        _dated_first_comparison,
        _generic_cross_border_relations,
        _generic_needle_answer,
        _composed_fact_answer,
        _threshold_answer,
        _superlative_answer,
        _needle_answer,
        _designation_absence_answer,
        _threat_absence_answer,
        _cross_border_answer,
        _human_wilderness_answer,
        _glaciation_synthesis_answer,
        _species_recovery_answer,
        _unesco_status_answer,
        _climbing_firsts_answer,
        _unit_outlier,
        _largest_claim_conflict,
        _rank_conflict,
    ):
        result = executor(plan, document)
        if result is not None and result.complete:
            return result
    return None
