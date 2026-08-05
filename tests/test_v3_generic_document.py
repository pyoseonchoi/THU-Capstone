"""Regression tests for domain-neutral repeated-entity compilation and execution."""

from __future__ import annotations

from app.schemas import DocumentPage, QuestionRequest
from app.v3.compiler import compile_document
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    EvidenceCandidate,
    NumberFact,
    OperationKind,
    Strategy,
    V3MapResult,
    V3QuestionPlan,
)
from app.v3.question_compiler import compile_question
from app.v3.reducers import build_evidence_packet
from app.v3.structured_executor import execute_structured


def _station_pages() -> list[DocumentPage]:
    entities = [
        (
            "Alpha Ridge Observatory",
            "Asteria",
            (
                "Alpha is introduced as 'the highest station in Asteria'. "
                "The Skyvault Annex began as a weather-kite shed in 1911. "
                "On 2 March 1936, two researchers completed the first winter traverse. "
                "Alpha joined the tentative list for the International Field Heritage "
                "Register in 2018. Alpha straddles the border between Asteria and "
                "Cyrenia."
            ),
            (
                "100 Area monitored (sq km)\n"
                "3,050 Highest operating point (m)\n"
                "5,000 Annual visiting researchers\n"
                "1932 Year established"
            ),
        ),
        (
            "Beta Alpine Station",
            "Asteria",
            (
                "Beta was nominated for the International Field Heritage Register "
                "in 2020. Beta is a shared transboundary research network spanning "
                "Asteria and Demeria."
            ),
            (
                "Area monitored (sq km)\n200\n"
                "Highest operating point (m)\n3,180\n"
                "Annual visiting researchers\n6,000\n"
                "Year established\n1950"
            ),
        ),
        (
            "Gamma Urban Ecology Hub",
            "Demeria",
            (
                "Gamma is described as 'Demeria's largest protected research landscape'. "
                "On 3 March 1987, two divers completed the first sub-ice dive. Gamma "
                "has been formally paired since 2008 with Frostline Field Laboratory "
                "across the Demeria-Ilyria border."
            ),
            (
                "Area monitored: 950 sq km\n"
                "Highest operating point: 680 m\n"
                "Annual visiting researchers: 18,500\n"
                "Established: 1968"
            ),
        ),
        (
            "Delta Caldera Institute",
            "Demeria",
            (
                "Delta was inscribed on the International Field Heritage Register "
                "in 2012."
            ),
            (
                "1,400\nArea monitored (sq km)\n"
                "2,400\nHighest operating point (m)\n"
                "7,400\nAnnual visiting researchers\n"
                "1980\nYear established"
            ),
        ),
    ]
    pages = [
        DocumentPage(
            page_number=1,
            text=(
                "# Field Guide\n## Contents\n"
                "1. Alpha Ridge Observatory\n2. Beta Alpine Station\n"
                "3. Gamma Urban Ecology Hub\n4. Delta Caldera Institute"
            ),
        )
    ]
    page_number = 2
    for title, country, prose, card in entities:
        pages.append(DocumentPage(
            page_number=page_number,
            text=f"# {title}\n**Country: {country}**\n\n{prose}",
        ))
        pages.append(DocumentPage(
            page_number=page_number + 1,
            text=f"## Station in numbers\n\n{card}\n\n## Notes",
        ))
        page_number += 2
    return pages


def _plan(question_id: str, question: str, category: str):
    return compile_question(QuestionRequest(
        question_id=question_id,
        question=question,
        category=category,
    ))


def test_generic_compiler_builds_trusted_registry_and_mixed_number_cards():
    document = compile_document("stations", _station_pages())

    assert document.record_kind == "repeated_entity"
    assert document.entity_label == "station"
    assert document.registry_trusted is True
    assert document.registry_signals["fact_sections"] == 4
    assert document.registry_signals["contents_entries"] == 4
    assert [record.title for record in document.records] == [
        "Alpha Ridge Observatory",
        "Beta Alpine Station",
        "Gamma Urban Ecology Hub",
        "Delta Caldera Institute",
    ]
    assert [record.country for record in document.records] == [
        "Asteria",
        "Asteria",
        "Demeria",
        "Demeria",
    ]
    for record in document.records:
        assert {fact.field for fact in record.number_facts} == {
            "area",
            "highest_point",
            "annual_visitors",
            "establishment_year",
        }


def test_registry_is_not_trusted_when_contents_and_fact_sections_disagree():
    pages = _station_pages()
    pages[0].text = pages[0].text.replace("\n4. Delta Caldera Institute", "")

    document = compile_document("stations", pages)

    assert document.registry_signals["contents_entries"] == 3
    assert document.registry_signals["fact_sections"] == 4
    assert document.registry_trusted is False
    assert "Contents lists 3 entities but compiler found 4" in document.warnings


def test_question_compiler_emits_composed_operations_without_count_confusion():
    threshold = _plan(
        "q1",
        "Counting across every station, how many have a highest operating point "
        "of 3000 metres or more? Name them.",
        "aggregation",
    )
    superlative = _plan(
        "q2",
        "Which station reports the largest number of visiting researchers per year, "
        "and how many?",
        "superlative",
    )
    filtered = _plan(
        "q3",
        "Among stations described as inscribed, nominated, or tentative for the "
        "International Field Heritage Register, which has the largest annual number "
        "of visiting researchers?",
        "cross_section",
    )
    event = _plan(
        "q4",
        "For each event, combine its date with the station's establishment year and "
        "determine which occurred closer to founding and by how many years.",
        "cross_section",
    )
    relations = _plan(
        "q5",
        "Identify the station that straddles a border, the shared transboundary "
        "network, and the station formally paired with an unprofiled partner.",
        "cross_section",
    )

    assert [step.kind for step in threshold.operations] == [
        OperationKind.FILTER,
        OperationKind.COUNT,
        OperationKind.LIST,
    ]
    assert threshold.target_fields == ["highest_point"]
    assert [step.kind for step in superlative.operations] == [OperationKind.ARGMAX]
    assert [step.kind for step in filtered.operations] == [
        OperationKind.FILTER,
        OperationKind.ARGMAX,
    ]
    assert [step.kind for step in event.operations] == [
        OperationKind.JOIN,
        OperationKind.DATE_DIFFERENCE,
        OperationKind.COMPARE,
    ]
    assert [step.kind for step in relations.operations] == [OperationKind.JOIN]


def test_composed_structured_execution_is_consistent_and_complete():
    document = compile_document("stations", _station_pages())
    questions = {
        "count": _plan(
            "count",
            "Counting across every station, how many have a highest operating point "
            "of 3000 metres or more? Name them.",
            "aggregation",
        ),
        "visitors": _plan(
            "visitors",
            "Which station reports the largest number of visiting researchers per "
            "year, and how many?",
            "superlative",
        ),
        "status": _plan(
            "status",
            "Among stations described as inscribed, nominated, or tentative for the "
            "International Field Heritage Register, which reports the largest annual "
            "number of visiting researchers? State its exact status.",
            "cross_section",
        ),
        "rank": _plan(
            "rank",
            "The guide makes a claim about Alpha's rank among Asterian stations that "
            "its figures contradict. What is the claim and counterevidence?",
            "contradiction",
        ),
        "landscape": _plan(
            "landscape",
            "The guide calls one station the largest protected research landscape in "
            "its country but gives another a larger monitored area. Which two?",
            "contradiction",
        ),
        "events": _plan(
            "events",
            "Two profiles record a dated research first. Combine each event date with "
            "the station's establishment year and determine which occurred closer to "
            "founding and by how many years.",
            "cross_section",
        ),
        "needle": _plan(
            "needle",
            "What did the Skyvault Annex begin as, and in what year?",
            "needle",
        ),
        "relations": _plan(
            "relations",
            "The guide describes a station that physically straddles a border, a "
            "shared transboundary research network, and a profiled station formally "
            "paired with an unprofiled partner. Identify each arrangement, give the "
            "countries, and explain how the structures differ.",
            "cross_section",
        ),
    }
    answers = {
        key: execute_structured(plan, document)
        for key, plan in questions.items()
    }

    assert all(answer is not None for answer in answers.values())
    assert answers["count"].answer.startswith("2 stations qualify:")
    assert "Alpha Ridge Observatory" in answers["count"].answer
    assert "Beta Alpine Station" in answers["count"].answer
    assert "Gamma Urban Ecology Hub" in answers["visitors"].answer
    assert "18,500" in answers["visitors"].answer
    assert "Delta Caldera Institute" in answers["status"].answer
    assert "inscribed" in answers["status"].answer
    assert "Beta Alpine Station" in answers["rank"].answer
    assert "Gamma Urban Ecology Hub" in answers["landscape"].answer
    assert "Delta Caldera Institute" in answers["landscape"].answer
    assert "4 years after establishment" in answers["events"].answer
    assert "19 years after establishment" in answers["events"].answer
    assert "by 15 years" in answers["events"].answer
    assert answers["needle"].answer == (
        "The Skyvault Annex began as a weather-kite shed in 1911."
    )
    assert "Alpha Ridge Observatory" in answers["relations"].answer
    assert "Beta Alpine Station" in answers["relations"].answer
    assert "Frostline Field Laboratory" in answers["relations"].answer
    assert "Demeria-Ilyria" not in answers["relations"].answer


def test_synthesis_reducer_drops_evidence_from_an_unrelated_process():
    plan = V3QuestionPlan(
        question_id="fire",
        question=(
            "How does fire function across the different station profiles, "
            "as a natural process and a management problem?"
        ),
        category="global_synthesis",
        strategy=Strategy.HIERARCHICAL_SYNTHESIS,
    )
    result = V3MapResult(
        question_id="fire",
        record_id="record-001",
        status="evidence_found",
        evidence=[
            EvidenceCandidate(
                question_id="fire",
                record_id="record-001",
                claim="Controlled burning maintains the grassland.",
                exact_quote="Staff use controlled burning to maintain the grassland.",
                page=2,
                field="fire_management",
            ),
            EvidenceCandidate(
                question_id="fire",
                record_id="record-001",
                claim="The glacier has retreated since 1990.",
                exact_quote="The glacier has retreated rapidly since 1990.",
                page=3,
                field="glacier_retreat",
            ),
        ],
    )

    packet = build_evidence_packet(plan, [result], {"record-001"})

    assert [item.field for item in packet.evidence] == ["fire_management"]
    assert packet.warnings == ["Dropped 1 off-topic synthesis evidence items"]


def test_country_learned_from_document_labels_when_a_record_omits_it():
    """Country detection must not depend on the hardcoded European alias list.

    One record explicitly labels its (fictional, non-European) country; a
    second record only mentions that same country in prose, with no label at
    all. The compiler should still resolve the second record's country by
    reusing the name it already proved this document uses, not by matching
    against the closed `_COUNTRY_ALIASES` table.
    """
    pages = [
        DocumentPage(
            page_number=1,
            text=(
                "# Field Guide\n## Contents\n"
                "1. Kestrel Highland Reserve\n2. Marrow Delta Wetlands"
            ),
        ),
        DocumentPage(
            page_number=2,
            text=(
                "# Kestrel Highland Reserve\n"
                "**Country: Valdoria.**\n\n"
                "Kestrel protects a highland plateau in the north of the country."
            ),
        ),
        DocumentPage(
            page_number=3,
            text="## Station in numbers\n\n100 Area monitored (sq km)\n\n## Notes",
        ),
        DocumentPage(
            page_number=10,
            text=(
                "# Marrow Delta Wetlands\n"
                "Marrow sits downstream in southern Valdoria and shares its "
                "river system with Kestrel."
            ),
        ),
        DocumentPage(
            page_number=11,
            text="## Station in numbers\n\n200 Area monitored (sq km)\n\n## Notes",
        ),
    ]

    document = compile_document("reserves", pages)

    assert [record.country for record in document.records] == [
        "Valdoria",
        "Valdoria",
    ]


def _designation_record(record_id, title, ordinal, text, number_facts=None):
    return CompiledRecord(
        record_id=record_id, ordinal=ordinal, title=title,
        page_start=ordinal, page_end=ordinal, anchor_page=ordinal,
        text=f"[Page {ordinal}]\n{title}\n{text}",
        number_facts=number_facts or [],
    )


def test_generic_designation_status_answer_uses_no_practice_document_names():
    """The Unesco-status classifier must work for entirely invented entities.

    Uses fictional stations and a fictional heritage register -- none of the
    real practice document's park or designation names -- to prove the
    replacement for `_unesco_status_answer` generalizes by status vocabulary
    (inscribed/tentative/nominated/biosphere reserve) rather than by
    hardcoding which named entity holds which status.
    """
    records = [
        _designation_record(
            "s01", "Frostpeak Research Station", 1,
            "Frostpeak has been on the Antarctic Heritage World Heritage List "
            "since 1985.",
        ),
        _designation_record(
            "s02", "Silverbrook Field Station", 2,
            "Silverbrook entered the Antarctic Heritage World Heritage List "
            "in 1991.",
        ),
        _designation_record(
            "s03", "Windrift Outpost", 3,
            "The outpost has been on the tentative list of Antarctic Heritage "
            "World Heritage Sites.",
            number_facts=[NumberFact(
                record_id="s03", field="year_added_to_tentative_list",
                label="Year added to the tentative list", value=2014,
                raw_value="2014", page=3,
                quote="2014 Year added to the tentative list",
            )],
        ),
        _designation_record(
            "s04", "Halcyon Bay Camp", 4,
            "The camp was formally nominated for Antarctic Heritage World "
            "Heritage status in early 2009.",
        ),
        _designation_record(
            "s05", "Emberline Ridge Post", 5,
            "Emberline Ridge holds biosphere reserve status.",
        ),
        _designation_record(
            "s06", "Glasswater Depot", 6,
            "Glasswater is also recognised with biosphere reserve status.",
        ),
    ]
    document = CompiledDocument(
        document_id="doc", record_kind="repeated_entity", records=records,
    )
    plan = V3QuestionPlan(
        question_id="q1",
        question=(
            "Which stations are actually inscribed by Antarctic Heritage, and "
            "which are only nominated or on a tentative list?"
        ),
        strategy=Strategy.HIERARCHICAL_SYNTHESIS,
    )

    result = execute_structured(plan, document)

    assert result is not None
    assert "Frostpeak Research Station, since 1985" in result.answer
    assert "Silverbrook Field Station, since 1991" in result.answer
    assert "tentative list from 2014" in result.answer
    assert "formally nominated in early 2009" in result.answer
    assert "biosphere reserves" in result.answer
    assert "Emberline Ridge Post" in result.answer
    assert "Glasswater Depot" in result.answer


def test_generic_first_events_answer_uses_no_practice_document_names():
    """The climbing-firsts lister must work for entirely invented entities.

    Uses fictional peaks, climbers, and stations -- none of the real
    practice document's names -- to prove `_generic_first_events_answer`
    generalizes by sentence pattern ("first ascent/climb ... date") rather
    than by hardcoding Barre des Écrins or Snowdonia. Also checks that a
    location named in the sentence *before* the matching one still
    surfaces, since real prose often splits the subject and the "first X"
    claim across two sentences.
    """
    records = [
        _designation_record(
            "s01", "Cloudspire Research Station", 1,
            "Three surveyors made the first ascent of Mount Ashgrave on "
            "12 March 1902.",
        ),
        _designation_record(
            "s02", "Thistledown Outpost", 2,
            "The route follows the ridge above Hollow Tarn. Since 1911, "
            "local guides have marked it as the first recorded climb in "
            "the region.",
        ),
    ]
    document = CompiledDocument(
        document_id="doc", record_kind="repeated_entity", records=records,
    )
    plan = V3QuestionPlan(
        question_id="q1",
        question=(
            "Two station entries each record a climbing first with a date. "
            "What are the events and dates?"
        ),
        strategy=Strategy.HIERARCHICAL_SYNTHESIS,
    )

    result = execute_structured(plan, document)

    assert result is not None
    assert "Mount Ashgrave" in result.answer
    assert "12 March 1902" in result.answer
    assert "Hollow Tarn" in result.answer
    assert "1911" in result.answer


def test_claim_conflict_picks_true_maximum_over_first_worded_comparison():
    """A peer with an explicit "compared with" sentence must not outrank a
    larger, silently-reported peer.

    Regression test for a bug found via a real generalization document: when
    two peers both exceed the claimed figure, the executor was picking
    whichever one happened to phrase its rebuttal in prose ("larger than
    X"), even when a third peer's plain reported figure was larger still.
    Uses fictional reservoirs so the fix is proven by pattern, not by
    matching one specific document's wording.
    """
    claimant = CompiledRecord(
        record_id="r01", ordinal=1, title="Hollowmere Reservoir", country="Kestria",
        page_start=1, page_end=1, anchor_page=1,
        text=(
            "[Page 1]\nHollowmere Reservoir\n"
            "Hollowmere Reservoir is described as Kestria's largest protected "
            "reservoir landscape."
        ),
        number_facts=[NumberFact(
            record_id="r01", field="area", label="Area", value=950,
            raw_value="950 sq km", unit="sq km", page=1, quote="950 sq km",
        )],
    )
    worded_peer = CompiledRecord(
        record_id="r02", ordinal=2, title="Ashcombe Reservoir", country="Kestria",
        page_start=2, page_end=2, anchor_page=2,
        text=(
            "[Page 2]\nAshcombe Reservoir\n"
            "Ashcombe Reservoir is larger than Hollowmere, covering more ground."
        ),
        number_facts=[NumberFact(
            record_id="r02", field="area", label="Area", value=1400,
            raw_value="1,400 sq km", unit="sq km", page=2, quote="1,400 sq km",
        )],
    )
    true_max_peer = CompiledRecord(
        record_id="r03", ordinal=3, title="Brackenfen Reservoir", country="Kestria",
        page_start=3, page_end=3, anchor_page=3,
        text=(
            "[Page 3]\nBrackenfen Reservoir\n"
            "Brackenfen Reservoir is the newest addition to the Kestria basin "
            "network."
        ),
        number_facts=[NumberFact(
            record_id="r03", field="area", label="Area", value=7360,
            raw_value="7,360 sq km", unit="sq km", page=3, quote="7,360 sq km",
        )],
    )
    document = CompiledDocument(
        document_id="doc", record_kind="repeated_entity",
        records=[claimant, worded_peer, true_max_peer],
    )
    plan = V3QuestionPlan(
        question_id="q1",
        question=(
            "The guide calls one Kestrian reservoir the country's largest, but "
            "another profile reports a larger area. Which two are involved, and "
            "what are their areas?"
        ),
        category="contradiction",
        strategy=Strategy.HIERARCHICAL_SYNTHESIS,
    )

    result = execute_structured(plan, document)

    assert result is not None
    assert "Brackenfen Reservoir" in result.answer
    assert "7,360 sq km" in result.answer
    assert "Ashcombe Reservoir" not in result.answer
