"""Regression tests for domain-neutral repeated-entity compilation and execution."""

from __future__ import annotations

from app.schemas import DocumentPage, QuestionRequest
from app.v3.compiler import compile_document
from app.v3.models import (
    EvidenceCandidate,
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
