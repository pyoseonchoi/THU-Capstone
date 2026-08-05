"""Answers must carry the specifics the evidence already establishes."""

from __future__ import annotations

from app.schemas import QuestionRequest
from app.v3.answerer import _parse_answer
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    EvidencePacket,
    NumberFact,
    QuestionShape,
    ShapePlan,
    Strategy,
    TopicAssessment,
    V3MapResult,
    V3QuestionPlan,
)
from app.v3.question_compiler import compile_question
from app.v3.reducers import reduce_absence, reduce_mapped_structure
from app.v3.structured_executor import execute_structured


def _record(
    ordinal: int, title: str, group: str, value: float, subject: str = ""
) -> CompiledRecord:
    record_id = f"record-{ordinal:03d}"
    return CompiledRecord(
        record_id=record_id,
        ordinal=ordinal,
        title=title,
        country=group,
        page_start=ordinal,
        page_end=ordinal,
        anchor_page=ordinal,
        text=f"[Page {ordinal}]\n{title}",
        number_facts=[
            NumberFact(
                record_id=record_id,
                field="highest_point",
                label="Highest point (m)",
                value=value,
                raw_value=str(int(value)),
                unit="m",
                subject=subject,
                page=ordinal,
                quote=f"Highest point: {subject} {value:g}",
            )
        ],
    )


def _document() -> CompiledDocument:
    return CompiledDocument(
        document_id="d",
        record_kind="repeated_entity",
        entity_label="park",
        records=[
            _record(1, "Alpha Park", "France", 4102, "Barre des Alpha"),
            _record(2, "Beta Park", "Spain", 3479, "Mulhacen"),
            _record(3, "Gamma Park", "Italy", 3350, "Etna"),
        ],
        field_catalog=["highest_point"],
        registry_trusted=True,
    )


def _plan(question: str, category: str, document, shape: ShapePlan):
    return compile_question(
        QuestionRequest(question_id="q", question=question, category=category),
        document,
        None,
        shape,
    )


def test_extremum_names_what_was_measured_and_where():
    document = _document()
    shape = ShapePlan(
        shape=QuestionShape.EXTREMUM,
        field="highest_point",
        direction="max",
        confident=True,
    )

    result = execute_structured(
        _plan("Which park has the highest summit?", "superlative", document, shape),
        document,
    )

    assert result is not None
    assert "Alpha Park" in result.answer
    # The summit's own name and the park's country are already bound to the
    # winning row, and the question asks for both.
    assert "Barre des Alpha" in result.answer
    assert "France" in result.answer
    # A superlative is a comparison, so the runners-up belong in the answer.
    assert "Beta Park" in result.answer


def test_mapped_extremum_carries_the_same_specifics():
    document = _document()
    shape = ShapePlan(
        shape=QuestionShape.EXTREMUM,
        field="highest_point",
        direction="max",
        confident=True,
    )
    plan = _plan("Which park has the highest summit?", "superlative", document, shape)
    packet = EvidencePacket(
        question_id="q",
        expected_records=3,
        records_scanned=3,
        complete=True,
    )

    result = reduce_mapped_structure(plan, packet, document)

    assert result is not None
    assert "Barre des Alpha" in result.answer
    assert "France" in result.answer
    assert "Beta Park" in result.answer


def test_absence_names_where_each_present_subject_appears():
    document = _document()
    plan = V3QuestionPlan(
        question_id="q",
        question="Of glaciation, wildfire and tourism, which is never raised?",
        category="absence",
        strategy=Strategy.ABSENCE_MATRIX,
        candidate_topics=["glaciation", "wildfire", "tourism"],
    )
    results = []
    for index, record in enumerate(document.records, start=1):
        topics = []
        for topic in plan.candidate_topics:
            present = topic != "wildfire" and index == 1
            topics.append(TopicAssessment(
                question_id="q",
                record_id=record.record_id,
                topic=topic,
                level="substantive" if present else "none",
                exact_quote=f"{topic} is discussed here" if present else "",
                page=index,
            ))
        results.append(V3MapResult(
            question_id="q",
            record_id=record.record_id,
            status="evidence_found",
            topics=topics,
        ))

    result = reduce_absence(
        plan,
        results,
        {record.record_id for record in document.records},
        document=document,
    )

    assert result is not None
    assert "wildfire is the only subject" in result.answer
    # Saying the others are covered is a claim; naming where is the evidence.
    assert "Alpha Park" in result.answer


def test_an_object_shaped_answer_keeps_its_figures():
    # A model that answers with its own object must not lose the number to a
    # Python repr, which is how "475 BC" became "{'date': 'BC'}".
    assert _parse_answer('{"answer": {"date": "475 BC"}}') == "date: 475 BC"
    assert "475" in _parse_answer('{"year": 475, "era": "BC"}')
    assert _parse_answer('{"answer": "Plain prose."}') == "Plain prose."


def test_a_dismissed_record_carrying_rare_question_terms_is_revisited():
    from app.pipeline import FullScanPipeline, _distinctive_terms

    document = _document()
    # Only one record mentions the climb; every record mentions parks.
    document.records[2].text += " Britain's first recorded rock climb on Clogwyn."
    question = "Which entry records the first rock climb on Clogwyn?"

    terms = _distinctive_terms(question, document.records)

    # A word every record carries cannot single one out, so it is not a term.
    assert "clogwyn" in terms
    assert "entry" not in terms

    results = [
        V3MapResult(question_id="q", record_id=record.record_id, status="no_evidence")
        for record in document.records
    ]
    plan = V3QuestionPlan(
        question_id="q",
        question=question,
        category="needle",
        strategy=Strategy.EXHAUSTIVE_LOOKUP,
    )

    revisited = FullScanPipeline._dismissed_matches(plan, document, results)

    assert [record.record_id for record in revisited] == ["record-003"]


def test_nothing_is_revisited_when_the_scan_found_evidence_everywhere():
    from app.pipeline import FullScanPipeline

    document = _document()
    document.records[2].text += " Britain's first recorded rock climb on Clogwyn."
    results = [
        V3MapResult(question_id="q", record_id=record.record_id, status="evidence_found")
        for record in document.records
    ]
    plan = V3QuestionPlan(
        question_id="q",
        question="Which entry records the first rock climb on Clogwyn?",
        category="needle",
        strategy=Strategy.EXHAUSTIVE_LOOKUP,
    )

    assert FullScanPipeline._dismissed_matches(plan, document, results) == []
