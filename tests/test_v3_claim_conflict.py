"""Resolving a claim-vs-figure contradiction without inventing one.

None of this fixture's wording, groups, or entity names come from any real
document — it exists to pin the general behaviour: a sentence that merely
contains "highest" is not a boast, a negated or hedged superlative is not a
claim, a superlative about something a record contains is not a claim about
the record, and a claim scoped to something this registry cannot delimit is
left unresolved rather than guessed at.
"""

from __future__ import annotations

from app.schemas import QuestionRequest
from app.v3.models import (
    CompiledDocument,
    CompiledRecord,
    NumberFact,
    QuestionShape,
    ShapePlan,
)
from app.v3.question_compiler import compile_question
from app.v3.structured_executor import execute_structured


def _record(
    ordinal: int,
    title: str,
    group: str,
    text: str,
    facts: list[tuple[str, str, float, str]],
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
        text=f"[Page {ordinal}]\n{title}\n\n{text}",
        number_facts=[
            NumberFact(
                record_id=record_id,
                field=field,
                label=label,
                value=value,
                raw_value=str(value),
                unit=unit,
                page=ordinal,
                quote=f"{label}: {value} {unit}".strip(),
            )
            for field, label, value, unit in facts
        ],
    )


def _document(records: list[CompiledRecord]) -> CompiledDocument:
    field_catalog = sorted({fact.field for record in records for fact in record.number_facts})
    return CompiledDocument(
        document_id="d",
        record_kind="repeated_entity",
        entity_label="station",
        records=records,
        field_catalog=field_catalog,
        registry_trusted=True,
    )


def _plan(question: str, category: str, document: CompiledDocument, shape: ShapePlan):
    return compile_question(
        QuestionRequest(question_id="q", question=question, category=category),
        document,
        None,
        shape,
    )


def test_a_sentence_merely_containing_the_word_is_not_a_claim():
    # No claim marker, no possessive — a descriptive phrase about a larger
    # feature the record sits within, not a boast about the record itself.
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "Gentle hills give way to steeper slopes, rearing up to the "
            "highest points of the Kestrel range."
        ), [("highest_point", "Highest point (m)", 800.0, "m")]),
        _record(2, "Beta Station", "Erebos", "A quiet outpost in the valley.", [
            ("highest_point", "Highest point (m)", 1500.0, "m"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, confident=True)
    question = "The guide makes a claim about a station's rank that its own figures contradict."
    plan = _plan(question, "contradiction", document, shape)

    assert execute_structured(plan, document) is None


def test_a_negated_superlative_is_not_a_claim():
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "Alpha may not be one of Erebos's largest stations, but its trails "
            "reward those who make the trip."
        ), [("area_covered", "Area covered (sq km)", 10.0, "sq km")]),
        _record(2, "Beta Station", "Erebos", "A larger outpost nearby.", [
            ("area_covered", "Area covered (sq km)", 90.0, "sq km"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, field="area_covered", confident=True)
    question = "The guide calls one station the largest in its country. Which two, and what areas?"
    plan = _plan(question, "contradiction", document, shape)

    assert execute_structured(plan, document) is None


def test_a_partitive_object_is_not_a_claim_about_the_record():
    # "Colony of gulls" is a claim about the gulls, not about the station.
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "Cliffs along the shoreline shelter the world's largest colony "
            "of gulls, a spectacle for early risers."
        ), [("area_covered", "Area covered (sq km)", 10.0, "sq km")]),
        _record(2, "Beta Station", "Erebos", "A larger outpost nearby.", [
            ("area_covered", "Area covered (sq km)", 90.0, "sq km"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, field="area_covered", confident=True)
    question = "The guide calls one station the largest in its country. Which two, and what areas?"
    plan = _plan(question, "contradiction", document, shape)

    assert execute_structured(plan, document) is None


def test_field_selection_prefers_the_matching_axis_over_a_coincidental_word():
    # Both of Alpha's own fields contain "highest"; only one is an elevation.
    # A claim about height must not bind to a wind-speed field that happens
    # to share the word, just because the claim sentence also mentions the
    # country the wind-speed field's name was slugified from.
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "The guide calls Alpha the home of Erebos's second-highest peak, "
            "described as a draw for climbers for a century."
        ), [
            ("highest_point", "Highest point (m)", 900.0, "m"),
            (
                "highest_recorded_wind_speed_in_erebos",
                "Highest recorded wind speed in Erebos (km/h)",
                300.0,
                "km/h",
            ),
        ]),
        _record(2, "Beta Station", "Erebos", "The tallest peak in the range.", [
            ("highest_point", "Highest point (m)", 1400.0, "m"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, confident=True)
    question = "The guide makes a claim about a station's rank that its own figures contradict."
    plan = _plan(question, "contradiction", document, shape)

    result = execute_structured(plan, document)

    assert result is not None
    assert "900" in result.answer
    assert "1400" in result.answer
    assert "300" not in result.answer


def test_a_claim_scoped_beyond_any_known_group_is_left_unresolved():
    # "Meridia's largest" spans several of the registry's own countries; no
    # single group here is known to hold every relevant peer, and guessing
    # narrow or wide both risk comparing against the wrong population.
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "Alpha is Meridia's largest research station, anchoring the "
            "continent's northern network."
        ), [("area_covered", "Area covered (sq km)", 400.0, "sq km")]),
        _record(2, "Beta Station", "Kallos", "A modest coastal outpost.", [
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, field="area_covered", confident=True)
    question = "The guide calls one station the largest in its country. Which two, and what areas?"
    plan = _plan(question, "contradiction", document, shape)

    assert execute_structured(plan, document) is None


def test_a_universally_scoped_claim_compares_against_the_whole_registry():
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "Alpha is the world's largest research station, dwarfing every "
            "other outpost on the network."
        ), [("area_covered", "Area covered (sq km)", 400.0, "sq km")]),
        _record(2, "Beta Station", "Kallos", "A modest coastal outpost.", [
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, field="area_covered", confident=True)
    question = "The guide calls one station the largest in its country. Which two, and what areas?"
    plan = _plan(question, "contradiction", document, shape)

    result = execute_structured(plan, document)

    assert result is not None
    assert "Alpha Station" in result.answer
    assert "Beta Station" in result.answer
    assert "900" in result.answer


def test_a_country_scoped_claim_is_checked_within_its_own_country():
    document = _document([
        _record(1, "Alpha Station", "Erebos", (
            "Alpha is Erebos's largest research station on the eastern plain."
        ), [("area_covered", "Area covered (sq km)", 400.0, "sq km")]),
        _record(2, "Gamma Station", "Erebos", "A newer western outpost.", [
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
        ]),
        _record(3, "Beta Station", "Kallos", "The largest station in its own right.", [
            ("area_covered", "Area covered (sq km)", 2000.0, "sq km"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.CLAIM_CONFLICT, field="area_covered", confident=True)
    question = "The guide calls one station the largest in its country. Which two, and what areas?"
    plan = _plan(question, "contradiction", document, shape)

    result = execute_structured(plan, document)

    assert result is not None
    assert "Alpha Station" in result.answer
    assert "Gamma Station" in result.answer
    # Beta's own figure is larger still, but it is in a different country and
    # was never part of Alpha's claim.
    assert "Beta Station" not in result.answer


def test_the_narrowest_contradiction_check_runs_before_the_generic_one():
    # A question about mismatched units must reach the unit-outlier answer
    # even when the router did not confidently shape it, rather than being
    # intercepted by the broader claim-vs-figure fallback that runs for any
    # unrouted contradiction question.
    document = _document([
        _record(1, "Alpha Station", "Erebos", "Reports its footprint in full.", [
            ("area_covered", "Area covered (sq km)", 400.0, "sq km"),
        ]),
        _record(2, "Beta Station", "Kallos", "Reports its footprint in full.", [
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
        ]),
        _record(3, "Gamma Station", "Kallos", "Uses the older imperial figure.", [
            ("area_covered", "Area covered (sq miles)", 50.0, "sq miles"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.SYNTHESIS, confident=False)
    question = (
        "Every station reports its area, but one is reported in different "
        "units from all the others. Which station, and what units?"
    )
    plan = _plan(question, "contradiction", document, shape)

    result = execute_structured(plan, document)

    assert result is not None
    assert "Gamma Station" in result.answer
    assert "sq miles" in result.answer


def test_the_measurement_the_question_names_decides_which_field_is_checked():
    """Two fields each have one odd unit; only one is what was asked about.

    Ordering by how many records report a field would pick whichever is more
    common, which is not the same as picking the one the question is about.
    """
    document = _document([
        _record(1, "Alpha Station", "Erebos", "Reports both figures.", [
            ("area_covered", "Area covered (sq km)", 400.0, "sq km"),
            ("ice_thickness", "Ice thickness (m)", 900.0, "m"),
        ]),
        _record(2, "Beta Station", "Kallos", "Reports both figures.", [
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
            ("ice_thickness", "Ice thickness (m)", 700.0, "m"),
        ]),
        _record(3, "Gamma Station", "Kallos", "Uses older units throughout.", [
            ("area_covered", "Area covered (sq miles)", 50.0, "sq miles"),
            ("ice_thickness", "Ice thickness (km)", 1.2, "km"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.SYNTHESIS, confident=False)
    question = (
        "Every station reports its ice thickness, but one is reported in "
        "different units from all the others. Which station, and what units?"
    )
    plan = _plan(question, "contradiction", document, shape)

    result = execute_structured(plan, document)

    assert result is not None
    assert "Gamma Station" in result.answer
    # The question asked about thickness, not area, even though both qualify.
    assert "km" in result.answer
    assert "sq miles" not in result.answer


def test_a_field_guessed_from_the_question_does_not_block_a_better_one():
    """A guessed field is a candidate, not a verdict.

    Matching the question's words against every field in the document can
    land on a field that merely shares a word — here "station area" reaching
    a field about years of activity. Restricting the check to that guess
    leaves the real answer unfound; ranking it first and moving on does not.
    """
    document = _document([
        _record(1, "Alpha Station", "Erebos", "Long-running site.", [
            ("years_of_activity_in_the_station_area", "Years of activity", 40.0, ""),
            ("area_covered", "Area covered (sq km)", 400.0, "sq km"),
        ]),
        _record(2, "Beta Station", "Kallos", "Newer site.", [
            ("years_of_activity_in_the_station_area", "Years of activity", 12.0, ""),
            ("area_covered", "Area covered (sq km)", 900.0, "sq km"),
        ]),
        _record(3, "Gamma Station", "Kallos", "Uses the older imperial figure.", [
            ("area_covered", "Area covered (sq miles)", 50.0, "sq miles"),
        ]),
    ])
    shape = ShapePlan(shape=QuestionShape.SYNTHESIS, confident=False)
    question = (
        "Every station area is reported, but one station is reported in "
        "different units from all the others. Which station, and what units?"
    )
    plan = _plan(question, "contradiction", document, shape)

    result = execute_structured(plan, document)

    assert result is not None
    assert "Gamma Station" in result.answer
    assert "sq miles" in result.answer
