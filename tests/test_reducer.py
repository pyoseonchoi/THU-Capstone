"""Tests for the deterministic reducer."""

from __future__ import annotations

import pytest

from app.reduction.deterministic_reducer import reduce
from app.schemas import (
    Condition,
    EntityRecord,
    EvidenceLedger,
    ExtractionStatus,
    Operator,
    QueryPlan,
)


def _make_plan(
    operator: Operator,
    target_fields: list[str] | None = None,
    conditions: list[Condition] | None = None,
    candidate_topics: list[str] | None = None,
) -> QueryPlan:
    return QueryPlan(
        question_id="q1",
        original_question="test",
        operator=operator,
        target_fields=target_fields or [],
        conditions=conditions or [],
        candidate_topics=candidate_topics or [],
    )


def _make_entity(name: str, fields: dict) -> EntityRecord:
    return EntityRecord(
        entity_name=name,
        normalized_name=name.lower(),
        fields=fields,
    )


def _make_ledger(
    entities: list[EntityRecord],
    total_chunks: int = 10,
    chunk_statuses: dict | None = None,
    topic_matrix: dict | None = None,
    plan: QueryPlan | None = None,
) -> EvidenceLedger:
    return EvidenceLedger(
        question_id="q1",
        plan=plan,
        entities=entities,
        total_chunks=total_chunks,
        chunk_statuses=chunk_statuses or {
            f"c{i}": ExtractionStatus.NO_EVIDENCE
            for i in range(total_chunks)
        },
        topic_matrix=topic_matrix or {},
    )


class TestFilterCountList:
    def test_basic_filter(self):
        entities = [
            _make_entity("Park A", {"highest_point": 3500}),
            _make_entity("Park B", {"highest_point": 2500}),
            _make_entity("Park C", {"highest_point": 4000}),
            _make_entity("Park D", {"highest_point": 1000}),
        ]
        plan = _make_plan(
            Operator.FILTER_COUNT_LIST,
            target_fields=["highest_point"],
            conditions=[Condition(field="highest_point", operator=">=", value="3000")],
        )
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == 2
        assert set(result.result_list) == {"Park A", "Park C"}

    def test_missing_fields_reported(self):
        entities = [
            _make_entity("Park A", {"highest_point": 3500}),
            _make_entity("Park B", {}),  # Missing field
        ]
        plan = _make_plan(
            Operator.FILTER_COUNT_LIST,
            target_fields=["highest_point"],
            conditions=[Condition(field="highest_point", operator=">=", value="3000")],
        )
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == 1
        assert "Park B" in result.missing_field_entities

    def test_each_condition_uses_its_own_field(self):
        entities = [
            _make_entity("A", {"height": 3500, "country": "France"}),
            _make_entity("B", {"height": 3600, "country": "Italy"}),
        ]
        plan = _make_plan(
            Operator.FILTER_COUNT_LIST,
            target_fields=["height"],
            conditions=[
                Condition(field="height", operator=">=", value="3000"),
                Condition(field="country", operator="contains", value="fran"),
            ],
        )

        result = reduce(plan, _make_ledger(entities))

        assert result.result_list == ["A"]


class TestArgmax:
    def test_basic_argmax(self):
        entities = [
            _make_entity("Park A", {"area": 500}),
            _make_entity("Park B", {"area": 1200}),
            _make_entity("Park C", {"area": 800}),
        ]
        plan = _make_plan(Operator.ARGMAX, target_fields=["area"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == 1200.0
        assert result.result_list == ["Park B"]

    def test_argmax_with_missing(self):
        entities = [
            _make_entity("Park A", {"area": 500}),
            _make_entity("Park B", {}),
        ]
        plan = _make_plan(Operator.ARGMAX, target_fields=["area"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == 500.0
        assert "Park B" in result.missing_field_entities

    def test_argmax_no_values(self):
        entities = [
            _make_entity("Park A", {}),
            _make_entity("Park B", {}),
        ]
        plan = _make_plan(Operator.ARGMAX, target_fields=["area"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value is None
        assert len(result.warnings) > 0


class TestArgmin:
    def test_basic_argmin(self):
        entities = [
            _make_entity("Park A", {"elevation": 500}),
            _make_entity("Park B", {"elevation": 100}),
            _make_entity("Park C", {"elevation": 800}),
        ]
        plan = _make_plan(Operator.ARGMIN, target_fields=["elevation"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == 100.0
        assert result.result_list == ["Park B"]


class TestAbsence:
    def test_topic_present(self):
        plan = _make_plan(
            Operator.ABSENCE,
            candidate_topics=["climate", "wildlife"],
        )
        topic_matrix = {
            "climate": {"c0": "substantive", "c1": "none"},
            "wildlife": {"c0": "none", "c1": "none"},
        }
        ledger = _make_ledger(
            entities=[],
            total_chunks=2,
            chunk_statuses={
                "c0": ExtractionStatus.EVIDENCE_FOUND,
                "c1": ExtractionStatus.NO_EVIDENCE,
            },
            topic_matrix=topic_matrix,
            plan=plan,
        )
        result = reduce(plan, ledger)

        assert "climate" not in result.result_list  # Present
        assert "wildlife" in result.result_list  # Absent

    def test_absence_fails_with_uncertain_chunks(self):
        plan = _make_plan(
            Operator.ABSENCE,
            candidate_topics=["geology"],
        )
        topic_matrix = {
            "geology": {"c0": "none"},
        }
        ledger = _make_ledger(
            entities=[],
            total_chunks=2,
            chunk_statuses={
                "c0": ExtractionStatus.NO_EVIDENCE,
                "c1": ExtractionStatus.UNCERTAIN,  # Unresolved!
            },
            topic_matrix=topic_matrix,
            plan=plan,
        )
        result = reduce(plan, ledger)

        # Cannot declare absent when coverage is incomplete
        assert "geology" not in result.result_list
        assert result.topic_verdicts["geology"] == "uncertain_incomplete_coverage"
        assert len(result.warnings) > 0

    def test_mention_only_not_substantive(self):
        plan = _make_plan(
            Operator.ABSENCE,
            candidate_topics=["archaeology"],
        )
        topic_matrix = {
            "archaeology": {"c0": "mention_only", "c1": "none"},
        }
        ledger = _make_ledger(
            entities=[],
            total_chunks=2,
            chunk_statuses={
                "c0": ExtractionStatus.EVIDENCE_FOUND,
                "c1": ExtractionStatus.NO_EVIDENCE,
            },
            topic_matrix=topic_matrix,
            plan=plan,
        )
        result = reduce(plan, ledger)

        assert result.topic_verdicts["archaeology"] == "mention_only"


class TestSumAverage:
    def test_sum(self):
        entities = [
            _make_entity("A", {"visitors": 1000}),
            _make_entity("B", {"visitors": 2000}),
            _make_entity("C", {"visitors": 3000}),
        ]
        plan = _make_plan(Operator.SUM, target_fields=["visitors"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == 6000.0

    def test_average(self):
        entities = [
            _make_entity("A", {"visitors": 1000}),
            _make_entity("B", {"visitors": 2000}),
            _make_entity("C", {"visitors": 3000}),
        ]
        plan = _make_plan(Operator.AVERAGE, target_fields=["visitors"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == pytest.approx(2000.0)


class TestPercentChange:
    def test_positive_change(self):
        entities = [
            _make_entity("Old", {"price": 100}),
            _make_entity("New", {"price": 150}),
        ]
        plan = _make_plan(Operator.PERCENT_CHANGE, target_fields=["price"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value == pytest.approx(50.0)

    def test_zero_denominator(self):
        entities = [
            _make_entity("Old", {"price": 0}),
            _make_entity("New", {"price": 100}),
        ]
        plan = _make_plan(Operator.PERCENT_CHANGE, target_fields=["price"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_value is None
        assert any("Zero" in w for w in result.warnings)

    def test_explicit_operand_order(self):
        entities = [
            _make_entity("New", {"price": 150}),
            _make_entity("Old", {"price": 100}),
        ]
        plan = _make_plan(Operator.PERCENT_CHANGE, target_fields=["price"])
        plan.operand_entities = ["Old", "New"]

        result = reduce(plan, _make_ledger(entities))

        assert result.result_value == pytest.approx(50.0)


class TestCompare:
    def test_comparison_table(self):
        entities = [
            _make_entity("A", {"area": 500, "height": 3000}),
            _make_entity("B", {"area": 800, "height": 2500}),
        ]
        plan = _make_plan(
            Operator.COMPARE,
            target_fields=["area", "height"],
        )
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert len(result.result_table) == 2
        assert result.result_table[0]["entity"] == "A"
        assert result.result_table[0]["area"] == 500


class TestTemporal:
    def test_sort_by_date(self):
        entities = [
            _make_entity("Event A", {"date": "2020-01-01"}),
            _make_entity("Event C", {"date": "2023-06-15"}),
            _make_entity("Event B", {"date": "2021-03-10"}),
        ]
        plan = _make_plan(Operator.TEMPORAL, target_fields=["date"])
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        assert result.result_list == ["Event C"]  # Latest
        assert result.result_value == "2023-06-15"


class TestNoSilentSkipping:
    """Verify that the reducer does not silently skip entities."""

    def test_all_entities_accounted_for(self):
        entities = [
            _make_entity("A", {"val": 100}),
            _make_entity("B", {"val": 200}),
            _make_entity("C", {}),  # Missing field
        ]
        plan = _make_plan(
            Operator.FILTER_COUNT_LIST,
            target_fields=["val"],
            conditions=[Condition(field="val", operator=">=", value="100")],
        )
        ledger = _make_ledger(entities)
        result = reduce(plan, ledger)

        total = (
            len(result.included_entities)
            + len(result.excluded_entities)
            + len(result.missing_field_entities)
        )
        assert total == 3, "Every entity must appear somewhere"
