"""Record profiler: source-literal validation and schema-free field discovery."""

from __future__ import annotations

import json

import pytest

from app.config import Settings
from app.llm.base import BaseLLMClient, LLMResponse
from app.llm.router import LLMRouter
from app.llm.usage_tracker import UsageTracker
from app.v3.compiler import field_id
from app.v3.models import CompiledDocument, CompiledRecord
from app.v3.record_profiler import RecordProfiler, metric_field


class ScriptedClient(BaseLLMClient):
    """Return a canned payload per record id."""

    def __init__(self, payloads: dict[str, dict]):
        self._payloads = payloads
        self.calls = 0

    async def chat(self, messages, *, model, chunk_id="", **kwargs) -> LLMResponse:
        self.calls += 1
        payload = self._payloads.get(chunk_id, {})
        return LLMResponse(content=json.dumps(payload))

    async def close(self) -> None:
        return None


def _settings(tmp_path) -> Settings:
    return Settings(
        llm_provider="openai_compatible",
        llm_base_url="http://localhost:1/v1",
        data_dir=tmp_path,
        max_concurrent_requests=2,
    )


def _record(record_id: str, text: str, title: str = "") -> CompiledRecord:
    return CompiledRecord(
        record_id=record_id,
        ordinal=int(record_id.split("-")[-1]),
        title=title,
        page_start=1,
        page_end=1,
        anchor_page=1,
        text=text,
    )


def _profiler(tmp_path, payloads):
    settings = _settings(tmp_path)
    client = ScriptedClient(payloads)
    router = LLMRouter(settings, client=client)
    return RecordProfiler(router, UsageTracker(), settings), client


@pytest.mark.asyncio
async def test_facts_must_be_quoted_from_the_record(tmp_path):
    record = _record(
        "record-001",
        "[Page 1]\nThe reservoir holds 412 megalitres at full supply.",
    )
    profiler, _ = _profiler(tmp_path, {
        "record-001": {
            "name": "",
            "category": "",
            "facts": [
                # Supported by the record.
                {"metric": "Capacity", "value": 412, "unit": "megalitres",
                 "quote": "holds 412 megalitres"},
                # Quote absent from the record: must be rejected.
                {"metric": "Depth", "value": 99, "unit": "m",
                 "quote": "depth of 99 m"},
                # Quote present but the number is not in it: must be rejected.
                {"metric": "Staff", "value": 7, "unit": "",
                 "quote": "at full supply"},
            ],
        }
    })

    document = CompiledDocument(
        document_id="d", record_kind="repeated_entity", records=[record]
    )
    await profiler.profile(document)

    fields = {fact.field for fact in record.number_facts}
    assert fields == {"capacity"}
    assert record.number_facts[0].value == 412


@pytest.mark.asyncio
async def test_title_filled_only_when_verbatim_and_unresolved(tmp_path):
    unresolved = _record(
        "record-001",
        "[Page 1]\nGlassfen Wetland Station monitors peat hydrology.",
        title="Record 1",
    )
    hallucinated = _record(
        "record-002",
        "[Page 2]\nA second site reports nothing by name.",
        title="Record 2",
    )
    settled = _record(
        "record-003",
        "[Page 3]\nRed Mesa Ecology Station tracks runoff.",
        title="Existing Title",
    )
    profiler, _ = _profiler(tmp_path, {
        "record-001": {"name": "Glassfen Wetland Station", "facts": []},
        "record-002": {"name": "Invented Site Name", "facts": []},
        "record-003": {"name": "Red Mesa Ecology Station", "facts": []},
    })

    document = CompiledDocument(
        document_id="d",
        record_kind="repeated_entity",
        records=[unresolved, hallucinated, settled],
    )
    await profiler.profile(document)

    assert unresolved.title == "Glassfen Wetland Station"
    assert hallucinated.title == "Record 2"
    assert settled.title == "Existing Title"


@pytest.mark.asyncio
async def test_catalog_keeps_repeated_metrics_and_drops_one_offs(tmp_path):
    records = [
        _record(f"record-{index:03d}", f"[Page {index}]\nArea covered 100 sq km.")
        for index in range(1, 5)
    ]
    records[0].text += " Legend counts 3 dragons."
    payloads = {
        f"record-{index:03d}": {
            "facts": [
                {"metric": "Area covered", "value": 100, "unit": "sq km",
                 "quote": "Area covered 100 sq km"},
            ]
        }
        for index in range(1, 5)
    }
    payloads["record-001"]["facts"].append(
        {"metric": "Dragons", "value": 3, "unit": "", "quote": "counts 3 dragons"}
    )

    profiler, _ = _profiler(tmp_path, payloads)
    document = CompiledDocument(
        document_id="d", record_kind="repeated_entity", records=records
    )
    await profiler.profile(document)

    assert document.field_catalog == ["area_covered"]
    assert any(f.field == "dragons" for f in records[0].number_facts)


@pytest.mark.asyncio
async def test_profiles_are_cached_across_runs(tmp_path):
    record = _record("record-001", "[Page 1]\nArea covered 100 sq km.")
    payloads = {
        "record-001": {
            "facts": [
                {"metric": "Area covered", "value": 100, "unit": "sq km",
                 "quote": "Area covered 100 sq km"},
            ]
        }
    }
    profiler, client = _profiler(tmp_path, payloads)
    document = CompiledDocument(
        document_id="d", record_kind="repeated_entity", records=[record]
    )
    await profiler.profile(document)
    assert client.calls == 1

    again = _record("record-001", "[Page 1]\nArea covered 100 sq km.")
    profiler2, client2 = _profiler(tmp_path, payloads)
    reloaded = CompiledDocument(
        document_id="d", record_kind="repeated_entity", records=[again]
    )
    await profiler2.profile(reloaded)
    assert client2.calls == 0
    assert again.number_facts[0].value == 100


@pytest.mark.asyncio
async def test_failed_profiles_are_reported_not_hidden(tmp_path):
    record = _record("record-001", "[Page 1]\nSomething unparseable.")

    class Broken(BaseLLMClient):
        async def chat(self, messages, **kwargs):
            raise RuntimeError("upstream down")

        async def close(self) -> None:
            return None

    settings = _settings(tmp_path)
    profiler = RecordProfiler(
        LLMRouter(settings, client=Broken()), UsageTracker(), settings
    )
    document = CompiledDocument(
        document_id="d", record_kind="repeated_entity", records=[record]
    )
    await profiler.profile(document)

    assert record.number_facts == []
    assert any("could not be profiled" in w for w in document.warnings)
    assert document.registry_signals["profile_failures"] == 1


def test_profiler_and_compiler_agree_on_field_identity():
    """Both stages must file one measurement under one field.

    Independent derivations would split a metric's coverage between the rows
    the layout parser bound and the rows the model read, so neither half ever
    reaches the completeness deterministic reduction requires.
    """
    for label in (
        "Maximum depth (m)",
        "Maximum depth: Blue Basin (m)",
        "Annual throughput",
    ):
        assert metric_field(label) == field_id(label)


def test_field_id_strips_record_specific_detail():
    # The subject measured and the unit vary per record; the field must not.
    assert field_id("Maximum depth: Blue Basin (m)") == "maximum_depth"
    assert field_id("Maximum depth: Red Basin (m)") == "maximum_depth"
    assert field_id("Maximum depth") == "maximum_depth"
    assert field_id("Maximum depth (m)") == "maximum_depth"
