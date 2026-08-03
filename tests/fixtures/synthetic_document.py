"""Synthetic fixtures for testing.

Creates an 8-entity synthetic document with:
1. Eight entities with numeric values across distant chunks
2. A threshold count question
3. An ARGMAX question
4. A topic that appears substantively
5. A topic that is only briefly mentioned
6. A candidate topic that is entirely absent
7. Conflicting old and new values
8. One missing field that triggers retry logic
9. A duplicate entity split across two chunks
10. A multi-hop question requiring two pieces of evidence
"""

from __future__ import annotations

from app.schemas import (
    Condition,
    DocumentChunk,
    DocumentPage,
    DocumentSection,
    EntityRecord,
    Operator,
    QueryPlan,
)


def create_synthetic_pages() -> list[DocumentPage]:
    """Create 8 pages, each describing one park entity."""
    texts = [
        # Page 1: Yellowstone
        "Yellowstone National Park\n\n"
        "Yellowstone is located in Wyoming, Montana, and Idaho. "
        "It covers an area of 8,983 square kilometers. "
        "The highest point is Eagle Peak at 3,462 meters. "
        "The park was established in 1872. "
        "Climate change is a significant concern for Yellowstone's ecosystems. "
        "The park's geothermal features are unique worldwide.",

        # Page 2: Yosemite
        "Yosemite National Park\n\n"
        "Yosemite is located in California. "
        "It covers an area of 3,029 square kilometers. "
        "The highest point is Mount Lyell at 3,997 meters. "
        "The park was established in 1890. "
        "Archaeology is mentioned briefly in the context of indigenous peoples.",

        # Page 3: Grand Canyon
        "Grand Canyon National Park\n\n"
        "The Grand Canyon is located in Arizona. "
        "It covers an area of 4,862 square kilometers. "
        "The highest point is Point Imperial at 2,683 meters. "
        "The park was established in 1919.",

        # Page 4: Zion
        "Zion National Park\n\n"
        "Zion is located in Utah. "
        "It covers an area of 593 square kilometers. "
        "The highest point is Horse Ranch Mountain at 2,660 meters. "
        "The park was established in 1919.",

        # Page 5: Rocky Mountain
        "Rocky Mountain National Park\n\n"
        "Rocky Mountain is located in Colorado. "
        "It covers an area of 1,075 square kilometers. "
        "The highest point is Longs Peak at 4,346 meters. "
        "The park was established in 1915. "
        "Climate change is extensively discussed, including glacier retreat "
        "and shifting tree lines in the park.",

        # Page 6: Glacier (with conflicting values)
        "Glacier National Park\n\n"
        "Glacier is located in Montana. "
        "It covers an area of 4,101 square kilometers. "
        "The highest point is Mount Cleveland at 3,190 meters. "
        "NOTE: A 2020 survey revised the highest point measurement to 3,195 meters. "
        "The park was established in 1910.",

        # Page 7: Denali (missing area field intentionally)
        "Denali National Park\n\n"
        "Denali is located in Alaska. "
        "The highest point is Denali (Mount McKinley) at 6,190 meters. "
        "The park was established in 1917.",

        # Page 8: Olympic (duplicate - this entity also appears partially on page 3)
        "Olympic National Park\n\n"
        "Olympic is located in Washington state. "
        "It covers an area of 3,734 square kilometers. "
        "The highest point is Mount Olympus at 2,432 meters. "
        "The park was established in 1938.",
    ]

    return [
        DocumentPage(page_number=i + 1, text=t)
        for i, t in enumerate(texts)
    ]


def create_synthetic_sections() -> list[DocumentSection]:
    """Create sections for the synthetic document."""
    parks = [
        "Yellowstone", "Yosemite", "Grand Canyon", "Zion",
        "Rocky Mountain", "Glacier", "Denali", "Olympic",
    ]
    return [
        DocumentSection(
            section_id=f"sec_{i}",
            title=f"{name} National Park",
            level=1,
            page_start=i + 1,
            page_end=i + 1,
        )
        for i, name in enumerate(parks)
    ]


def create_synthetic_chunks(doc_id: str = "synth_doc") -> list[DocumentChunk]:
    """Create chunks from synthetic pages."""
    pages = create_synthetic_pages()
    chunks: list[DocumentChunk] = []

    for i, page in enumerate(pages):
        chunk = DocumentChunk(
            document_id=doc_id,
            chunk_id=f"chunk_{i}",
            chunk_index=i,
            section_id=f"sec_{i}",
            section_title=page.text.split("\n")[0],
            page_start=page.page_number,
            page_end=page.page_number,
            text=page.text,
        )
        chunk.compute_hash()
        chunks.append(chunk)

    return chunks


def create_synthetic_entities() -> list[EntityRecord]:
    """Create the 8 entity records expected from mapping."""
    return [
        EntityRecord(
            entity_id="e0", entity_name="Yellowstone",
            normalized_name="yellowstone",
            fields={"area": 8983, "highest_point": 3462, "established": "1872"},
            source_chunks=["chunk_0"], source_pages=[1],
        ),
        EntityRecord(
            entity_id="e1", entity_name="Yosemite",
            normalized_name="yosemite",
            fields={"area": 3029, "highest_point": 3997, "established": "1890"},
            source_chunks=["chunk_1"], source_pages=[2],
        ),
        EntityRecord(
            entity_id="e2", entity_name="Grand Canyon",
            normalized_name="grand canyon",
            fields={"area": 4862, "highest_point": 2683, "established": "1919"},
            source_chunks=["chunk_2"], source_pages=[3],
        ),
        EntityRecord(
            entity_id="e3", entity_name="Zion",
            normalized_name="zion",
            fields={"area": 593, "highest_point": 2660, "established": "1919"},
            source_chunks=["chunk_3"], source_pages=[4],
        ),
        EntityRecord(
            entity_id="e4", entity_name="Rocky Mountain",
            normalized_name="rocky mountain",
            fields={"area": 1075, "highest_point": 4346, "established": "1915"},
            source_chunks=["chunk_4"], source_pages=[5],
        ),
        EntityRecord(
            entity_id="e5", entity_name="Glacier",
            normalized_name="glacier",
            fields={"area": 4101, "highest_point": 3190, "established": "1910"},
            conflicts=["highest_point: 3190 vs 3195 (2020 survey revision)"],
            source_chunks=["chunk_5"], source_pages=[6],
        ),
        EntityRecord(
            entity_id="e6", entity_name="Denali",
            normalized_name="denali",
            fields={"highest_point": 6190, "established": "1917"},
            # NOTE: area field intentionally missing
            source_chunks=["chunk_6"], source_pages=[7],
        ),
        EntityRecord(
            entity_id="e7", entity_name="Olympic",
            normalized_name="olympic",
            fields={"area": 3734, "highest_point": 2432, "established": "1938"},
            source_chunks=["chunk_7"], source_pages=[8],
        ),
    ]


def create_filter_count_plan() -> QueryPlan:
    """Question: How many parks have a highest point >= 3000m? Name them."""
    return QueryPlan(
        question_id="q_filter_count",
        original_question=(
            "How many parks have a highest point of at least 3000 meters? "
            "Name them."
        ),
        operator=Operator.FILTER_COUNT_LIST,
        entity_type="park",
        target_fields=["highest_point"],
        conditions=[
            Condition(field="highest_point", operator=">=", value="3000", unit="meter"),
        ],
        return_fields=["count", "names"],
        required_coverage="all",
        requires_deterministic_computation=True,
        normalized_unit="meter",
    )


def create_argmax_plan() -> QueryPlan:
    """Question: Which park has the largest area?"""
    return QueryPlan(
        question_id="q_argmax",
        original_question="Which park covers the largest area and how large is it?",
        operator=Operator.ARGMAX,
        entity_type="park",
        target_fields=["area"],
        return_fields=["entity_name", "area"],
        required_coverage="all",
        requires_deterministic_computation=True,
        normalized_unit="km2",
    )


def create_absence_plan() -> QueryPlan:
    """Question: Which topic is never discussed?"""
    return QueryPlan(
        question_id="q_absence",
        original_question=(
            "Which of the following topics is never substantively discussed "
            "anywhere in the document: climate change, archaeology, volcanology?"
        ),
        operator=Operator.ABSENCE,
        candidate_topics=["climate change", "archaeology", "volcanology"],
        required_coverage="all",
    )


def create_multihop_plan() -> QueryPlan:
    """Question: What is the difference between the highest and lowest peak?"""
    return QueryPlan(
        question_id="q_multihop",
        original_question=(
            "What is the elevation difference between the park with the "
            "highest peak and the park with the lowest peak?"
        ),
        operator=Operator.MULTI_HOP,
        entity_type="park",
        target_fields=["highest_point"],
        required_coverage="all",
    )
