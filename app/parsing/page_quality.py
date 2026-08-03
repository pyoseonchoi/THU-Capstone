"""Page quality assessment and reporting."""

from __future__ import annotations

from app.logging_config import get_logger
from app.schemas import DocumentPage, PageExtractionStatus, PageQualityRecord

logger = get_logger("parsing.page_quality")


def generate_quality_report(pages: list[DocumentPage]) -> list[PageQualityRecord]:
    """Generate a quality report for all pages.

    Returns:
        List of PageQualityRecord, one per page.
    """
    records: list[PageQualityRecord] = []
    for page in pages:
        if page.quality is not None:
            records.append(page.quality)
        else:
            records.append(PageQualityRecord(
                page_number=page.page_number,
                character_count=len(page.text),
                extraction_status=PageExtractionStatus.OK,
            ))

    low_text_pages = [r for r in records if r.low_text_warning]
    if low_text_pages:
        logger.warning(
            "%d pages with low/no extracted text: %s",
            len(low_text_pages),
            [r.page_number for r in low_text_pages],
        )

    return records


def get_unresolved_pages(quality_report: list[PageQualityRecord]) -> list[int]:
    """Return page numbers that could not be fully extracted."""
    return [
        r.page_number
        for r in quality_report
        if r.extraction_status in (
            PageExtractionStatus.NO_TEXT,
            PageExtractionStatus.UNRESOLVED,
        )
    ]
