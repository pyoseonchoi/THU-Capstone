"""Document inspection tool for debugging."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.chunking.structural_chunker import chunk_document, generate_chunk_manifest
from app.parsing.page_quality import generate_quality_report
from app.parsing.pdf_parser import parse_pdf
from app.parsing.structure_detector import detect_sections


def main():
    parser = argparse.ArgumentParser(description="Inspect a PDF document")
    parser.add_argument("pdf", type=Path, help="Path to PDF")
    parser.add_argument("--pages", action="store_true", help="Show page details")
    parser.add_argument("--sections", action="store_true", help="Show sections")
    parser.add_argument("--chunks", action="store_true", help="Show chunks")
    parser.add_argument("--quality", action="store_true", help="Show quality report")
    parser.add_argument(
        "--target-tokens", type=int, default=3500,
        help="Chunk target tokens",
    )
    args = parser.parse_args()

    metadata, pages = parse_pdf(args.pdf)
    sections = detect_sections(pages)
    quality = generate_quality_report(pages)

    print(f"\nDocument: {metadata.filename}")
    print(f"File hash: {metadata.file_hash}")
    print(f"Pages: {metadata.page_count}")
    print(f"Total characters: {metadata.total_characters}")

    if args.quality:
        print(f"\n{'='*40} Page Quality {'='*40}")
        for q in quality:
            status = "⚠️ " if q.low_text_warning else "  "
            print(
                f"  {status}Page {q.page_number:3d}: "
                f"{q.character_count:6d} chars, "
                f"{q.block_count:3d} blocks, "
                f"{q.image_count:2d} images, "
                f"status={q.extraction_status.value}"
            )

    if args.sections:
        print(f"\n{'='*40} Sections {'='*40}")
        for s in sections:
            print(
                f"  L{s.level} [{s.page_start}-{s.page_end}] {s.title}"
            )

    chunks = chunk_document(
        metadata.document_id, pages, sections,
        target_tokens=args.target_tokens,
    )

    print(f"\nChunks: {len(chunks)}")

    if args.chunks:
        manifest = generate_chunk_manifest(chunks)
        print(f"\n{'='*40} Chunks {'='*40}")
        for c in manifest["chunks"]:
            print(
                f"  #{c['chunk_index']:3d} [{c['page_start']}-{c['page_end']}] "
                f"~{c['tokens_approx']} tokens  {c['section_title'][:50]}"
            )

    if args.pages:
        print(f"\n{'='*40} Page Text Preview {'='*40}")
        for p in pages[:5]:
            print(f"\n--- Page {p.page_number} ---")
            print(p.text[:500])


if __name__ == "__main__":
    main()
