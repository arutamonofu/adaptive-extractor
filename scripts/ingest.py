# scripts/ingest.py

import argparse
import logging
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, NamedTuple, Optional, Type

from tqdm import tqdm

from aee.core.logging import setup_logging
from aee.core.types import DocumentMetadata, ProcessedDocument
from aee.ingestion import BaseParser, DoclingParser, PyMuPDFParser

logger = setup_logging()

# --- Configuration ---

# Separators used to help the LLM distinguish context.
# Matches the logic required for 'target_source' field in extraction tasks.
HEADER_MAIN = "# MAIN ARTICLE"
HEADER_SI = "# SUPPLEMENTARY INFORMATION"
SEPARATOR_VISUAL = "=" * 40

# Regex to detect Supplementary Information files.
# Matches: _si.pdf, -supp.pdf, .supplementary.pdf, etc. (case-insensitive)
SI_SUFFIX_PATTERN = re.compile(
    r"[\_\-\.](si|supp|supplementary|supporting information|supporting_info)$",
    re.IGNORECASE,
)

PARSER_REGISTRY: Dict[str, Type[BaseParser]] = {
    "docling": DoclingParser,
    "pymupdf": PyMuPDFParser,
}


class FileGroup(NamedTuple):
    """Container for matched file pairs."""
    doc_id: str
    main_path: Path
    si_path: Optional[Path] = None


# --- Core Logic ---


def group_files(file_paths: List[Path]) -> List[FileGroup]:
    """
    Groups PDF files into bundles (Main Article + Optional SI) based on filenames.

    Strategy:
    1. Identify files ending with SI suffixes.
    2. Match them to canonical filenames (by stripping the suffix).
    3. Treat remaining files as Main Articles.

    Args:
        file_paths: List of paths to PDF files.

    Returns:
        List of FileGroup objects representing unique documents.
    """
    groups: Dict[str, Dict[str, Path]] = defaultdict(dict)

    for path in file_paths:
        stem = path.stem
        match = SI_SUFFIX_PATTERN.search(stem)

        if match:
            # It's an SI file. Canonical ID is the stem without the suffix.
            canonical_id = stem[: match.start()].lower()
            groups[canonical_id]["si"] = path
        else:
            # It's a Main file.
            canonical_id = stem.lower()
            groups[canonical_id]["main"] = path

    # Convert to immutable NamedTuples, filtering out orphans (SI without Main)
    results = []
    for doc_id, bundle in groups.items():
        if "main" in bundle:
            results.append(
                FileGroup(
                    doc_id=doc_id,
                    main_path=bundle["main"],
                    si_path=bundle.get("si"),
                )
            )
        else:
            logger.warning(
                f"Found orphan SI file without main article: {bundle.get('si')}. Skipping."
            )

    return sorted(results, key=lambda x: x.doc_id)


def merge_documents(
    main_doc: ProcessedDocument, si_doc: ProcessedDocument, si_filename: str
) -> ProcessedDocument:
    """
    Merges Main and SI documents into a single ProcessedDocument with context markers.
    Handles Pydantic immutability for metadata.
    """
    # 1. Merge Text Content with explicit headers for the LLM
    merged_content = (
        f"{HEADER_MAIN}\n\n"
        f"{main_doc.text_content}\n\n"
        f"{SEPARATOR_VISUAL}\n"
        f"{HEADER_SI}\n\n"
        f"{si_doc.text_content}"
    )

    # 2. Merge Metadata (Metadata is frozen, so we reconstruct it)
    total_pages = (main_doc.metadata.page_count or 0) + (si_doc.metadata.page_count or 0)
    
    # Extend 'extra' dictionary safely
    new_extra = main_doc.metadata.extra.copy()
    new_extra.update(
        {
            "has_si": True,
            "si_filename": si_filename,
            "si_page_count": si_doc.metadata.page_count,
        }
    )

    new_metadata = DocumentMetadata(
        source_path=main_doc.metadata.source_path,
        filename=main_doc.metadata.filename,
        page_count=total_pages,
        extra=new_extra,
    )

    # 3. Create new document
    # Note: Lists (tables, images) are simply concatenated
    return ProcessedDocument(
        text_content=merged_content,
        tables=main_doc.tables + si_doc.tables,
        images=main_doc.images + si_doc.images,
        metadata=new_metadata,
    )


def process_single_group(
    group: FileGroup, parser: BaseParser, output_dir: Path, force: bool
) -> bool:
    """
    Parses and merges files for a single document ID.

    Returns:
        True if processed, False if skipped (exists) or failed.
    """
    output_path = output_dir / f"{group.doc_id}.json"

    if output_path.exists() and not force:
        return False

    try:
        # A. Parse Main Document
        main_doc = parser.parse(group.main_path)
        final_doc = main_doc

        # B. Parse and Merge SI (if exists)
        if group.si_path:
            logger.debug(f"[{group.doc_id}] Parsing SI: {group.si_path.name}")
            si_doc = parser.parse(group.si_path)
            final_doc = merge_documents(main_doc, si_doc, group.si_path.name)
        else:
            # Just add header to main doc for consistency
            final_doc.text_content = f"{HEADER_MAIN}\n\n{final_doc.text_content}"

        # C. Save to disk
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(final_doc.model_dump_json(indent=2))

        return True

    except Exception as e:
        logger.error(f"Failed to process '{group.doc_id}': {e}", exc_info=True)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest PDFs: Parse, Clean, and Merge SI.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=str,
        default="data/raw",
        help="Directory containing raw PDF files.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/processed",
        help="Directory to save processed JSON files.",
    )
    parser.add_argument(
        "--parser",
        type=str,
        default="docling",
        choices=PARSER_REGISTRY.keys(),
        help="Select the parsing engine.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing processed files.",
    )
    args = parser.parse_args()

    # --- 1. Validation & Setup ---
    input_dir = Path(args.input)
    output_dir = Path(args.output)

    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)

    output_dir.mkdir(parents=True, exist_ok=True)

    # Initialize Parser
    logger.info(f"Initializing parser: {args.parser}...")
    try:
        parser_cls = PARSER_REGISTRY[args.parser]
        doc_parser = parser_cls()
    except Exception as e:
        logger.error(f"Failed to initialize parser: {e}")
        sys.exit(1)

    # --- 2. Scanning ---
    logger.info(f"Scanning {input_dir}...")
    # Security: Ensure we only pick up PDFs
    all_pdfs = sorted(list(input_dir.glob("*.pdf")))
    
    if not all_pdfs:
        logger.warning("No PDF files found.")
        sys.exit(0)

    # Group files
    file_groups = group_files(all_pdfs)
    logger.info(
        f"Found {len(all_pdfs)} PDFs, grouped into {len(file_groups)} unique documents."
    )

    # --- 3. Execution ---
    success_count = 0
    with tqdm(total=len(file_groups), desc="Ingesting") as pbar:
        for group in file_groups:
            if process_single_group(group, doc_parser, output_dir, args.force):
                success_count += 1
            pbar.update(1)

    logger.info(f"✅ Ingestion complete. Processed {success_count} new documents.")


if __name__ == "__main__":
    main()