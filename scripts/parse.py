#!/usr/bin/env python3
"""PDF parsing and ingestion script for AutoEvoExtractor."""

import argparse
import sys
from pathlib import Path
from typing import List, Optional
from tqdm import tqdm

from aee import setup_logging, settings
from aee.ingestion import DoclingParser, MarkerParser
from aee.data import ProcessedDocument

logger = setup_logging()

PARSER_FACTORY = {
    "docling": (DoclingParser, settings.parsing.docling),
    "marker": (MarkerParser, settings.parsing.marker),
}

def get_parser_instance(parser_name: str):
    """Initialize and return a parser instance.
    
    Args:
        parser_name: Name of the parser to initialize.
        
    Returns:
        Initialized parser instance.
        
    Raises:
        ValueError: If parser name is not supported.
    """
    if parser_name not in PARSER_FACTORY:
        raise ValueError(f"Unknown parser: {parser_name}")
    
    parser_cls, parser_config = PARSER_FACTORY[parser_name]
    return parser_cls(config=parser_config)

def load_settings(config_path: Optional[str] = None):
    """Load application settings with optional custom config.
    
    Args:
        config_path: Path to custom configuration file.
        
    Returns:
        Loaded settings instance.
    """
    if config_path:
        return settings.load(config_path)
    return settings

def process_document(
    pdf_path: Path, 
    output_dir: Path, 
    doc_parser, 
    overwrite: bool = False
) -> bool:
    """Process a single PDF document.
    
    Args:
        pdf_path: Path to the input PDF file.
        output_dir: Directory to save processed documents.
        doc_parser: Parser instance to use.
        overwrite: Whether to overwrite existing files.
        
    Returns:
        True if processing was successful, False otherwise.
    """
    output_path = output_dir / f"{pdf_path.stem}.json"
    
    # Skip if file exists and overwrite is disabled
    if output_path.exists() and not overwrite:
        logger.debug(f"Skipping {pdf_path.name} (already exists)")
        return True

    try:
        processed_doc: ProcessedDocument = doc_parser.parse(pdf_path)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(processed_doc.model_dump_json(indent=2))
        logger.debug(f"Successfully processed {pdf_path.name}")
        return True
    except Exception as e:
        logger.error(f"Failed to process {pdf_path.name}: {e}")
        return False

def process_documents(
    input_dir: Path, 
    output_dir: Path, 
    parser_name: str, 
    overwrite: bool = False
) -> int:
    """Process all PDF documents in the input directory.
    
    Args:
        input_dir: Directory containing PDF files.
        output_dir: Directory to save processed documents.
        parser_name: Name of the parser to use.
        overwrite: Whether to overwrite existing files.
        
    Returns:
        Number of successfully processed documents.
    """
    # Find all PDF files
    all_pdfs: List[Path] = sorted(list(input_dir.glob("*.pdf")))
    
    if not all_pdfs:
        logger.warning(f"No PDF files found in {input_dir}")
        return 0

    logger.info(f"Found {len(all_pdfs)} PDF files to process")
    logger.info(f"Starting ingestion using {parser_name}...")
    
    # Initialize parser
    try:
        doc_parser = get_parser_instance(parser_name)
    except ValueError as e:
        logger.error(str(e))
        sys.exit(1)

    # Process documents
    success_count = 0
    for pdf_path in tqdm(all_pdfs, desc="Parsing"):
        if process_document(pdf_path, output_dir, doc_parser, overwrite):
            success_count += 1

    return success_count

def main() -> None:
    """Main entry point for the PDF parsing script."""
    parser = argparse.ArgumentParser(description="Ingest PDFs: Parse and Clean.")
    parser.add_argument("--config", type=str, help="Path to custom yaml config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    # Load settings
    app_settings = load_settings(args.config)
    
    # Validate directories
    input_dir = app_settings.paths.pdf_dir
    output_dir = app_settings.paths.parsed_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get parser configuration
    parser_name = app_settings.parsing.parser
    overwrite = args.overwrite or app_settings.parsing.overwrite

    # Process documents
    success_count = process_documents(input_dir, output_dir, parser_name, overwrite)
    
    logger.info(f"Ingestion complete. Processed {success_count} new documents.")

if __name__ == "__main__":
    main()