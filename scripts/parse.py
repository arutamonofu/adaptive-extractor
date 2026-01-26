# scripts/ingest.py

import argparse
import sys
from pathlib import Path
from tqdm import tqdm

from aee.core.logging import setup_logging
from aee.core.config import settings
from aee.ingestion import DoclingParser, MarkerParser

logger = setup_logging()

PARSER_FACTORY = {
    "docling": (DoclingParser, settings.parsing.docling),
    "marker": (MarkerParser, settings.parsing.marker),
}

def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest PDFs: Parse and Clean.")
    parser.add_argument("--config", type=str, help="Path to custom yaml config.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing files.")
    args = parser.parse_args()

    if args.config:
        global settings
        settings = settings.load(args.config)

    input_dir = settings.paths.pdf_dir
    output_dir = settings.paths.parsed_dir
    
    if not input_dir.exists():
        logger.error(f"Input directory not found: {input_dir}")
        sys.exit(1)
    output_dir.mkdir(parents=True, exist_ok=True)

    parser_name = settings.parsing.parser
    if parser_name not in PARSER_FACTORY:
        logger.error(f"Unknown parser: {parser_name}. Check your config.")
        sys.exit(1)
    
    parser_cls, parser_config = PARSER_FACTORY[parser_name]
    doc_parser = parser_cls(config=parser_config)

    all_pdfs = sorted(list(input_dir.glob("*.pdf")))
    if not all_pdfs:
        logger.warning(f"No PDF files found in {input_dir}")
        sys.exit(0)

    logger.info(f"Starting ingestion using {parser_name}...")
    
    success_count = 0
    overwrite = args.overwrite or settings.parsing.overwrite

    for pdf_path in tqdm(all_pdfs, desc="Parsing"):
        output_path = output_dir / f"{pdf_path.stem}.json"
        
        if output_path.exists() and not overwrite:
            continue

        try:
            processed_doc = doc_parser.parse(pdf_path)
            with open(output_path, "w", encoding="utf-8") as f:
                f.write(processed_doc.model_dump_json(indent=2))
            success_count += 1
        except Exception as e:
            logger.error(f"Failed to process {pdf_path.name}: {e}")

    logger.info(f"Ingestion complete. Processed {success_count} new documents.")

if __name__ == "__main__":
    main()