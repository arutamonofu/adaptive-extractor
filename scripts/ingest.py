# scripts/ingest.py

import argparse
from pathlib import Path
from tqdm import tqdm

from aee.core.logging import setup_logging
from aee.ingestion import DoclingParser, TextCleaner, PyMuPDFParser

logger = setup_logging()

def main():
    parser = argparse.ArgumentParser(description="Convert PDFs to JSON for AI agents.")
    parser.add_argument("--input", type=str, default="data/raw")
    parser.add_argument("--output", type=str, default="data/processed")
    parser.add_argument("--parser", type=str, default="docling", choices=["docling", "pymupdf"])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    # Setup
    in_dir = Path(args.input)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Init Parser
    if args.parser == "docling":
        doc_parser = DoclingParser()
    else:
        doc_parser = PyMuPDFParser()

    files = list(in_dir.glob("*.pdf"))
    logger.info(f"Found {len(files)} PDFs. Starting ingestion using {args.parser}...")

    for pdf in tqdm(files, desc="Ingesting"):
        try:
            json_out = out_dir / f"{pdf.stem}.json"
            if json_out.exists() and not args.force:
                continue

            # Parse & Clean
            doc = doc_parser.parse(pdf)
            doc.text_content = TextCleaner.clean_docling_markdown(doc.text_content)

            # Save
            with open(json_out, "w", encoding="utf-8") as f:
                f.write(doc.model_dump_json(indent=2))

        except Exception as e:
            logger.error(f"Failed {pdf.name}: {e}")

if __name__ == "__main__":
    main()