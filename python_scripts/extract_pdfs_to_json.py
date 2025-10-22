"""extract_pdfs_to_json.py

Scan a directory for PDF files, extract text (per page) using PyMuPDF (fitz), and write a JSON file per PDF to the output directory.

Usage:
    python extract_pdfs_to_json.py --input data --output text_output --workers 4

The script preserves PDF basenames. If input contains subfolders, the output will mirror the relative path.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Any, List

import fitz  # PyMuPDF


def extract_text_from_pdf(pdf_path: Path) -> Dict[str, Any]:
    """Extracts text and minimal metadata from a PDF file.

    Returns a dictionary with keys: 'path', 'num_pages', 'pages' (list of page texts), 'metadata'.
    """
    doc = fitz.open(pdf_path)
    pages: List[str] = []
    for page in doc:
        try:
            pages.append(page.get_text("text"))
        except Exception as e:
            logging.exception("Failed to extract text from page %s of %s: %s", page.number, pdf_path, e)
            pages.append("")

    info = doc.metadata or {}
    doc.close()

    return {
        "path": str(pdf_path),
        "num_pages": len(pages),
        "pages": pages,
        "metadata": info,
    }


def write_json_output(data: Dict[str, Any], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def find_pdfs(root: Path) -> List[Path]:
    return [p for p in root.rglob("*.pdf") if p.is_file()]


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract text from PDFs to JSON files using PyMuPDF.")
    parser.add_argument("--input", "-i", default="data", help="Input directory to search for PDFs (default: data)")
    parser.add_argument("--output", "-o", default="text_output", help="Output directory for JSON files (default: text_output)")
    parser.add_argument("--workers", "-w", type=int, default=1, help="Number of worker processes (not implemented yet)")
    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    args = parser.parse_args()

    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=log_level, format="%(asctime)s %(levelname)s %(message)s")

    input_root = Path(args.input)
    output_root = Path(args.output)

    if not input_root.exists():
        logging.error("Input directory does not exist: %s", input_root)
        return 2

    pdfs = find_pdfs(input_root)
    if not pdfs:
        logging.info("No PDF files found under %s", input_root)
        return 0

    logging.info("Found %d PDF files under %s", len(pdfs), input_root)

    for pdf in pdfs:
        try:
            rel = pdf.relative_to(input_root)
        except Exception:
            rel = pdf.name

        out_file = (output_root / rel).with_suffix(".json")

        logging.info("Processing: %s -> %s", pdf, out_file)
        try:
            data = extract_text_from_pdf(pdf)
            write_json_output(data, out_file)
        except Exception as e:
            logging.exception("Failed to process %s: %s", pdf, e)

    logging.info("Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
