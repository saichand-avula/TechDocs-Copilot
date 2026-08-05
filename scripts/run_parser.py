"""
run_parser.py
=============
CLI entry point for the TechDocs Copilot document parser.

Usage
-----
    # Run with Docling (default)
    python scripts/run_parser.py \\
        --pdf  data/raw/manuals/01_HP_LaserJet_Repair_Manual_770p.pdf \\
        --output data/parsed/docling_raw/printer_manual

    # Run with Marker
    python scripts/run_parser.py \\
        --parser marker \\
        --pdf  data/raw/manuals/01_HP_LaserJet_Repair_Manual_770p.pdf \\
        --output data/parsed/docling_raw/printer_manual_marker

Optional flags
    --parser         docling | marker  (default: docling)
    --no-tables      Skip table extraction
    --no-figures     Skip figure extraction
    --ocr            Enable OCR (for scanned/image-only pages)
    --log-level      DEBUG | INFO | WARNING | ERROR  (default: INFO)
    --log-file       Path to write log file
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="run_parser",
        description="TechDocs Copilot — Document Parser (Stage 1)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--parser",
        choices=["docling", "marker"],
        default="docling",
        help="Parser backend to use.",
    )
    parser.add_argument(
        "--pdf",
        type=Path,
        required=True,
        metavar="PDF_PATH",
        help="Path to the input PDF file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="OUTPUT_DIR",
        help="Root directory where parsed output will be written.",
    )
    parser.add_argument(
        "--no-tables",
        action="store_true",
        help="Disable table extraction.",
    )
    parser.add_argument(
        "--no-figures",
        action="store_true",
        help="Disable figure / image extraction.",
    )
    parser.add_argument(
        "--ocr",
        action="store_true",
        help="Enable OCR for scanned or image-only pages.",
    )
    parser.add_argument(
        "--figure-format",
        choices=["png", "jpg"],
        default="png",
        help="Image format for extracted figures.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Logging verbosity.",
    )
    parser.add_argument(
        "--log-file",
        type=Path,
        default=None,
        metavar="LOG_FILE",
        help="Optional path to write logs to a file.",
    )

    return parser.parse_args()


def main() -> None:
    # Add project root to sys.path so 'src' is importable regardless of
    # where the script is invoked from.
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    args = parse_args()

    # Import here (after sys.path is set)
    from src.parsing.config import ParserConfig
    from src.parsing.docling_parser import DoclingParser
    from src.parsing.marker_parser import MarkerParser
    from src.parsing.utils import setup_logger

    logger = setup_logger("run_parser", level=args.log_level, log_file=args.log_file)

    logger.info("=" * 60)
    logger.info("TechDocs Copilot — Document Parser (Stage 1)")
    logger.info("=" * 60)
    logger.info("Parser    : %s", args.parser.upper())
    logger.info("PDF       : %s", args.pdf)
    logger.info("Output    : %s", args.output)
    logger.info("Tables    : %s", not args.no_tables)
    logger.info("Figures   : %s", not args.no_figures)
    logger.info("OCR       : %s", args.ocr)
    logger.info("=" * 60)

    try:
        config = ParserConfig(
            pdf_path=args.pdf,
            output_dir=args.output,
            extract_tables=not args.no_tables,
            extract_figures=not args.no_figures,
            do_ocr=args.ocr,
            figure_format=args.figure_format,
            log_level=args.log_level,
            log_file=args.log_file,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    # Parser factory
    _PARSERS = {
        "docling": DoclingParser,
        "marker": MarkerParser,
    }
    parser_obj = _PARSERS[args.parser](config)

    start = time.time()
    try:
        result = parser_obj.parse(config.pdf_path)
    except Exception as exc:
        logger.exception("Parsing failed: %s", exc)
        sys.exit(1)

    try:
        parser_obj.save(result, config.output_dir)
    except Exception as exc:
        logger.exception("Saving output failed: %s", exc)
        sys.exit(1)

    elapsed = time.time() - start
    stats = result.document.statistics
    logger.info("=" * 60)
    logger.info("DONE in %.1f seconds.", elapsed)
    logger.info("Output written to: %s", config.output_dir)
    logger.info(
        "Summary: %d text blocks | %d tables | %d figures | %d pages",
        stats.text_blocks,
        stats.tables,
        stats.figures,
        result.metadata.total_pages,
    )
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
