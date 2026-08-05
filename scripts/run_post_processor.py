"""
run_post_processor.py
=====================
CLI entry point for the TechDocs Copilot semantic post-processor.

Workflow
--------
Stage 1 (parser):        run_parser.py  → data/parsed/docling_raw/<doc>
Stage 2 (post-process):  run_post_processor.py  → data/parsed/parser_v1/<doc>

The post-processor reads from whichever directory you point it at and
enriches the page JSON files in-place.  Copy the raw parser output to
parser_v1/ first so the raw snapshot stays untouched:

    cp -R data/parsed/docling_raw/printer_manual \
          data/parsed/parser_v1/printer_manual

    python scripts/run_post_processor.py \
        --output data/parsed/parser_v1/printer_manual

    # Enrich multiple documents at once
    python scripts/run_post_processor.py \
        --output data/parsed/parser_v1/printer_manual \
                 data/parsed/parser_v1/another_manual
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.parsing.post_processor import PostProcessor
    from src.parsing.utils import setup_logger

    parser = argparse.ArgumentParser(
        prog="run_post_processor",
        description="TechDocs Copilot — Semantic Post-Processor",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--output",
        nargs="+",
        type=Path,
        required=True,
        metavar="OUTPUT_DIR",
        help="One or more parsed output directories to enrich.",
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logger = setup_logger("run_post_processor", level=args.log_level)
    pp = PostProcessor(logger=logger)

    for output_dir in args.output:
        logger.info("=" * 60)
        logger.info("Enriching: %s", output_dir)
        logger.info("=" * 60)
        try:
            stats = pp.enrich(output_dir)
            logger.info("Results:")
            for k, v in stats.items():
                logger.info("  %-25s %d", k, v)
        except Exception as exc:
            logger.exception("Post-processing failed for %s: %s", output_dir, exc)
            sys.exit(1)

    logger.info("=" * 60)
    logger.info("Post-processing complete.")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
