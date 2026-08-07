"""
run_chunker.py
==============
CLI entry point for the TechDocs Copilot Stage 3 Semantic Chunker.

Workflow
--------
Stage 1 (parse):        run_parser.py       → data/parsed/docling_raw/<doc>
Stage 2 (post-process): run_post_processor.py → data/parsed/parser_v1/<doc>
Stage 3 (chunk):        run_chunker.py      → data/chunks/<doc>

Usage
-----
    python scripts/run_chunker.py \\
        --input  data/parsed/parser_v1/printer_manual \\
        --output data/chunks/printer_manual

    # With custom thresholds
    python scripts/run_chunker.py \\
        --input  data/parsed/parser_v1/printer_manual \\
        --output data/chunks/printer_manual \\
        --max-tokens 512 \\
        --short-sibling-threshold 150 \\
        --log-level DEBUG
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> None:
    project_root = Path(__file__).parent.parent
    sys.path.insert(0, str(project_root))

    from src.chunking.chunker import SemanticChunker
    from src.chunking.config import ChunkerConfig
    from src.parsing.utils import setup_logger

    parser = argparse.ArgumentParser(
        prog="run_chunker",
        description="TechDocs Copilot — Stage 3: Semantic Chunker",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--input",
        type=Path,
        required=True,
        metavar="PARSED_DIR",
        help="Path to a parser_v1/<doc> directory (must contain document.json).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        metavar="OUTPUT_DIR",
        help="Directory where chunks.json will be written.",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        metavar="N",
        help="Overflow threshold: chunks projected to exceed N words are split (R4.1).",
    )
    parser.add_argument(
        "--short-sibling-threshold",
        type=int,
        default=150,
        metavar="N",
        help=(
            "Short-sibling merge threshold: adjacent H2 sections both under N words "
            "are merged into one chunk (R3.3)."
        ),
    )
    parser.add_argument(
        "--min-chunk-tokens",
        type=int,
        default=20,
        metavar="N",
        help="Validator: flag chunks with fewer than N words as 'too_small'.",
    )
    parser.add_argument(
        "--max-chunk-tokens",
        type=int,
        default=800,
        metavar="N",
        help="Validator: flag chunks with more than N words as 'too_large'.",
    )
    parser.add_argument(
        "--score-threshold",
        type=float,
        default=70.0,
        metavar="SCORE",
        help=(
            "Chunks with chunk_score below this are candidates for LLM review. "
            "Range: 0–100."
        ),
    )
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
    )
    args = parser.parse_args()

    logger = setup_logger("run_chunker", level=args.log_level)

    try:
        config = ChunkerConfig(
            parsed_dir=args.input,
            output_dir=args.output,
            max_tokens=args.max_tokens,
            short_sibling_threshold=args.short_sibling_threshold,
            min_chunk_tokens=args.min_chunk_tokens,
            max_chunk_tokens=args.max_chunk_tokens,
            score_threshold=args.score_threshold,
            log_level=args.log_level,
        )
    except Exception as exc:
        logger.error("Configuration error: %s", exc)
        sys.exit(1)

    chunker = SemanticChunker(config=config, logger=logger)

    try:
        manifest = chunker.run()
    except Exception as exc:
        logger.exception("Chunker failed: %s", exc)
        sys.exit(1)

    # Print a quick summary to stdout
    print()
    print("=" * 50)
    print("  Chunking complete")
    print("=" * 50)
    print(f"  Document ID    : {manifest.document_id}")
    print(f"  Total chunks   : {manifest.total_chunks}")
    print(f"  Clean          : {manifest.clean_chunks}")
    print(f"  Flagged        : {manifest.flagged_chunks}")
    print(f"  Blocks used    : {manifest.total_blocks_processed}")
    print(f"  Blocks excluded: {manifest.total_blocks_excluded}")
    if manifest.flag_summary:
        print()
        print("  Flag breakdown:")
        for flag_type, count in sorted(manifest.flag_summary.items()):
            print(f"    {flag_type:<25} {count}")
    print("=" * 50)
    out_path = args.output / "chunks.json"
    print(f"  Output: {out_path}")
    print("=" * 50)


if __name__ == "__main__":
    main()
