#!/usr/bin/env python3
"""
scripts/run_embedder.py
────────────────────────
CLI entry point for Stage 4 embedding.

Usage:
    python scripts/run_embedder.py
    python scripts/run_embedder.py --manual printer_manual
    python scripts/run_embedder.py --batch-size 100
    python scripts/run_embedder.py --delay 2.0

Output: data/embeddings/{manual_name}/
    embeddings.npy      – float32 (N_chunks × 1536)
    chunk_ids.json      – ordered chunk_id list
    manual_meta.json    – run metadata
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.embedding.config import EmbeddingConfig
from src.embedding.embedder import SemanticEmbedder


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, datefmt=datefmt)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Generate embeddings for all TechDocs manuals using gemini-embedding-2",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--manual",
        default=None,
        help="Embed a single manual by name (default: all manuals)",
    )
    p.add_argument(
        "--chunks-dir",
        default="data/chunks",
        help="Path to chunks directory (default: data/chunks)",
    )
    p.add_argument(
        "--output-dir",
        default="data/embeddings",
        help="Output directory for embeddings (default: data/embeddings)",
    )
    p.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Initial batch size (default: 50, ramps to 100 on success)",
    )
    p.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="Inter-batch delay in seconds (default: 1.0)",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    log = logging.getLogger("run_embedder")

    cfg = EmbeddingConfig(
        chunks_dir=Path(args.chunks_dir),
        embeddings_dir=Path(args.output_dir),
        initial_batch_size=args.batch_size,
        inter_batch_delay_s=args.delay,
    )

    embedder = SemanticEmbedder(cfg)
    start = time.perf_counter()

    if args.manual:
        # Single manual mode
        manifest_path = Path(args.chunks_dir) / args.manual / "chunks.json"
        if not manifest_path.exists():
            log.error("chunks.json not found: %s", manifest_path)
            sys.exit(1)
        out_dir = Path(args.output_dir) / args.manual
        n = embedder.embed_manual(manifest_path, out_dir, args.manual)
        summary = {args.manual: n}
    else:
        # All manuals
        summary = embedder.embed_all()

    elapsed = time.perf_counter() - start
    total_chunks = sum(summary.values())

    print()
    print("=" * 56)
    print("  Embedding complete")
    print("=" * 56)
    for manual, n in sorted(summary.items()):
        print(f"  {manual:<35} {n:>5} chunks")
    print("-" * 56)
    print(f"  Total chunks embedded : {total_chunks}")
    print(f"  Elapsed               : {elapsed:.1f}s")
    print(f"  Output                : {args.output_dir}/")
    print("=" * 56)


if __name__ == "__main__":
    main()
