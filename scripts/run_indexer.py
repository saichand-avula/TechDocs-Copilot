#!/usr/bin/env python3
"""
scripts/run_indexer.py
───────────────────────
CLI entry point for Stage 4 indexing.
Reads pre-generated embeddings (from run_embedder.py) and builds:
  1. ChromaDB vector collection  → data/vectordb/chroma/
  2. BM25 sparse index           → data/vectordb/bm25/

Precondition: run_embedder.py must have completed first.

Usage:
    python scripts/run_indexer.py
    python scripts/run_indexer.py --chunks-dir data/chunks
    python scripts/run_indexer.py --log-level DEBUG
"""
from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.indexing.config import IndexConfig
from src.indexing.indexer import Indexer


def _setup_logging(level: str = "INFO") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt, datefmt=datefmt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Build ChromaDB + BM25 index from pre-generated embeddings",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--chunks-dir",     default="data/chunks",     help="Chunks directory")
    p.add_argument("--embeddings-dir", default="data/embeddings", help="Embeddings directory")
    p.add_argument("--vectordb-dir",   default="data/vectordb",   help="Output vectordb directory")
    p.add_argument("--collection",     default="techdocs_chunks", help="ChromaDB collection name")
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)
    log = logging.getLogger("run_indexer")

    cfg = IndexConfig(
        chunks_dir=Path(args.chunks_dir),
        embeddings_dir=Path(args.embeddings_dir),
        vectordb_dir=Path(args.vectordb_dir),
        collection_name=args.collection,
    )

    # Sanity check: embeddings must exist
    emb_dir = Path(args.embeddings_dir)
    if not emb_dir.exists():
        log.error(
            "Embeddings directory not found: %s\n"
            "Run 'python scripts/run_embedder.py' first.",
            emb_dir,
        )
        sys.exit(1)

    emb_dirs = [d for d in emb_dir.iterdir() if d.is_dir() and (d / "embeddings.npy").exists()]
    if not emb_dirs:
        log.error(
            "No embedding files found in %s.\n"
            "Run 'python scripts/run_embedder.py' first.",
            emb_dir,
        )
        sys.exit(1)

    log.info("Found %d manual embeddings to index", len(emb_dirs))

    start = time.perf_counter()
    indexer = Indexer(cfg)
    summary = indexer.build()
    elapsed = time.perf_counter() - start

    print()
    print("=" * 56)
    print("  Indexing complete")
    print("=" * 56)
    print("  ChromaDB:")
    for manual, n in sorted(summary["chroma"].items()):
        print(f"    {manual:<33} {n:>5} vectors")
    print(f"    {'TOTAL':<33} {summary['chroma_total']:>5}")
    print()
    print(f"  BM25 documents  : {summary['bm25_docs']}")
    print(f"  Elapsed         : {elapsed:.1f}s")
    print(f"  VectorDB dir    : {args.vectordb_dir}/")
    print("=" * 56)
    print()
    print("  Next: use src/indexing/retriever.py for hybrid search")
    print("=" * 56)


if __name__ == "__main__":
    main()
