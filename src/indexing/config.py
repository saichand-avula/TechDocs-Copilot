"""
src/indexing/config.py
──────────────────────
Configuration for the indexing pipeline.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from src.embedding.config import GEMINI_API_KEYS


@dataclass
class IndexConfig:
    # ── Paths ──────────────────────────────────────────────────────────────
    chunks_dir: Path = Path("data/chunks")
    embeddings_dir: Path = Path("data/embeddings")
    vectordb_dir: Path = Path("data/vectordb")

    # ── ChromaDB ───────────────────────────────────────────────────────────
    collection_name: str = "techdocs_chunks"
    embedding_dims: int = 1536

    # ── BM25 ───────────────────────────────────────────────────────────────
    bm25_dir: Path = Path("data/vectordb/bm25")

    # ── Retrieval (used by retriever, not indexer itself) ──────────────────
    dense_candidates: int = 15   # Dense top-K before RRF
    bm25_candidates: int = 15    # BM25 top-K before RRF
    rrf_k: int = 60              # RRF constant
    final_top_k: int = 5         # Final results after fusion

    # ── Query-side API key rotation (same 5 keys) ──────────────────────────
    api_keys: list[str] = None

    def __post_init__(self) -> None:
        if self.api_keys is None:
            self.api_keys = list(GEMINI_API_KEYS)
        # Ensure dirs are Path objects
        self.chunks_dir = Path(self.chunks_dir)
        self.embeddings_dir = Path(self.embeddings_dir)
        self.vectordb_dir = Path(self.vectordb_dir)
        self.bm25_dir = Path(self.bm25_dir)
