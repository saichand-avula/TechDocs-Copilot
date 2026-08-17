"""
src/retrieval/config.py
────────────────────────
RetrieverConfig: all tunable knobs in one place.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from src.embedding.config import GEMINI_API_KEYS


@dataclass
class RetrieverConfig:
    # ── Sarvam (OpenAI-compatible) ─────────────────────────────────────────
    sarvam_api_key:  str  = field(default_factory=lambda: os.getenv("SARVAM_API_KEY", "").strip())
    sarvam_base_url: str  = "https://api.sarvam.ai/v1"
    sarvam_model:    str  = "sarvam-105b"

    # ── Gemini embedding (same 5 keys as Stage 4) ──────────────────────────
    gemini_api_keys:   list[str] = field(default_factory=lambda: list(GEMINI_API_KEYS))
    embedding_model:   str       = "gemini-embedding-2"
    embedding_dims:    int       = 1536

    # ── Paths ──────────────────────────────────────────────────────────────
    vectordb_dir:    Path = Path("data/vectordb")
    collection_name: str  = "techdocs_chunks"
    bm25_dir:        Path = Path("data/vectordb/bm25")
    chunks_dir:      Path = Path("data/chunks")

    # ── Candidate counts ───────────────────────────────────────────────────
    dense_candidates: int = 15    # Chroma top-N before RRF
    bm25_candidates:  int = 15    # BM25 top-N before RRF
    rrf_k:            int = 60    # RRF constant
    rrf_candidates:   int = 8     # after RRF, sent to reranker

    # ── Reranker + threshold ───────────────────────────────────────────────
    generator_top_k:      int   = 5    # max chunks to generator after threshold
    relevance_threshold:  float = 0.5  # score ≥ 5/10 → pass (calibrate on golden set)

    # ── HyDE ──────────────────────────────────────────────────────────────
    use_hyde:        bool = True   # config-switchable for A/B evaluation
    hyde_max_tokens: int  = 150

    def __post_init__(self) -> None:
        if not self.sarvam_api_key:
            raise ValueError("Missing SARVAM_API_KEY. Export it before running retrieval.")
        if not self.gemini_api_keys:
            raise ValueError("Missing GEMINI_API_KEYS. Set a comma-separated key list in env.")
        self.vectordb_dir = Path(self.vectordb_dir)
        self.bm25_dir     = Path(self.bm25_dir)
        self.chunks_dir   = Path(self.chunks_dir)
