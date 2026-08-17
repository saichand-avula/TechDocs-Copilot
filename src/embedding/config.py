"""
src/embedding/config.py
───────────────────────
Centralized configuration for the embedding pipeline.

API keys are stored here for V1. Move to environment variables / secret
manager before any production deployment.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path


def _load_gemini_api_keys() -> list[str]:
    """Load comma-separated keys from GEMINI_API_KEYS env var."""
    raw = os.getenv("GEMINI_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


GEMINI_API_KEYS: list[str] = _load_gemini_api_keys()


@dataclass
class EmbeddingConfig:
    # ── Model ──────────────────────────────────────────────────────────────
    model: str = "gemini-embedding-2"
    output_dimensionality: int = 1536          # MRL truncation from 3072

    # ── Batching ───────────────────────────────────────────────────────────
    initial_batch_size: int = 50               # Chunks per API call
    max_batch_size: int = 50                   # Keep fixed at 50 (RPM budget controls rate)
    batch_size_ramp_after: int = 999           # Effectively disabled — RPM budget handles it

    # ── Rate-limit / retry ─────────────────────────────────────────────────
    inter_batch_delay_s: float = 0.5           # Small polite delay; RPM budget does the real work
    max_retries: int = 5                       # Per-batch retry attempts
    retry_base_delay_s: float = 35.0           # Minimum wait on 429 (free-tier window is ~31s)
    retry_max_delay_s: float = 120.0           # Cap on wait time

    # ── Keys ───────────────────────────────────────────────────────────────
    api_keys: list[str] = field(default_factory=lambda: list(GEMINI_API_KEYS))

    # ── Paths ──────────────────────────────────────────────────────────────
    chunks_dir: Path = Path("data/chunks")
    embeddings_dir: Path = Path("data/embeddings")

    # ── Text format (asymmetric retrieval — confirmed in Gemini docs) ──────
    # Documents: "title: {heading} | text: {text}"
    # Queries  : "task: search result | query: {text}"
    doc_prefix_template: str = "title: {heading} | text: {text}"

    # ── Token budget ───────────────────────────────────────────────────────
    # gemini-embedding-2 limit = 8192 tokens ≈ ~6000 words
    # Truncate assembled text if it exceeds this
    max_text_words: int = 5500

    def __post_init__(self) -> None:
        if not self.api_keys:
            raise ValueError("Missing GEMINI_API_KEYS. Set a comma-separated key list in env.")
