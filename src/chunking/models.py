"""
models.py
=========
All Pydantic data models for the Stage 3 chunking pipeline.

Model hierarchy:
  NormalizedBlock   — a single block after normalizer processing
  PlannedChunk      — one chunk as decided by the Chunk Planner
  ChunkPlan         — the full planner output (intermediate, not written to disk)
  ValidationFlag    — a single quality issue raised by the Chunk Validator
  Chunk             — the final output unit written inside ChunkManifest
  ChunkManifest     — the top-level output file (chunks.json)
"""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# NormalizedBlock
# ---------------------------------------------------------------------------

class NormalizedBlock(BaseModel):
    """
    A single document block after the Normalizer has processed it.

    All fields mirror the parser_v1 schema; see SCHEMA.md for definitions.
    The normalizer adds two extra fields: excluded and exclusion_reason.
    """

    # ── Core identity ──────────────────────────────────────────────────────
    id: str
    type: str
    reading_order: int
    page: int

    # ── Text content (None for pure figure/table blocks) ──────────────────
    content: Optional[str] = None

    # ── Chunker guidance ──────────────────────────────────────────────────
    chunk_hint: Optional[str] = None
    semantic_role: Optional[str] = None

    # ── Section hierarchy ─────────────────────────────────────────────────
    section: Optional[str] = None
    section_id: Optional[str] = None
    parent_section_id: Optional[str] = None
    section_level: Optional[int] = None
    section_path: Optional[List[Dict[str, str]]] = None

    # ── Atomic grouping ───────────────────────────────────────────────────
    group_id: Optional[str] = None

    # ── Figure-specific ───────────────────────────────────────────────────
    image_path: Optional[str] = None
    caption: Optional[str] = None           # descriptive caption (number stripped)
    caption_id: Optional[str] = None        # ID of the linked caption block
    decorative: Optional[bool] = None
    width: Optional[int] = None
    height: Optional[int] = None
    image_hash: Optional[str] = None
    figure_number: Optional[str] = None
    previous_block_id: Optional[str] = None
    next_block_id: Optional[str] = None

    # ── Table-specific ────────────────────────────────────────────────────
    markdown: Optional[str] = None
    title: Optional[str] = None             # full caption text including number
    table_number: Optional[str] = None      # e.g. "Table 2-13"
    rows: Optional[int] = None
    cols: Optional[int] = None

    # ── Caption-specific ──────────────────────────────────────────────────
    caption_for: Optional[str] = None       # ID of the figure/table this describes

    # ── Admonition-specific ───────────────────────────────────────────────
    severity: Optional[str] = None          # warning/caution/note/tip/important
    level: Optional[int] = None             # heading depth (headings only)
    section_number: Optional[str] = None    # numeric prefix e.g. "5.2.1"

    # ── Provenance ────────────────────────────────────────────────────────
    source: Optional[str] = None            # pdf_text | ocr | vision
    bbox: Optional[Dict[str, float]] = None

    # ── Normalizer output ─────────────────────────────────────────────────
    excluded: bool = False
    exclusion_reason: Optional[str] = None

    # ── Internal normalizer annotations (not persisted to final output) ───
    # Used by the Chunk Planner; stripped in the Builder.
    _table_continuation_key: Optional[str] = None  # set for detected table continuations


# ---------------------------------------------------------------------------
# PlannedChunk  (Chunk Planner output — intermediate, not written to disk)
# ---------------------------------------------------------------------------

class PlannedChunk(BaseModel):
    """
    One chunk as planned by the Chunk Planner.
    This is an intermediate object; it is not written to disk.
    """

    chunk_id: str                               # "chunk_0001"
    block_ids: List[str]                        # IDs in reading order
    heading: Optional[str] = None              # nearest heading text for this chunk
    section_path: Optional[List[Dict[str, str]]] = None
    estimated_tokens: int = 0                  # word-count proxy
    reading_order_start: int = 0               # reading_order of first block
    reading_order_end: int = 0                 # reading_order of last block
    reason: str = ""                           # human-readable planning rationale
    rules_applied: List[str] = Field(          # e.g. ["R1.1", "R2.2", "R3.3"]
        default_factory=list
    )


class ChunkPlan(BaseModel):
    """Full output of the Chunk Planner. Passed to Validator and Builder."""

    document_id: str
    total_planned: int
    chunks: List[PlannedChunk]


# ---------------------------------------------------------------------------
# ValidationFlag  (Chunk Validator output)
# ---------------------------------------------------------------------------

FlagType = Literal[
    "too_small",
    "too_large",
    "no_heading",
    "orphan_caption",
    "table_integrity",
    "unknown_blocks",
]

FlagSeverity = Literal["warning", "error"]


class ValidationFlag(BaseModel):
    """A single quality issue raised by the Chunk Validator."""

    chunk_id: str
    flag_type: FlagType
    detail: str
    severity: FlagSeverity


# ---------------------------------------------------------------------------
# Chunk  (final output unit, lives inside ChunkManifest)
# ---------------------------------------------------------------------------

class Chunk(BaseModel):
    """
    The final output unit written to chunks.json.

    Contains everything needed for embedding, retrieval, and optional LLM review
    without requiring a round-trip to the original parser_v1 files.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    chunk_id: str
    document_id: str

    # ── Structural context ────────────────────────────────────────────────
    heading: Optional[str] = None
    section_path: Optional[List[Dict[str, str]]] = None
    pages: List[int] = Field(default_factory=list)       # sorted unique page numbers
    reading_order_start: int
    reading_order_end: int

    # ── Block inventory ───────────────────────────────────────────────────
    block_ids: List[str]
    blocks: List[Dict[str, Any]] = Field(                # full block objects
        default_factory=list,
        description=(
            "Full NormalizedBlock dicts in reading order. "
            "Included so downstream LLMs can reason over the chunk "
            "without reconstructing it from parser_v1 files."
        ),
    )

    # ── Content summary flags ─────────────────────────────────────────────
    has_figure: bool = False
    has_table: bool = False
    has_admonition: bool = False

    # ── Retrieval text ────────────────────────────────────────────────────
    text: str = Field(
        "",
        description=(
            "All text content concatenated in reading order. "
            "Figures render as '[Figure: <caption>]', tables inline as markdown."
        ),
    )
    figures: List[Dict[str, Any]] = Field(               # [{id, image_path, caption}]
        default_factory=list
    )
    tables: List[Dict[str, Any]] = Field(                # [{id, markdown, title}]
        default_factory=list
    )

    # ── Planning provenance ───────────────────────────────────────────────
    planning_reason: str = ""
    rules_applied: List[str] = Field(default_factory=list)  # e.g. ["R2.2", "R1.1"]

    # ── Quality score ─────────────────────────────────────────────────────
    token_estimate: int = 0
    chunk_score: float = Field(
        100.0,
        ge=0.0,
        le=100.0,
        description=(
            "Quality score 0–100. "
            "Deducted: 20 per error flag, 10 per warning flag. "
            "Chunks below config.score_threshold are candidates for LLM review."
        ),
    )
    flagged: bool = Field(
        False,
        description="Convenience alias: True when chunk_score < 100.",
    )
    validation_flags: List[ValidationFlag] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# ChunkManifest  (top-level output file: chunks.json)
# ---------------------------------------------------------------------------

class ChunkManifest(BaseModel):
    """
    Written once per document as chunks.json.
    Contains all chunks and a run-level summary.
    """

    document_id: str
    source_dir: str                             # parsed_dir path (string for JSON portability)
    chunked_at: str                             # UTC ISO-8601 timestamp
    config_snapshot: Dict[str, Any] = Field(   # ChunkerConfig values at run time
        default_factory=dict
    )

    # ── Run statistics ────────────────────────────────────────────────────
    total_chunks: int
    clean_chunks: int                           # chunk_score == 100
    flagged_chunks: int                         # chunk_score < 100
    total_blocks_processed: int
    total_blocks_excluded: int

    # ── Flag breakdown ────────────────────────────────────────────────────
    flag_summary: Dict[str, int] = Field(       # {"too_small": 8, "no_heading": 4, ...}
        default_factory=dict
    )

    # ── Chunks ────────────────────────────────────────────────────────────
    chunks: List[Chunk]
