"""
parser_interface.py
====================
Abstract base class and shared Pydantic data models for all parsers.

Schema v2.1 — Final Frozen Design
----------------------------------
Key principles:
  - PageBlock carries ONLY the fields relevant to its type (no nulls for other types)
  - document_id and page live at PageOutput level, not repeated in every block
  - Single field "section" for parent heading context (no parent_heading / parent_section split)
  - Figures carry previous_text_id / next_text_id for caption-free retrieval context
  - Serialize page files with exclude_none=True → clean, minimal JSON
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Dict, List, Literal, Optional

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Block type literals
# ---------------------------------------------------------------------------

BlockType = Literal[
    "paragraph",
    "heading",
    "list_item",
    "caption",
    "admonition",
    "footnote",
    "page_header",
    "page_footer",
    "code",
    "unknown",
]


# ---------------------------------------------------------------------------
# TextBlock — enriched
# ---------------------------------------------------------------------------

class TextBlock(BaseModel):
    """A single contiguous text element extracted from the document."""

    id: str
    type: BlockType
    page: int = Field(..., ge=1)
    section: Optional[str] = None       # nearest ancestor heading text
    content: str
    block_id: Optional[str] = None      # internal parser reference
    reading_order: Optional[int] = None
    level: Optional[int] = None         # heading depth: 1=title, 2=section… (headings only)
    bbox: Optional[Dict[str, float]] = None  # {l, t, r, b} in page coordinates


# ---------------------------------------------------------------------------
# TableBlock — enriched
# ---------------------------------------------------------------------------

class TableBlock(BaseModel):
    """A table extracted from the document."""

    id: str
    page: int = Field(..., ge=1)
    section: Optional[str] = None
    title: Optional[str] = None
    markdown: str
    block_id: Optional[str] = None
    reading_order: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None


# ---------------------------------------------------------------------------
# FigureBlock — enriched
# ---------------------------------------------------------------------------

class FigureBlock(BaseModel):
    """A figure / image extracted from the document."""

    id: str
    page: int = Field(..., ge=1)
    section: Optional[str] = None
    caption: Optional[str] = None       # null if absent — never invented
    path: str                           # relative path to saved image file
    block_id: Optional[str] = None
    reading_order: Optional[int] = None
    bbox: Optional[Dict[str, float]] = None
    width: Optional[int] = None         # pixels
    height: Optional[int] = None        # pixels
    image_path: Optional[str] = None    # explicit alias for path
    image_hash: Optional[str] = None    # SHA-256 for dedup detection
    previous_block_id: Optional[str] = None  # nearest preceding block id (any type)
    next_block_id: Optional[str] = None      # nearest following block id (any type)


# ---------------------------------------------------------------------------
# PageBlock — clean typed block entry (serialize with exclude_none=True)
# ---------------------------------------------------------------------------

class PageBlock(BaseModel):
    """
    One entry in a page's reading-order block list.

    Each block carries ONLY fields relevant to its type.
    Serialize with model_dump(exclude_none=True) to produce clean JSON with no nulls.

    Text block:   {type, id, reading_order, section?, content, level?, role?, bbox?}
    Admonition:   {type, id, reading_order, section?, content, severity}
    Caption:      {type, id, reading_order, section?, content, caption_for?}
    Figure block: {type, id, reading_order, section?, image_path,
                   caption?, caption_id?, bbox?, width?, height?, image_hash?,
                   previous_text_id?, next_text_id?}
    Table block:  {type, id, reading_order, section?, markdown, bbox?, caption_id?}
    """

    type: str
    id: str
    reading_order: int

    # ── Shared optional ────────────────────────────────────────────────────
    section: Optional[str] = None
    bbox: Optional[Dict[str, float]] = None

    # ── Text / heading / list / admonition / caption ─────────────────────
    content: Optional[str] = None
    level: Optional[int] = None            # heading depth (headings only)
    role: Optional[str] = None             # heading: chapter/section/subsection
                                           # list_item: procedure_step
                                           # admonition: warning/note/caution/tip/important
    severity: Optional[str] = None         # admonitions: warning/caution/note/tip/important
    caption_for: Optional[str] = None      # caption blocks: which figure/table id they describe

    # ── Figure block ─────────────────────────────────────────────────────
    image_path: Optional[str] = None
    caption: Optional[str] = None
    caption_id: Optional[str] = None       # figure/table: which caption block refers to them
    width: Optional[int] = None
    height: Optional[int] = None
    image_hash: Optional[str] = None
    previous_block_id: Optional[str] = None  # nearest preceding block (any type)
    next_block_id: Optional[str] = None      # nearest following block (any type)

    # ── Table block ──────────────────────────────────────────────────────
    markdown: Optional[str] = None


# ---------------------------------------------------------------------------
# PageOutput — clean page container
# ---------------------------------------------------------------------------

class PageOutput(BaseModel):
    """
    Per-page output. document_id and page live here, not in every block.
    Blocks are in reading order; navigate by index or reading_order value.
    """

    page: int = Field(..., ge=1)
    document_id: str
    page_bbox: Optional[Dict[str, float]] = Field(
        None, description="Page size {width, height} in points. Handles mixed-size PDFs."
    )
    blocks: List[PageBlock] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# DocumentIndex — lightweight document.json
# ---------------------------------------------------------------------------

class DocumentStatistics(BaseModel):
    text_blocks: int = 0
    tables: int = 0
    figures: int = 0


class PageIndexEntry(BaseModel):
    page: int
    path: str


class DocumentIndex(BaseModel):
    """
    Lightweight document-level index.
    Does NOT duplicate paragraph text — use page files for full content.
    """

    document_id: str
    title: str
    pages: int
    statistics: DocumentStatistics = Field(default_factory=DocumentStatistics)
    page_index: List[PageIndexEntry] = Field(default_factory=list)


# Backward compat alias
DocumentOutput = DocumentIndex


# ---------------------------------------------------------------------------
# DocumentMetadata
# ---------------------------------------------------------------------------

class DocumentMetadata(BaseModel):
    """Parser-level metadata stored as metadata.json."""

    document_id: str
    filename: str
    title: Optional[str] = None
    total_pages: int
    parser: str
    parser_version: Optional[str] = None
    parsed_at: str
    processing_time_seconds: Optional[float] = None


# ---------------------------------------------------------------------------
# ParsedDocument — top-level container returned by parse()
# ---------------------------------------------------------------------------

class ParsedDocument(BaseModel):
    """
    Complete parse result.

    - metadata     → saved as metadata.json
    - document     → saved as document.json (lightweight index)
    - page_outputs → saved as pages/page_XXX.json (clean ordered blocks)
    - table_blocks → saved as tables/tbl_XXXX.json
    - figure_blocks→ saved in figures/
    """

    metadata: DocumentMetadata
    document: DocumentIndex
    page_outputs: List[PageOutput] = Field(default_factory=list)
    table_blocks: List[TableBlock] = Field(default_factory=list)
    figure_blocks: List[FigureBlock] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# Abstract Parser interface
# ---------------------------------------------------------------------------

class Parser(ABC):
    """
    Abstract base class for all document parsers.

    Output directory contract:
        output_dir/
            document.json           ← lightweight index
            metadata.json           ← parser info + timing
            pages/
                page_001.json       ← clean ordered blocks (no nulls)
                page_002.json
                ...
            tables/
                tbl_0001.json       ← full table with markdown + bbox
                ...
            figures/
                figure_metadata.json ← enriched figure metadata
                figure_0001.png
                ...
    """

    @abstractmethod
    def parse(self, pdf_path: Path) -> ParsedDocument: ...

    @abstractmethod
    def save(self, result: ParsedDocument, output_dir: Path) -> None: ...
