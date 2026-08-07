"""
chunk_builder.py
================
Stage 3 — Step 4: Chunk Builder

Responsibility:
  Assemble the final list[Chunk] from:
    - ChunkPlan (planned structure from ChunkPlanner)
    - flags_by_chunk + scores_by_chunk (from ChunkValidator)
    - NormalizedBlocks (for content)
  Then write the ChunkManifest to disk as chunks.json.

The Builder is purely deterministic — no LLM calls, no randomness.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import ChunkerConfig
from .models import (
    Chunk,
    ChunkManifest,
    ChunkPlan,
    NormalizedBlock,
    PlannedChunk,
    ValidationFlag,
)
from ..parsing.utils import utc_now_iso


class ChunkBuilder:
    """
    Assembles Chunk objects from the plan + validator output and writes
    the ChunkManifest to disk.

    Usage::

        builder = ChunkBuilder(config=config, logger=logger)
        manifest = builder.build(blocks, plan, flags_by_chunk, scores_by_chunk, output_dir)
    """

    def __init__(
        self,
        config: ChunkerConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def build(
        self,
        blocks: List[NormalizedBlock],
        plan: ChunkPlan,
        flags_by_chunk: Dict[str, List[ValidationFlag]],
        scores_by_chunk: Dict[str, float],
        output_dir: Path,
        document_id: str = "",
    ) -> ChunkManifest:
        """
        Assemble and persist the chunk manifest.

        Parameters
        ----------
        blocks : list[NormalizedBlock]
            All blocks (including excluded) from the Normalizer.
        plan : ChunkPlan
            Output of the ChunkPlanner.
        flags_by_chunk : dict[chunk_id → list[ValidationFlag]]
            Output of the ChunkValidator.
        scores_by_chunk : dict[chunk_id → float]
            Quality scores from the ChunkValidator.
        output_dir : Path
            Directory to write chunks.json into.

        Returns
        -------
        ChunkManifest
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        block_map: Dict[str, NormalizedBlock] = {b.id: b for b in blocks}

        # --- Build Chunk objects ---
        chunks: List[Chunk] = []
        for planned in plan.chunks:
            chunk = self._assemble_chunk(
                planned=planned,
                block_map=block_map,
                flags=flags_by_chunk.get(planned.chunk_id, []),
                score=scores_by_chunk.get(planned.chunk_id, 100.0),
                document_id=document_id or plan.document_id,
            )
            chunks.append(chunk)

        # --- Statistics ---
        total = len(chunks)
        clean = sum(1 for c in chunks if c.chunk_score >= 100.0)
        flagged = total - clean
        excluded_count = sum(1 for b in blocks if b.excluded)

        flag_summary: Dict[str, int] = {}
        for flag_list in flags_by_chunk.values():
            for flag in flag_list:
                flag_summary[flag.flag_type] = flag_summary.get(flag.flag_type, 0) + 1

        # --- Config snapshot (for reproducibility) ---
        config_snapshot = {
            "max_tokens": self._config.max_tokens,
            "short_sibling_threshold": self._config.short_sibling_threshold,
            "min_chunk_tokens": self._config.min_chunk_tokens,
            "max_chunk_tokens": self._config.max_chunk_tokens,
            "score_threshold": self._config.score_threshold,
        }

        manifest = ChunkManifest(
            document_id=plan.document_id,
            source_dir=str(self._config.parsed_dir),
            chunked_at=utc_now_iso(),
            config_snapshot=config_snapshot,
            total_chunks=total,
            clean_chunks=clean,
            flagged_chunks=flagged,
            total_blocks_processed=sum(1 for b in blocks if not b.excluded),
            total_blocks_excluded=excluded_count,
            flag_summary=flag_summary,
            chunks=chunks,
        )

        # --- Write output ---
        out_path = output_dir / "chunks.json"
        self._write_manifest(manifest, out_path)
        self._log.info(
            "ChunkBuilder: wrote %d chunks to %s (%d clean, %d flagged)",
            total,
            out_path,
            clean,
            flagged,
        )

        return manifest

    # ------------------------------------------------------------------
    # Chunk assembly
    # ------------------------------------------------------------------

    def _assemble_chunk(
        self,
        planned: PlannedChunk,
        block_map: Dict[str, NormalizedBlock],
        flags: List[ValidationFlag],
        score: float,
        document_id: str = "",
    ) -> Chunk:
        """Assemble a single Chunk from a PlannedChunk and its block data."""
        chunk_blocks: List[NormalizedBlock] = [
            block_map[bid] for bid in planned.block_ids if bid in block_map
        ]

        # Content assembly
        text_parts: List[str] = []
        figures: List[Dict[str, Any]] = []
        tables: List[Dict[str, Any]] = []
        has_figure = False
        has_table = False
        has_admonition = False
        pages: List[int] = []

        for block in chunk_blocks:
            pages.append(block.page)

            if block.type == "figure":
                has_figure = True
                cap_text = block.caption or ""
                text_parts.append(f"[Figure: {cap_text}]" if cap_text else "[Figure]")
                figures.append({
                    "id": block.id,
                    "image_path": block.image_path,
                    "caption": block.caption,
                    "figure_number": block.figure_number,
                    "page": block.page,
                })

            elif block.type == "table":
                has_table = True
                if block.markdown:
                    text_parts.append(block.markdown)
                tables.append({
                    "id": block.id,
                    "markdown": block.markdown,
                    "title": block.title,
                    "table_number": block.table_number,
                    "rows": block.rows,
                    "cols": block.cols,
                    "page": block.page,
                })

            elif block.type == "admonition" or block.semantic_role in {
                "warning", "caution", "note", "tip", "important"
            }:
                has_admonition = True
                if block.content:
                    label = (block.severity or block.semantic_role or "NOTE").upper()
                    text_parts.append(f"[{label}] {block.content}")

            elif block.content:
                text_parts.append(block.content)

        text = "\n\n".join(p for p in text_parts if p.strip())
        unique_pages = sorted(set(pages))

        # Serialize blocks for downstream use (exclude internal fields)
        serialized_blocks = [
            self._serialize_block(b) for b in chunk_blocks
        ]

        return Chunk(
            chunk_id=planned.chunk_id,
            document_id=document_id,
            heading=planned.heading,
            section_path=planned.section_path,
            pages=unique_pages,
            reading_order_start=planned.reading_order_start,
            reading_order_end=planned.reading_order_end,
            block_ids=planned.block_ids,
            blocks=serialized_blocks,
            has_figure=has_figure,
            has_table=has_table,
            has_admonition=has_admonition,
            text=text,
            figures=figures,
            tables=tables,
            planning_reason=planned.reason,
            rules_applied=planned.rules_applied,
            token_estimate=planned.estimated_tokens,
            chunk_score=score,
            flagged=(score < 100.0),
            validation_flags=flags,
        )

    def _serialize_block(self, block: NormalizedBlock) -> Dict[str, Any]:
        """
        Serialize a NormalizedBlock to a dict for inclusion in the Chunk.
        Excludes internal normalizer fields (excluded, exclusion_reason,
        _table_continuation_key).
        """
        d = block.model_dump(exclude_none=True)
        # Remove normalizer-internal fields
        d.pop("excluded", None)
        d.pop("exclusion_reason", None)
        # Private attributes are not in model_dump, so no need to remove _table_continuation_key
        return d

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def _write_manifest(self, manifest: ChunkManifest, path: Path) -> None:
        """Write the ChunkManifest to a JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = manifest.model_dump(mode="json")
        with path.open("w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2, ensure_ascii=False)
        size_kb = path.stat().st_size / 1024
        self._log.info("ChunkBuilder: manifest written to %s (%.1f KB)", path, size_kb)
