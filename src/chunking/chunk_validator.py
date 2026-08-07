"""
chunk_validator.py
==================
Stage 3 — Step 3: Chunk Validator

Responsibility:
  Inspect the ChunkPlan (produced by ChunkPlanner) and produce a list of
  ValidationFlags. Assign a chunk_score to each planned chunk.

  The Validator does NOT modify chunks or the plan.
  It only reads, inspects, and reports.

Scoring formula:
  chunk_score starts at 100.0
  - 20 points per "error" flag
  - 10 points per "warning" flag
  Clamped to [0.0, 100.0].

  Chunks with chunk_score < config.score_threshold are candidates for LLM review.
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from .config import ChunkerConfig
from .models import ChunkPlan, NormalizedBlock, PlannedChunk, ValidationFlag


# Scoring weights
_ERROR_PENALTY = 20.0
_WARNING_PENALTY = 10.0


class ChunkValidator:
    """
    Validates a ChunkPlan and produces per-chunk quality flags and scores.

    Usage::

        validator = ChunkValidator(config=config, logger=logger)
        flags_by_chunk, scores_by_chunk = validator.validate(plan, blocks)
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

    def validate(
        self,
        plan: ChunkPlan,
        blocks: List[NormalizedBlock],
    ) -> Tuple[Dict[str, List[ValidationFlag]], Dict[str, float]]:
        """
        Run all validation checks against the ChunkPlan.

        Parameters
        ----------
        plan : ChunkPlan
            Output of the ChunkPlanner.
        blocks : list[NormalizedBlock]
            All blocks (including excluded) for cross-reference checks.

        Returns
        -------
        flags_by_chunk : dict[chunk_id → list[ValidationFlag]]
        scores_by_chunk : dict[chunk_id → float]  (0.0–100.0)
        """
        block_map: Dict[str, NormalizedBlock] = {b.id: b for b in blocks}

        # Build a reverse map: block_id → chunk_id
        block_to_chunk: Dict[str, str] = {}
        for chunk in plan.chunks:
            for bid in chunk.block_ids:
                block_to_chunk[bid] = chunk.chunk_id

        flags_by_chunk: Dict[str, List[ValidationFlag]] = defaultdict(list)

        for chunk in plan.chunks:
            chunk_blocks = [block_map[bid] for bid in chunk.block_ids if bid in block_map]

            self._check_too_small(chunk, flags_by_chunk)
            self._check_too_large(chunk, flags_by_chunk)
            self._check_no_heading(chunk, flags_by_chunk)
            self._check_orphan_caption(chunk, chunk_blocks, block_to_chunk, flags_by_chunk)
            self._check_table_integrity(chunk, chunk_blocks, block_to_chunk, flags_by_chunk)
            self._check_unknown_blocks(chunk, chunk_blocks, flags_by_chunk)
            # sec_auto_heading check removed: validated against real output, only produced
            # 3 flags on 770 pages, all for valid headings. No actionable signal.

        # Compute scores
        scores_by_chunk: Dict[str, float] = {}
        for chunk in plan.chunks:
            chunk_flags = flags_by_chunk.get(chunk.chunk_id, [])
            score = 100.0
            for flag in chunk_flags:
                if flag.severity == "error":
                    score -= _ERROR_PENALTY
                else:
                    score -= _WARNING_PENALTY
            scores_by_chunk[chunk.chunk_id] = max(0.0, min(100.0, score))

        # Log summary
        self._log_summary(plan, flags_by_chunk, scores_by_chunk)

        return dict(flags_by_chunk), scores_by_chunk

    # ------------------------------------------------------------------
    # Individual checks
    # ------------------------------------------------------------------

    def _check_too_small(
        self,
        chunk: PlannedChunk,
        flags: Dict[str, List[ValidationFlag]],
    ) -> None:
        """Flag chunks whose token estimate is suspiciously low."""
        if chunk.estimated_tokens < self._config.min_chunk_tokens:
            flags[chunk.chunk_id].append(
                ValidationFlag(
                    chunk_id=chunk.chunk_id,
                    flag_type="too_small",
                    detail=(
                        f"Chunk has only {chunk.estimated_tokens} tokens "
                        f"(threshold: {self._config.min_chunk_tokens}). "
                        f"Blocks: {chunk.block_ids}"
                    ),
                    severity="warning",
                )
            )

    def _check_too_large(
        self,
        chunk: PlannedChunk,
        flags: Dict[str, List[ValidationFlag]],
    ) -> None:
        """Flag chunks that exceed the max_chunk_tokens threshold."""
        if chunk.estimated_tokens > self._config.max_chunk_tokens:
            flags[chunk.chunk_id].append(
                ValidationFlag(
                    chunk_id=chunk.chunk_id,
                    flag_type="too_large",
                    detail=(
                        f"Chunk has {chunk.estimated_tokens} tokens "
                        f"(max: {self._config.max_chunk_tokens}). "
                        f"Could not be split at H3 boundaries or P1 constraints prevented splitting."
                    ),
                    severity="warning",
                )
            )

    def _check_no_heading(
        self,
        chunk: PlannedChunk,
        flags: Dict[str, List[ValidationFlag]],
    ) -> None:
        """Flag substantial chunks with no heading context."""
        if chunk.heading is None and chunk.estimated_tokens > 100:
            flags[chunk.chunk_id].append(
                ValidationFlag(
                    chunk_id=chunk.chunk_id,
                    flag_type="no_heading",
                    detail=(
                        f"Chunk has no heading (section context: "
                        f"{chunk.section_path}) but contains "
                        f"{chunk.estimated_tokens} tokens. "
                        f"May be an orphan paragraph or structureless section."
                    ),
                    severity="warning",
                )
            )

    def _check_orphan_caption(
        self,
        chunk: PlannedChunk,
        chunk_blocks: List[NormalizedBlock],
        block_to_chunk: Dict[str, str],
        flags: Dict[str, List[ValidationFlag]],
    ) -> None:
        """
        Error: a caption block whose caption_for target is in a DIFFERENT chunk.
        This means figure and caption were split, violating R1.1/R1.2.
        """
        for block in chunk_blocks:
            if block.type != "caption" or not block.caption_for:
                continue
            target_chunk = block_to_chunk.get(block.caption_for)
            if target_chunk is None:
                # Target was excluded or not found — not our problem here
                continue
            if target_chunk != chunk.chunk_id:
                flags[chunk.chunk_id].append(
                    ValidationFlag(
                        chunk_id=chunk.chunk_id,
                        flag_type="orphan_caption",
                        detail=(
                            f"Caption block {block.id} references "
                            f"figure/table '{block.caption_for}' which is in "
                            f"chunk '{target_chunk}', not this chunk. "
                            f"R1.1/R1.2 violated."
                        ),
                        severity="error",
                    )
                )

    def _check_table_integrity(
        self,
        chunk: PlannedChunk,
        chunk_blocks: List[NormalizedBlock],
        block_to_chunk: Dict[str, str],
        flags: Dict[str, List[ValidationFlag]],
    ) -> None:
        """
        Error: a table block whose caption_id is in a DIFFERENT chunk.
        Bidirectional with orphan_caption — catches both directions.
        """
        for block in chunk_blocks:
            if block.type != "table" or not block.caption_id:
                continue
            caption_chunk = block_to_chunk.get(block.caption_id)
            if caption_chunk is None:
                continue
            if caption_chunk != chunk.chunk_id:
                flags[chunk.chunk_id].append(
                    ValidationFlag(
                        chunk_id=chunk.chunk_id,
                        flag_type="table_integrity",
                        detail=(
                            f"Table block {block.id} has caption '{block.caption_id}' "
                            f"which is in chunk '{caption_chunk}', not this chunk. "
                            f"R1.2 violated."
                        ),
                        severity="error",
                    )
                )

    def _check_unknown_blocks(
        self,
        chunk: PlannedChunk,
        chunk_blocks: List[NormalizedBlock],
        flags: Dict[str, List[ValidationFlag]],
    ) -> None:
        """
        Warning: a substantial chunk with unknown-type blocks.
        Unknown blocks may have been misclassified by the parser.
        """
        if chunk.estimated_tokens <= 200:
            return  # Only flag substantial chunks
        unknown = [b for b in chunk_blocks if b.type == "unknown"]
        if unknown:
            flags[chunk.chunk_id].append(
                ValidationFlag(
                    chunk_id=chunk.chunk_id,
                    flag_type="unknown_blocks",
                    detail=(
                        f"Chunk contains {len(unknown)} 'unknown'-type block(s): "
                        f"{[b.id for b in unknown]}. Parser could not classify them."
                    ),
                    severity="warning",
                )
            )


    # ------------------------------------------------------------------
    # Summary logging
    # ------------------------------------------------------------------

    def _log_summary(
        self,
        plan: ChunkPlan,
        flags_by_chunk: Dict[str, List[ValidationFlag]],
        scores_by_chunk: Dict[str, float],
    ) -> None:
        threshold = self._config.score_threshold
        flagged = sum(1 for s in scores_by_chunk.values() if s < 100.0)
        needs_llm = sum(1 for s in scores_by_chunk.values() if s < threshold)
        clean = plan.total_planned - flagged

        self._log.info(
            "Validator: %d total | %d clean | %d flagged | %d below %.0f (LLM candidates)",
            plan.total_planned,
            clean,
            flagged,
            needs_llm,
            threshold,
        )

        # Count by flag type
        type_counts: Dict[str, int] = defaultdict(int)
        for flag_list in flags_by_chunk.values():
            for flag in flag_list:
                type_counts[flag.flag_type] += 1

        if type_counts:
            self._log.info("Validator flag breakdown:")
            for flag_type, count in sorted(type_counts.items()):
                self._log.info("  %-25s %d", flag_type + ":", count)
