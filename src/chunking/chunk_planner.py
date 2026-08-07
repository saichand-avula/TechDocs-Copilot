"""
chunk_planner.py
================
Stage 3 — Step 2: Chunk Planner

Responsibility:
  Apply the V1 semantic chunking rules to a list of NormalizedBlocks and
  produce a ChunkPlan — an intermediate, in-memory plan of how blocks should
  be grouped into chunks.

Rules applied (see semantic_chunking_rules.md for full specification):
  P1 — Atomic Attachment:  R1.1 R1.2 R1.3 R1.4 R1.5
  P2 — Structural Boundary: R2.1 R2.2 R2.3 R2.4 R2.5
  P3 — Aggregation:        R3.1 R3.2 R3.3 R3.4 R3.5 R3.6
  P4 — Overflow / Split:   R4.1 R4.2 R4.3 R4.4 R4.5
  P5 — Fallback:           R5.1 R5.2 R5.3 R5.4 R5.5

What this module does NOT do:
  - It does not load files (Normalizer's job).
  - It does not validate chunk quality (Validator's job).
  - It does not write output (Builder's job).
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Set

from .config import ChunkerConfig
from .models import ChunkPlan, NormalizedBlock, PlannedChunk

# ---------------------------------------------------------------------------
# Rule constants — used in rules_applied tracking
# ---------------------------------------------------------------------------
R1_1 = "R1.1"   # figure + caption atomic
R1_2 = "R1.2"   # table + caption + notes atomic
R1_3 = "R1.3"   # admonition attaches to trigger block
R1_4 = "R1.4"   # equation + definition
R1_5 = "R1.5"   # list + intro paragraph
R2_1 = "R2.1"   # H1 always new chunk
R2_2 = "R2.2"   # H2 default new chunk
R2_3 = "R2.3"   # H3+ grouped under H2
R3_1 = "R3.1"   # multi-page table continuation merged
R3_2 = "R3.2"   # procedure steps merged
R3_3 = "R3.3"   # short sibling H2 merge
R3_4 = "R3.4"   # orphan paragraph attaches to preceding
R3_5 = "R3.5"   # subfigure series merged
R3_6 = "R3.6"   # code block attaches to intro
R4_1 = "R4.1"   # overflow threshold flush
R4_2 = "R4.2"   # overflow split at H3
R4_3 = "R4.3"   # overflow split at procedure phases
R4_4 = "R4.4"   # overflow split at table row groups
R5_1 = "R5.1"   # document metadata first chunk
R5_2 = "R5.2"   # orphan block attaches to previous
R5_4 = "R5.4"   # cross-reference attaches to paragraph

# Semantic roles that represent admonitions (R1.3)
_ADMONITION_ROLES: frozenset = frozenset(
    {"warning", "caution", "note", "tip", "important"}
)

# Block types that always attach to the current chunk (never trigger a flush)
_STICKY_TYPES: frozenset = frozenset(
    {"caption", "list_item", "code", "reference", "footnote"}
)


def _count_tokens(blocks: List[NormalizedBlock]) -> int:
    """
    Estimate token count as total word count across all text content.
    V1 approximation: len(text.split()). Sufficient for threshold comparisons.
    """
    total = 0
    for block in blocks:
        if block.content:
            total += len(block.content.split())
        if block.markdown:
            total += len(block.markdown.split())
        if block.caption:
            total += len(block.caption.split())
    return total


class ChunkPlanner:
    """
    Applies V1 semantic chunking rules to produce a ChunkPlan.

    Usage::

        planner = ChunkPlanner(config=config, logger=logger)
        plan = planner.plan(blocks, document_id="doc_7e4fe272")
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

    def plan(
        self, blocks: List[NormalizedBlock], document_id: str
    ) -> ChunkPlan:
        """
        Produce a ChunkPlan from a list of NormalizedBlocks.

        Parameters
        ----------
        blocks : list[NormalizedBlock]
            All blocks in reading order, including excluded ones.
        document_id : str
            The document_id from the parser_v1 document.json.

        Returns
        -------
        ChunkPlan
        """
        active = [b for b in blocks if not b.excluded]
        self._log.info("ChunkPlanner: planning %d active blocks", len(active))

        # --- Main pass ---
        chunks = self._main_pass(active)

        # --- Post-pass: merge short H2 siblings (R3.3) ---
        chunks = self._merge_short_siblings(chunks, active)

        # --- Overflow split pass (R4.1-R4.4) ---
        chunks = self._split_overflow(chunks, active)

        self._log.info("ChunkPlanner: produced %d chunks", len(chunks))

        return ChunkPlan(
            document_id=document_id,
            total_planned=len(chunks),
            chunks=chunks,
        )

    # ------------------------------------------------------------------
    # Main planning pass
    # ------------------------------------------------------------------

    def _main_pass(self, active: List[NormalizedBlock]) -> List[PlannedChunk]:
        """
        Single forward pass over active blocks applying P1/P2/P3/P5 rules.
        Returns a list of PlannedChunks (before overflow splitting).
        """
        chunks: List[PlannedChunk] = []
        current: List[NormalizedBlock] = []
        current_rules: List[str] = []
        group_ids_in_current: Set[str] = set()

        # Track table continuation keys currently being accumulated
        active_continuation_keys: Set[str] = set()

        def flush(reason: str, extra_rules: Optional[List[str]] = None) -> None:
            """Flush current_blocks into a new PlannedChunk."""
            if not current:
                return
            rules = sorted(set(current_rules + (extra_rules or [])))
            chunk = self._make_chunk(
                blocks=current,
                chunk_index=len(chunks) + 1,
                reason=reason,
                rules=rules,
            )
            chunks.append(chunk)
            current.clear()
            current_rules.clear()
            group_ids_in_current.clear()
            active_continuation_keys.clear()

        for block in active:

            # ── P1: Atomic group membership (R1.1, R1.2, R1.3, R1.4, R1.5) ──
            # Blocks with a group_id always attach to the current chunk.
            # Their group_id being new is not a boundary trigger.
            if block.group_id is not None:
                rule = self._group_rule(block)
                if rule not in current_rules:
                    current_rules.append(rule)
                group_ids_in_current.add(block.group_id)
                current.append(block)
                continue

            # ── P1: Sticky types (captions, lists, code, references, footnotes) ──
            if block.type in _STICKY_TYPES:
                rule = self._sticky_rule(block)
                if rule not in current_rules:
                    current_rules.append(rule)
                current.append(block)
                continue

            # ── P1: Admonition attaches to preceding content (R1.3) ──
            if block.semantic_role in _ADMONITION_ROLES or block.type == "admonition":
                if R1_3 not in current_rules:
                    current_rules.append(R1_3)
                current.append(block)
                continue

            # ── P3: Table continuation slice stays in current chunk (R3.1) ──
            cont_key = block._table_continuation_key
            if cont_key is not None:
                if cont_key not in active_continuation_keys and active_continuation_keys:
                    # Different table — flush first
                    flush(
                        reason=f"New table continuation group '{cont_key}' after prior table.",
                        extra_rules=[R3_1],
                    )
                active_continuation_keys.add(cont_key)
                if R3_1 not in current_rules:
                    current_rules.append(R3_1)
                current.append(block)
                continue

            # ── P2: H1 → unconditional boundary (R2.1) ──
            if block.type == "heading" and (block.section_level or 99) == 1:
                flush(
                    reason=f"H1 heading '{block.content}' — unconditional boundary.",
                    extra_rules=[R2_1],
                )
                current.append(block)
                current_rules.append(R2_1)
                continue

            # ── P2: H2 → default boundary (R2.2) ──
            if block.type == "heading" and (block.section_level or 99) == 2:
                flush(
                    reason=f"H2 heading '{block.content}' — default section boundary.",
                    extra_rules=[R2_2],
                )
                current.append(block)
                current_rules.append(R2_2)
                continue

            # ── P2: H3+ → grouped under H2, no boundary (R2.3) ──
            if block.type == "heading" and (block.section_level or 99) >= 3:
                if R2_3 not in current_rules:
                    current_rules.append(R2_3)
                # Check overflow before adding
                projected = _count_tokens(current) + _count_tokens([block])
                if projected > self._config.max_tokens and current:
                    flush(
                        reason="H3+ heading caused overflow — split before it.",
                        extra_rules=[R4_1, R2_3],
                    )
                current.append(block)
                if R2_3 not in current_rules:
                    current_rules.append(R2_3)
                continue

            # ── P4: Overflow check (R4.1) ──
            # CRITICAL: Never flush if the last block in current has an open
            # group_id. That would split an atomic P1 group (e.g. figure from
            # its caption). Only flush when no group is open.
            projected = _count_tokens(current) + _count_tokens([block])
            last_block_group = current[-1].group_id if current else None
            if projected > self._config.max_tokens and current and last_block_group is None:
                flush(
                    reason=f"Token overflow at block {block.id} "
                    f"(projected {projected} > {self._config.max_tokens}).",
                    extra_rules=[R4_1],
                )

            # ── Default: add to current chunk ──
            current.append(block)
            # Infer rule for first block in a new chunk
            if len(current) == 1:
                current_rules.append(self._default_rule(block))

        # Final flush
        flush(reason="End of document — final flush.")
        return chunks

    # ------------------------------------------------------------------
    # Post-pass: merge short H2 siblings (R3.3)
    # ------------------------------------------------------------------

    def _merge_short_siblings(
        self,
        chunks: List[PlannedChunk],
        active: List[NormalizedBlock],
    ) -> List[PlannedChunk]:
        """
        Merge adjacent H2-level chunks that are both below the short-sibling
        threshold AND share the same parent_section_id. (R3.3)

        Applied transitively: keeps merging until stable.
        """
        block_map: Dict[str, NormalizedBlock] = {b.id: b for b in active}
        threshold = self._config.short_sibling_threshold

        def first_block(chunk: PlannedChunk) -> Optional[NormalizedBlock]:
            return block_map.get(chunk.block_ids[0]) if chunk.block_ids else None

        changed = True
        while changed:
            changed = False
            merged: List[PlannedChunk] = []
            i = 0
            while i < len(chunks):
                if i + 1 >= len(chunks):
                    merged.append(chunks[i])
                    i += 1
                    continue

                a = chunks[i]
                b = chunks[i + 1]
                fa = first_block(a)
                fb = first_block(b)

                # Both must be under threshold
                if a.estimated_tokens >= threshold or b.estimated_tokens >= threshold:
                    merged.append(a)
                    i += 1
                    continue

                # Both must be H2-level sections (first block is H2)
                a_is_h2 = fa is not None and fa.type == "heading" and (fa.section_level or 0) == 2
                b_is_h2 = fb is not None and fb.type == "heading" and (fb.section_level or 0) == 2
                if not (a_is_h2 and b_is_h2):
                    merged.append(a)
                    i += 1
                    continue

                # Must share the same EXPLICIT parent_section_id.
                # If parent_section_id is None or "" on either side, these are
                # top-level unnumbered sections — merging them would combine
                # unrelated content (e.g. "Service Manual: Repair" + "Copyright").
                # An empty match is NOT a valid shared parent.
                a_parent = fa.parent_section_id or ""
                b_parent = fb.parent_section_id or ""
                if not a_parent or not b_parent or a_parent != b_parent:
                    merged.append(a)
                    i += 1
                    continue

                # Merge b into a
                self._log.debug(
                    "R3.3: merging short sibling chunks %s + %s (tokens: %d + %d)",
                    a.chunk_id,
                    b.chunk_id,
                    a.estimated_tokens,
                    b.estimated_tokens,
                )
                merged_ids = a.block_ids + b.block_ids
                merged_tokens = a.estimated_tokens + b.estimated_tokens
                merged_rules = sorted(set(a.rules_applied + b.rules_applied + [R3_3]))
                merged_chunk = PlannedChunk(
                    chunk_id=a.chunk_id,
                    block_ids=merged_ids,
                    heading=a.heading,
                    section_path=a.section_path,
                    estimated_tokens=merged_tokens,
                    reading_order_start=a.reading_order_start,
                    reading_order_end=b.reading_order_end,
                    reason=(
                        f"Merged short sibling H2 sections "
                        f"'{a.heading}' ({a.estimated_tokens}t) + "
                        f"'{b.heading}' ({b.estimated_tokens}t). "
                        f"Both under {threshold}-token threshold."
                    ),
                    rules_applied=merged_rules,
                )
                merged.append(merged_chunk)
                changed = True
                i += 2  # skip b

            chunks = merged

        return chunks

    # ------------------------------------------------------------------
    # Overflow split pass (R4.1 - R4.4)
    # ------------------------------------------------------------------

    def _split_overflow(
        self,
        chunks: List[PlannedChunk],
        active: List[NormalizedBlock],
    ) -> List[PlannedChunk]:
        """
        Re-scan chunks that exceed max_chunk_tokens and split them at H3
        boundaries. R4.2. Never splits P1 atomic groups.

        R4.3 (procedure phases) and R4.4 (table row groups) are not
        implemented in V1 — they require domain knowledge that will come
        from evaluation on real manuals.
        """
        block_map: Dict[str, NormalizedBlock] = {b.id: b for b in active}
        result: List[PlannedChunk] = []
        new_chunk_counter = [len(chunks)]  # mutable counter for new chunk IDs

        for chunk in chunks:
            if chunk.estimated_tokens <= self._config.max_tokens:
                result.append(chunk)
                continue

            # Try to split at H3 boundaries (R4.2)
            split_chunks = self._split_at_h3(chunk, block_map, new_chunk_counter)
            if len(split_chunks) > 1:
                self._log.debug(
                    "R4.2: split chunk %s (%d tokens) into %d sub-chunks at H3 boundaries",
                    chunk.chunk_id,
                    chunk.estimated_tokens,
                    len(split_chunks),
                )
                result.extend(split_chunks)
            else:
                # Could not split (no H3s, or P1 constraint) — keep as-is
                # Validator will flag it as too_large
                result.append(chunk)

        return result

    def _split_at_h3(
        self,
        chunk: PlannedChunk,
        block_map: Dict[str, NormalizedBlock],
        counter: List[int],
    ) -> List[PlannedChunk]:
        """
        Split a chunk at H3 (level >= 3) heading boundaries.
        Preserves P1 atomic groups (group_id members stay together).
        Returns a list of 1+ PlannedChunks.
        """
        sub: List[NormalizedBlock] = []
        result: List[PlannedChunk] = []
        active_group_ids: Set[str] = set()

        def flush_sub(reason: str) -> None:
            if not sub:
                return
            counter[0] += 1
            new_id = f"chunk_{counter[0]:04d}"
            result.append(
                self._make_chunk(
                    blocks=sub,
                    chunk_index=counter[0],
                    reason=reason,
                    rules=sorted(set(chunk.rules_applied + [R4_2])),
                )
            )
            sub.clear()
            active_group_ids.clear()

        for bid in chunk.block_ids:
            block = block_map.get(bid)
            if block is None:
                continue

            # P1: group members never trigger a split
            if block.group_id is not None:
                active_group_ids.add(block.group_id)
                sub.append(block)
                continue

            # H3+ heading = split boundary, but only if we have content already
            if block.type == "heading" and (block.section_level or 99) >= 3 and sub:
                # Don't split if the current sub is only a heading (avoid empty chunks)
                non_heading = [b for b in sub if b.type != "heading"]
                if non_heading:
                    flush_sub(
                        reason=f"R4.2 overflow split at H3 heading '{block.content}'."
                    )

            sub.append(block)

        flush_sub(reason="R4.2 final sub-chunk flush.")
        return result if result else [chunk]

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _make_chunk(
        self,
        blocks: List[NormalizedBlock],
        chunk_index: int,
        reason: str,
        rules: List[str],
    ) -> PlannedChunk:
        """Create a PlannedChunk from a list of NormalizedBlocks."""
        block_ids = [b.id for b in blocks]
        heading = self._find_heading(blocks)
        section_path = self._find_section_path(blocks)
        tokens = _count_tokens(blocks)
        ro_start = blocks[0].reading_order if blocks else 0
        ro_end = blocks[-1].reading_order if blocks else 0

        return PlannedChunk(
            chunk_id=f"chunk_{chunk_index:04d}",
            block_ids=block_ids,
            heading=heading,
            section_path=section_path,
            estimated_tokens=tokens,
            reading_order_start=ro_start,
            reading_order_end=ro_end,
            reason=reason,
            rules_applied=rules,
        )

    def _find_heading(self, blocks: List[NormalizedBlock]) -> Optional[str]:
        """Return the content of the first heading block, or None."""
        for b in blocks:
            if b.type == "heading" and b.content:
                return b.content.strip()
        return None

    def _find_section_path(
        self, blocks: List[NormalizedBlock]
    ) -> Optional[List[Dict[str, str]]]:
        """Return the section_path of the first block that has one."""
        for b in blocks:
            if b.section_path:
                return b.section_path
        return None

    def _group_rule(self, block: NormalizedBlock) -> str:
        """Return the primary P1 rule for a group-member block."""
        if block.type in {"figure", "caption"} and "fig" in (block.group_id or ""):
            return R1_1
        if block.type in {"table", "caption"} and "tbl" in (block.group_id or ""):
            return R1_2
        return R1_1  # default to figure rule for unknown group types

    def _sticky_rule(self, block: NormalizedBlock) -> str:
        """Return the P1/P3 rule for a sticky-type block."""
        if block.type == "caption":
            return R1_1
        if block.type == "list_item":
            return R1_5
        if block.type == "code":
            return R3_6
        if block.type in {"reference", "footnote"}:
            return R5_4
        return R1_5  # default

    def _default_rule(self, block: NormalizedBlock) -> str:
        """Return the most applicable rule for the first block of a new chunk."""
        if block.type == "heading":
            level = block.section_level or 99
            if level == 1:
                return R2_1
            if level == 2:
                return R2_2
            return R2_3
        return R5_2  # orphan / default attach
