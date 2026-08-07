"""
normalizer.py
=============
Stage 3 — Step 1: Normalizer

Responsibility:
  Load all blocks from parser_v1/<doc>/, apply exclusions, resolve group_id
  completeness, and detect table continuations.

Output:
  list[NormalizedBlock] in reading order (excluded blocks are included in the
  list but marked excluded=True so the planner can skip them transparently).

What this module does NOT do:
  - It does not make any chunking decisions.
  - It does not modify the parser_v1 files.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

from .models import NormalizedBlock

# Block types that are purely navigation/chrome and never content
_CHROME_TYPES: frozenset = frozenset({"page_header", "page_footer"})

# Semantic roles that indicate navigation chrome
_CHROME_ROLES: frozenset = frozenset({"navigation"})

# Block types that are TOC — always excluded
_TOC_TYPES: frozenset = frozenset({"toc"})


class Normalizer:
    """
    Loads parser_v1 output and produces a clean, ordered list of NormalizedBlocks.

    Usage::

        normalizer = Normalizer(logger=logger)
        blocks = normalizer.normalize(parsed_dir)
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self._log = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def normalize(self, parsed_dir: Path) -> List[NormalizedBlock]:
        """
        Load and normalize all blocks from a parser_v1/<doc>/ directory.

        Parameters
        ----------
        parsed_dir : Path
            Path to the parser_v1/<doc>/ directory (must contain document.json).

        Returns
        -------
        list[NormalizedBlock]
            All blocks in global reading order.
            Excluded blocks are present but marked excluded=True.
        """
        parsed_dir = Path(parsed_dir)
        self._log.info("Normalizer: loading from %s", parsed_dir)

        # Step 1: Load and sort all raw blocks
        raw_blocks = self._load_all_blocks(parsed_dir)
        self._log.info("Normalizer: loaded %d raw blocks", len(raw_blocks))

        # Step 2: Convert to NormalizedBlock objects
        blocks = [self._to_normalized(b) for b in raw_blocks]

        # Step 3: Exclusion pass
        excluded_count = self._apply_exclusions(blocks)
        self._log.info("Normalizer: excluded %d chrome/toc/decorative blocks", excluded_count)

        # Step 4: Group completeness check
        self._check_group_completeness(blocks)

        # Step 5: Table continuation detection
        continuation_count = self._detect_table_continuations(blocks)
        if continuation_count:
            self._log.info(
                "Normalizer: detected %d table continuation groups", continuation_count
            )

        active = sum(1 for b in blocks if not b.excluded)
        self._log.info(
            "Normalizer: %d active blocks ready for planning (%d excluded)",
            active,
            excluded_count,
        )
        return blocks

    # ------------------------------------------------------------------
    # Step 1: Load raw blocks from all page files
    # ------------------------------------------------------------------

    def _load_all_blocks(self, parsed_dir: Path) -> List[dict]:
        """
        Load all blocks from page files, sorted by reading_order.

        Also builds self._toc_pages: the set of page numbers where
        toc_tables > 0 in page_stats. Blocks on those pages that are
        heading-type navigation titles (e.g. 'Table of contents',
        'List of tables') will be excluded in the exclusion pass.
        """
        doc_path = parsed_dir / "document.json"
        with doc_path.open(encoding="utf-8") as fh:
            doc_index = json.load(fh)

        page_entries = doc_index.get("page_index", [])
        if not page_entries:
            raise ValueError(f"document.json has no page_index entries: {doc_path}")

        # Track pages that are TOC pages (toc_tables > 0)
        self._toc_pages: set = set()

        all_blocks: List[dict] = []
        for entry in page_entries:
            page_file = parsed_dir / entry["path"]
            if not page_file.exists():
                self._log.warning("Page file not found, skipping: %s", page_file)
                continue
            with page_file.open(encoding="utf-8") as fh:
                page_data = json.load(fh)

            # Mark as TOC page if it contains toc_tables
            page_stats = page_data.get("page_stats", {})
            if page_stats.get("toc_tables", 0) > 0:
                self._toc_pages.add(page_data.get("page", -1))

            for block in page_data.get("blocks", []):
                all_blocks.append(block)

        if self._toc_pages:
            self._log.debug("Normalizer: TOC pages detected: %s", sorted(self._toc_pages))

        # Sort by reading_order (guaranteed monotonically increasing by schema,
        # but sort defensively in case of any edge cases)
        all_blocks.sort(key=lambda b: b.get("reading_order", 0))
        return all_blocks

    # ------------------------------------------------------------------
    # Step 2: Convert raw dict → NormalizedBlock
    # ------------------------------------------------------------------

    def _to_normalized(self, raw: dict) -> NormalizedBlock:
        """Convert a raw block dict from the page JSON to a NormalizedBlock."""
        return NormalizedBlock(
            id=raw["id"],
            type=raw["type"],
            reading_order=raw.get("reading_order", 0),
            page=raw.get("page", 0),
            content=raw.get("content"),
            chunk_hint=raw.get("chunk_hint"),
            semantic_role=raw.get("semantic_role"),
            section=raw.get("section"),
            section_id=raw.get("section_id"),
            parent_section_id=raw.get("parent_section_id"),
            section_level=raw.get("section_level"),
            section_path=raw.get("section_path"),
            group_id=raw.get("group_id"),
            # figure fields
            image_path=raw.get("image_path"),
            caption=raw.get("caption"),
            caption_id=raw.get("caption_id"),
            decorative=raw.get("decorative"),
            width=raw.get("width"),
            height=raw.get("height"),
            image_hash=raw.get("image_hash"),
            figure_number=raw.get("figure_number"),
            previous_block_id=raw.get("previous_block_id"),
            next_block_id=raw.get("next_block_id"),
            # table fields
            markdown=raw.get("markdown"),
            title=raw.get("title"),
            table_number=raw.get("table_number"),
            rows=raw.get("rows"),
            cols=raw.get("cols"),
            # caption fields
            caption_for=raw.get("caption_for"),
            # admonition/heading fields
            severity=raw.get("severity"),
            level=raw.get("level"),
            section_number=raw.get("section_number"),
            # provenance
            source=raw.get("source"),
            bbox=raw.get("bbox"),
        )

    # ------------------------------------------------------------------
    # Step 3: Exclusion pass
    # ------------------------------------------------------------------

    def _apply_exclusions(self, blocks: List[NormalizedBlock]) -> int:
        """
        Mark blocks as excluded based on their type/role.

        Rules applied:
          - type in {toc}              → excluded (R5.5: TOC is isolated by exclusion)
          - type in {page_header, page_footer} → excluded (R5.3)
          - semantic_role == navigation → excluded (running headers etc.)
          - decorative figure (width < 20 AND height < 20) → excluded (Known Behaviour 2)
          - chunk_hint absent AND type in chrome types → excluded

        Returns the count of newly excluded blocks.
        """
        excluded = 0
        for block in blocks:
            if block.excluded:
                continue  # already excluded by a prior pass

            reason = self._get_exclusion_reason(block)
            if reason:
                block.excluded = True
                block.exclusion_reason = reason
                excluded += 1

        return excluded

    def _get_exclusion_reason(self, block: NormalizedBlock) -> Optional[str]:
        """Return an exclusion reason string, or None if the block should be kept."""
        # TOC blocks — always excluded (R5.5)
        if block.type in _TOC_TYPES:
            return "toc_navigation"

        # Page chrome — always excluded (R5.3)
        if block.type in _CHROME_TYPES:
            return "page_chrome"

        # Navigation role (running headers, page numbers)
        if block.semantic_role in _CHROME_ROLES:
            return "page_chrome_role"

        # Heading blocks on TOC pages — these are navigation section titles
        # (e.g. 'Table of contents', 'List of tables'), not content headings.
        # Detected via page_stats.toc_tables > 0 on the page they appear on.
        toc_pages = getattr(self, "_toc_pages", set())
        if block.type == "heading" and block.page in toc_pages:
            return "toc_page_heading"

        # Decorative figures — very small icons (Known Behaviour 2 in SCHEMA.md)
        if block.type == "figure":
            w = block.width or 0
            h = block.height or 0
            if w < 20 and h < 20:
                return "decorative_figure"

        return None

    # ------------------------------------------------------------------
    # Step 4: Group completeness check
    # ------------------------------------------------------------------

    def _check_group_completeness(self, blocks: List[NormalizedBlock]) -> None:
        """
        Verify that all members of each group_id are present and non-excluded.

        Logs a warning for any incomplete group. Does not modify blocks
        (the planner handles incomplete groups gracefully).
        """
        # Collect group members
        groups: Dict[str, List[NormalizedBlock]] = {}
        for block in blocks:
            if block.group_id and not block.excluded:
                groups.setdefault(block.group_id, []).append(block)

        # Check expected members: figure/table + caption should form a pair
        for group_id, members in groups.items():
            types = {b.type for b in members}
            has_content = bool(types & {"figure", "table"})
            has_caption = "caption" in types
            if has_content and not has_caption:
                self._log.warning(
                    "Group %s has a figure/table but no caption block. "
                    "IDs: %s — planner will keep them together anyway.",
                    group_id,
                    [b.id for b in members],
                )

    # ------------------------------------------------------------------
    # Step 5: Table continuation detection
    # ------------------------------------------------------------------

    def _detect_table_continuations(self, blocks: List[NormalizedBlock]) -> int:
        """
        Detect table continuation slices and mark them with a shared
        _table_continuation_key so the Chunk Planner can merge them.

        Detection rule (validated against actual schema data):
          A table block is a continuation slice if:
            1. Its table_number matches a previously seen table_number
            2. AND "(continued)" appears in its title (case-insensitive)

        This is grounded in the actual parser_v1 data:
          page 746: tbl_0075, table_number="Table 2-13", title="...Alphabetical parts list (continued)"
          page 747: tbl_0076, table_number="Table 2-13", title="...Alphabetical parts list (continued)"
          → Both are continuation slices of the same logical table.

        Each unique (table_number) gets a shared continuation_key.
        The first occurrence with that table_number (whether or not it
        says "continued") is the anchor.

        Returns the number of continuation groups found.
        """
        # Map table_number → continuation_key (= table_number itself)
        seen_table_numbers: Dict[str, str] = {}
        continuation_groups: Set[str] = set()

        table_blocks = [b for b in blocks if b.type == "table" and not b.excluded]

        for block in table_blocks:
            t_num = block.table_number
            if not t_num:
                continue

            title = (block.title or "").lower()
            is_continuation = "continued" in title

            if t_num in seen_table_numbers:
                # This table_number was seen before — mark as continuation
                key = seen_table_numbers[t_num]
                block._table_continuation_key = key
                continuation_groups.add(key)
                self._log.debug(
                    "Table continuation: block %s (page %d) → group key '%s'",
                    block.id,
                    block.page,
                    key,
                )
            else:
                # First occurrence of this table_number
                seen_table_numbers[t_num] = t_num  # key = table_number string
                if is_continuation:
                    # It says "continued" but we haven't seen the anchor —
                    # treat it as its own anchor (anchor may have been excluded
                    # or on a page we skipped)
                    block._table_continuation_key = t_num
                    continuation_groups.add(t_num)
                    self._log.debug(
                        "Table continuation anchor (no prior): block %s (page %d), key '%s'",
                        block.id,
                        block.page,
                        t_num,
                    )

        return len(continuation_groups)
