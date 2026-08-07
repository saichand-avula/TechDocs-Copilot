"""
chunker.py
==========
Stage 3 — Orchestrator: SemanticChunker

Wires Normalizer → ChunkPlanner → ChunkValidator → ChunkBuilder
into a single, clean entry point.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Optional

from .chunk_builder import ChunkBuilder
from .chunk_planner import ChunkPlanner
from .chunk_validator import ChunkValidator
from .config import ChunkerConfig
from .models import ChunkManifest
from .normalizer import Normalizer


class SemanticChunker:
    """
    Orchestrates the full Stage 3 semantic chunking pipeline.

    Pipeline stages:
      1. Normalizer    — load IR, exclude chrome, resolve groups, detect continuations
      2. ChunkPlanner  — apply V1 rules → ChunkPlan
      3. ChunkValidator — inspect plan → ValidationFlags + scores
      4. ChunkBuilder  — assemble Chunk objects → write chunks.json

    Usage::

        config = ChunkerConfig(
            parsed_dir=Path("data/parsed/parser_v1/printer_manual"),
            output_dir=Path("data/chunks/printer_manual"),
        )
        chunker = SemanticChunker(config=config)
        manifest = chunker.run()
    """

    def __init__(
        self,
        config: ChunkerConfig,
        logger: Optional[logging.Logger] = None,
    ) -> None:
        self._config = config
        self._log = logger or logging.getLogger(__name__)

        self.normalizer = Normalizer(logger=self._log)
        self.planner = ChunkPlanner(config=config, logger=self._log)
        self.validator = ChunkValidator(config=config, logger=self._log)
        self.builder = ChunkBuilder(config=config, logger=self._log)

    def run(self) -> ChunkManifest:
        """
        Execute the full chunking pipeline.

        Returns
        -------
        ChunkManifest
            The completed manifest (also written to output_dir/chunks.json).
        """
        start = time.perf_counter()
        self._log.info("=" * 60)
        self._log.info("SemanticChunker: starting")
        self._log.info("  parsed_dir : %s", self._config.parsed_dir)
        self._log.info("  output_dir : %s", self._config.output_dir)
        self._log.info("  max_tokens : %d", self._config.max_tokens)
        self._log.info("=" * 60)

        # Step 1 — Normalize
        self._log.info("[1/4] Normalizer")
        blocks = self.normalizer.normalize(self._config.parsed_dir)

        # Extract document_id from the first non-excluded block
        document_id = self._extract_document_id(blocks)

        # Step 2 — Plan
        self._log.info("[2/4] ChunkPlanner")
        plan = self.planner.plan(blocks, document_id=document_id)

        # Step 3 — Validate
        self._log.info("[3/4] ChunkValidator")
        flags_by_chunk, scores_by_chunk = self.validator.validate(plan, blocks)

        # Step 4 — Build
        self._log.info("[4/4] ChunkBuilder")
        manifest = self.builder.build(
            blocks=blocks,
            plan=plan,
            flags_by_chunk=flags_by_chunk,
            scores_by_chunk=scores_by_chunk,
            output_dir=self._config.output_dir,
            document_id=document_id,
        )

        elapsed = time.perf_counter() - start
        self._log.info("=" * 60)
        self._log.info("SemanticChunker: complete in %.2f seconds", elapsed)
        self._log.info(
            "  Total chunks : %d", manifest.total_chunks
        )
        self._log.info(
            "  Clean        : %d", manifest.clean_chunks
        )
        self._log.info(
            "  Flagged      : %d", manifest.flagged_chunks
        )
        self._log.info("=" * 60)

        return manifest

    def _extract_document_id(self, blocks) -> str:
        """Extract document_id from blocks. Falls back to 'unknown'."""
        for block in blocks:
            if not block.excluded:
                # document_id isn't on NormalizedBlock, but we can read it from
                # the parsed_dir's document.json
                break
        try:
            import json
            doc_path = self._config.parsed_dir / "document.json"
            with doc_path.open(encoding="utf-8") as fh:
                return json.load(fh).get("document_id", "unknown")
        except Exception:
            return "unknown"
