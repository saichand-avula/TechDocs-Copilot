"""
src/indexing/indexer.py
────────────────────────
Indexer orchestrator: ChunkStore → ChromaDB + BM25 in one pass.
"""
from __future__ import annotations

import json
import logging
import time
from pathlib import Path

from .config import IndexConfig
from .chunk_store import ChunkStore
from .chroma_indexer import ChromaIndexer
from .bm25_indexer import BM25Indexer

log = logging.getLogger(__name__)


class Indexer:
    """
    Orchestrates the full indexing pipeline:
      1. Warm ChunkStore (load all chunks.json into memory)
      2. Build ChromaDB collection from pre-generated embeddings
      3. Build BM25 index over full corpus

    Precondition: SemanticEmbedder must have already been run
    (data/embeddings/{manual}/ must exist).
    """

    def __init__(self, config: IndexConfig | None = None) -> None:
        self._cfg = config or IndexConfig()
        self.store = ChunkStore(self._cfg.chunks_dir)
        self._chroma = ChromaIndexer(self._cfg)
        self._bm25 = BM25Indexer(self._cfg)

    def build(self) -> dict:
        """
        Run the full indexing pipeline.
        Returns a summary dict with stats for both indexes.
        """
        start = time.perf_counter()

        log.info("=" * 60)
        log.info("Indexer: starting")
        log.info("  chunks_dir    : %s", self._cfg.chunks_dir)
        log.info("  embeddings_dir: %s", self._cfg.embeddings_dir)
        log.info("  vectordb_dir  : %s", self._cfg.vectordb_dir)
        log.info("=" * 60)

        # Step 1: Warm store (loads all 11 chunks.json into memory)
        log.info("[1/3] ChunkStore warm")
        self.store.warm()

        # Patch manual_name into chunks for BM25 corpus_map
        # (ChunkStore.get_corpus() iterates sorted manuals — we tag each chunk)
        self._tag_chunks_with_manual_name()

        # Step 2: ChromaDB
        log.info("[2/3] ChromaIndexer")
        chroma_summary = self._chroma.build(self.store)

        # Step 3: BM25
        log.info("[3/3] BM25Indexer")
        bm25_meta = self._bm25.build(self.store)

        elapsed = time.perf_counter() - start
        summary = {
            "chroma": chroma_summary,
            "chroma_total": sum(chroma_summary.values()),
            "bm25_docs": bm25_meta["num_documents"],
            "elapsed_s": round(elapsed, 2),
        }

        # Save run summary
        out_path = Path(self._cfg.vectordb_dir) / "index_run.json"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

        log.info("=" * 60)
        log.info("Indexer: complete in %.2fs", elapsed)
        log.info("  ChromaDB vectors : %d", summary["chroma_total"])
        log.info("  BM25 documents   : %d", summary["bm25_docs"])
        log.info("=" * 60)

        return summary

    def _tag_chunks_with_manual_name(self) -> None:
        """
        Inject _manual_name into each chunk dict so BM25Indexer
        can build corpus_map correctly.
        chunk_ids are not globally unique — manual_name is essential.
        """
        for manual_name in self.store.list_manuals():
            for chunk in self.store.get_all_chunks(manual_name):
                chunk["_manual_name"] = manual_name
