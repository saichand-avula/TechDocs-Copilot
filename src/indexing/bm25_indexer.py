"""
src/indexing/bm25_indexer.py
─────────────────────────────
Builds a BM25 sparse index over all chunk text for hybrid retrieval.

The BM25 index is corpus-wide (all 11 manuals).
Saved to data/vectordb/bm25/:
    corpus_index.pkl    – serialized BM25Okapi object
    corpus_map.json     – [{manual_name, chunk_id, heading}] in corpus order
    bm25_meta.json      – build stats

At retrieval time, BM25 scores are fused with dense scores via RRF.
"""
from __future__ import annotations

import json
import logging
import pickle
from pathlib import Path

from .config import IndexConfig
from .chunk_store import ChunkStore

log = logging.getLogger(__name__)


class BM25Indexer:
    """Builds and persists a BM25 index over the full chunk corpus."""

    def __init__(self, config: IndexConfig | None = None) -> None:
        self._cfg = config or IndexConfig()

    def build(self, store: ChunkStore) -> dict:
        """
        Build BM25 index from all chunks in the store.
        Returns build stats.
        """
        from rank_bm25 import BM25Okapi

        log.info("BM25: loading corpus from ChunkStore…")
        all_chunks = store.get_corpus()
        log.info("BM25: %d chunks in corpus", len(all_chunks))

        # Build tokenized corpus
        corpus: list[list[str]] = []
        corpus_map: list[dict] = []

        for chunk in all_chunks:
            text = chunk.get("text") or ""
            tokens = text.lower().split()
            corpus.append(tokens)
            corpus_map.append(
                {
                    "manual_name": self._get_manual_name(chunk),
                    "chunk_id":    chunk["chunk_id"],
                    "heading":     chunk.get("heading") or "",
                }
            )

        log.info("BM25: fitting index on %d documents…", len(corpus))
        bm25 = BM25Okapi(corpus)

        # Persist
        bm25_dir = Path(self._cfg.bm25_dir)
        bm25_dir.mkdir(parents=True, exist_ok=True)

        index_path = bm25_dir / "corpus_index.pkl"
        map_path = bm25_dir / "corpus_map.json"
        meta_path = bm25_dir / "bm25_meta.json"

        with open(index_path, "wb") as fh:
            pickle.dump(bm25, fh, protocol=pickle.HIGHEST_PROTOCOL)

        map_path.write_text(json.dumps(corpus_map, indent=2), encoding="utf-8")

        import time
        meta = {
            "num_documents": len(corpus),
            "avg_tokens_per_doc": sum(len(t) for t in corpus) / max(len(corpus), 1),
            "bm25_class": "BM25Okapi",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log.info(
            "BM25: saved — %d docs, avg %.0f tokens/doc → %s",
            meta["num_documents"],
            meta["avg_tokens_per_doc"],
            bm25_dir,
        )
        return meta

    @staticmethod
    def _get_manual_name(chunk: dict) -> str:
        """
        chunk dicts from ChunkStore don't have manual_name directly
        (it's in the manifest root). We recover it from document_id by
        looking at the section_path or we rely on corpus_map building
        order — caller provides it via get_corpus() which iterates
        sorted manual names.

        Since ChunkStore.get_corpus() is sorted, and we build corpus_map
        in the same order, we can patch manual_name in the Indexer that
        calls us. For now, fall back to chunk_id prefix heuristic which
        is safe enough for the map (manual_name is also stored in Chroma).
        """
        # Will be patched by Indexer.build() — see indexer.py
        return chunk.get("_manual_name", "")
