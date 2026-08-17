"""
src/indexing/chroma_indexer.py
───────────────────────────────
Builds (or updates) the single ChromaDB collection from pre-generated
embedding files + chunk metadata.

Collection: techdocs_chunks
  - id     : "{document_id}__{chunk_id}"  (globally unique)
  - document: formatted text (for BM25 within Chroma, if needed)
  - embedding: 1536-dim float32 vector
  - metadata: flat key-value store (manual_name, chunk_id, pages, etc.)
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np

from .config import IndexConfig
from .chunk_store import ChunkStore

log = logging.getLogger(__name__)


class ChromaIndexer:
    """
    Builds the ChromaDB vector collection from saved embeddings + chunk metadata.
    Re-running is safe: existing IDs are upserted (not duplicated).
    """

    def __init__(self, config: IndexConfig | None = None) -> None:
        self._cfg = config or IndexConfig()

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def build(self, store: ChunkStore) -> dict[str, int]:
        """
        Index all manuals that have embeddings in cfg.embeddings_dir.
        Returns {manual_name: chunks_indexed}.
        """
        import chromadb

        # Persistent ChromaDB at data/vectordb/chroma/
        chroma_path = self._cfg.vectordb_dir / "chroma"
        chroma_path.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(path=str(chroma_path))
        collection = client.get_or_create_collection(
            name=self._cfg.collection_name,
            metadata={"hnsw:space": "cosine"},
        )

        log.info(
            "ChromaDB: collection '%s' at %s", self._cfg.collection_name, chroma_path
        )

        emb_root = Path(self._cfg.embeddings_dir)
        summary: dict[str, int] = {}

        for emb_dir in sorted(emb_root.iterdir()):
            if not emb_dir.is_dir():
                continue
            manual_name = emb_dir.name
            n = self._index_manual(collection, emb_dir, manual_name, store)
            summary[manual_name] = n

        total = sum(summary.values())
        log.info("ChromaDB: indexed %d chunks total across %d manuals", total, len(summary))
        log.info("ChromaDB: collection now has %d vectors", collection.count())
        return summary

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _index_manual(
        self,
        collection,
        emb_dir: Path,
        manual_name: str,
        store: ChunkStore,
    ) -> int:
        # Load embeddings
        emb_path = emb_dir / "embeddings.npy"
        ids_path = emb_dir / "chunk_ids.json"
        meta_path = emb_dir / "manual_meta.json"

        if not all(p.exists() for p in [emb_path, ids_path, meta_path]):
            log.warning("Skipping %s — missing embedding files", manual_name)
            return 0

        embeddings = np.load(str(emb_path)).astype(np.float32)
        chunk_ids: list[str] = json.loads(ids_path.read_text())
        run_meta: dict = json.loads(meta_path.read_text())
        document_id = run_meta.get("document_id", manual_name)

        log.info(
            "ChromaDB: indexing %s — %d vectors (shape %s)",
            manual_name,
            len(chunk_ids),
            embeddings.shape,
        )

        # Upsert in batches of 500 (ChromaDB recommended max)
        batch_size = 500
        indexed = 0
        seen_ids: set[str] = set()  # per-manual dedup guard (handles cross-batch duplicates too)

        for i in range(0, len(chunk_ids), batch_size):
            batch_chunk_ids = chunk_ids[i : i + batch_size]
            batch_embeddings = embeddings[i : i + batch_size]

            ids = []
            documents = []
            metadatas = []
            emb_list = []

            for chunk_id, emb in zip(batch_chunk_ids, batch_embeddings):
                try:
                    chunk = store.get_chunk(manual_name, chunk_id)
                except KeyError:
                    log.warning("Chunk %s/%s not in store — skipping", manual_name, chunk_id)
                    continue

                # Build globally unique ChromaDB ID
                chroma_id = f"{document_id}__{chunk_id}"

                # If this chunk_id is duplicated within the manual (chunker bug),
                # make it unique by appending reading_order_start
                if chroma_id in seen_ids:
                    suffix = chunk.get("reading_order_start", 0)
                    chroma_id = f"{chroma_id}__ro{suffix}"
                    log.debug(
                        "Duplicate chunk_id %s/%s — using dedup id %s",
                        manual_name, chunk_id, chroma_id,
                    )
                seen_ids.add(chroma_id)

                ids.append(chroma_id)
                documents.append(chunk.get("text", "")[:2000])  # ChromaDB stores first 2KB
                emb_list.append(emb.tolist())
                metadatas.append(self._build_metadata(chunk, manual_name, document_id))


            if ids:
                collection.upsert(
                    ids=ids,
                    documents=documents,
                    embeddings=emb_list,
                    metadatas=metadatas,
                )
                indexed += len(ids)

        log.info("ChromaDB: %s — %d chunks upserted", manual_name, indexed)
        return indexed

    @staticmethod
    def _build_metadata(chunk: dict, manual_name: str, document_id: str) -> dict:
        """Build the flat metadata dict stored in ChromaDB."""
        pages = chunk.get("pages") or []
        section_path = chunk.get("section_path") or []
        section_id = section_path[-1]["id"] if section_path else ""

        return {
            # Identity
            "chunk_id":        chunk["chunk_id"],
            "document_id":     document_id,
            "manual_name":     manual_name,

            # Structure
            "heading":         (chunk.get("heading") or "")[:256],  # ChromaDB string limit
            "section_id":      section_id,
            "pages":           ",".join(str(p) for p in pages),     # comma-joined (ChromaDB flat)
            "reading_order_start": chunk.get("reading_order_start", 0),
            "reading_order_end":   chunk.get("reading_order_end", 0),

            # Content flags (pre-filter before vector search)
            "has_figure":      chunk.get("has_figure", False),
            "has_table":       chunk.get("has_table", False),
            "has_admonition":  chunk.get("has_admonition", False),
            "token_estimate":  chunk.get("token_estimate", 0),
            "chunk_score":     float(chunk.get("chunk_score", 100.0)),

            # Embedding info
            "embedding_model": "gemini-embedding-2",
            "embedding_dims":  1536,
        }
