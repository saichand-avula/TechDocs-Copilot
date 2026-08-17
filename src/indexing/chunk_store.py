"""
src/indexing/chunk_store.py
───────────────────────────
ChunkStore: abstracts chunk retrieval from the retriever.
Storage backend today = JSON files. Nothing outside this module
touches the file system for chunk data.

IMPORTANT: chunk_ids are NOT globally unique.
  chunk_0001 exists in all 11 manuals.
  Every lookup requires (manual_name, chunk_id) as a pair.
  get_chunk(manual_name, chunk_id)  ←  always use this signature.
"""
from __future__ import annotations

import json
import logging
from pathlib import Path

log = logging.getLogger(__name__)


class ChunkStore:
    """
    In-memory-cached chunk store backed by chunks.json files.

    After first access for a manual, all chunks for that manual
    are held in memory as a dict keyed by chunk_id → chunk dict.
    Subsequent lookups are O(1).

    All 11 manifests warmed ≈ 35 MB in memory — negligible.
    """

    def __init__(self, chunks_dir: str | Path = "data/chunks") -> None:
        self._root = Path(chunks_dir)
        self._cache: dict[str, dict[str, dict]] = {}   # manual_name → {chunk_id → chunk}
        self._manifests: dict[str, dict] = {}          # manual_name → full manifest

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_chunk(self, manual_name: str, chunk_id: str) -> dict:
        """
        Return a single chunk dict.

        Raises KeyError if manual_name or chunk_id is unknown.
        """
        self._ensure_loaded(manual_name)
        try:
            return self._cache[manual_name][chunk_id]
        except KeyError:
            raise KeyError(
                f"chunk_id '{chunk_id}' not found in manual '{manual_name}'. "
                f"Available chunk count: {len(self._cache.get(manual_name, {}))}"
            )

    def get_all_chunks(self, manual_name: str) -> list[dict]:
        """
        Return all chunks for a manual (used for BM25 corpus building).
        Returned in reading order (as stored in chunks.json).
        """
        self._ensure_loaded(manual_name)
        return list(self._cache[manual_name].values())

    def get_corpus(self) -> list[dict]:
        """
        Return all chunks across ALL manuals.
        Loads any not-yet-cached manuals on demand.
        """
        self._load_all()
        corpus: list[dict] = []
        for manual_name in sorted(self._cache):
            corpus.extend(self._cache[manual_name].values())
        return corpus

    def list_manuals(self) -> list[str]:
        """Return sorted list of manual names found in chunks_dir."""
        return sorted(
            d.name
            for d in self._root.iterdir()
            if d.is_dir() and (d / "chunks.json").exists()
        )

    def get_document_id(self, manual_name: str) -> str:
        """Return the document_id for a manual."""
        self._ensure_loaded(manual_name)
        return self._manifests[manual_name].get("document_id", "")

    def warm(self) -> None:
        """Pre-load all manuals into memory (call once at startup)."""
        self._load_all()
        total = sum(len(v) for v in self._cache.values())
        log.info("ChunkStore warmed: %d manuals, %d chunks total", len(self._cache), total)

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _ensure_loaded(self, manual_name: str) -> None:
        if manual_name in self._cache:
            return
        path = self._root / manual_name / "chunks.json"
        if not path.exists():
            raise KeyError(
                f"Manual '{manual_name}' not found at {path}. "
                f"Available: {self.list_manuals()}"
            )
        self._load_manifest(manual_name, path)

    def _load_manifest(self, manual_name: str, path: Path) -> None:
        log.debug("ChunkStore: loading %s", path)
        manifest = json.load(open(path, encoding="utf-8"))
        self._manifests[manual_name] = manifest
        # Index chunks by chunk_id for O(1) access
        # Preserve original list order so get_all_chunks() returns reading order
        self._cache[manual_name] = {
            c["chunk_id"]: c for c in manifest["chunks"]
        }
        log.debug(
            "ChunkStore: %s loaded — %d chunks", manual_name, len(self._cache[manual_name])
        )

    def _load_all(self) -> None:
        for name in self.list_manuals():
            if name not in self._cache:
                path = self._root / name / "chunks.json"
                self._load_manifest(name, path)
