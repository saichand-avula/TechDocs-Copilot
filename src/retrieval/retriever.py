"""
src/retrieval/retriever.py
───────────────────────────
Main retrieval orchestrator. One public function: retrieve().

Flow:
  Query Analyzer
    │
    ├─ PAGE_METADATA  → ChunkStore page scan → generate (1 API call)
    │
    ├─ EXACT_CODE     → Dense + BM25 → RRF → rerank → threshold → generate
    │
    ├─ NORMAL         → Dense + BM25 → RRF → rerank → threshold → generate
    │
    └─ VAGUE          → HyDE → Dense + BM25 → RRF → rerank → threshold → generate

Gemini embedding: same 5-key rotation used in Stage 4.
Sarvam: HyDE, reranker, generator.
"""
from __future__ import annotations

import json
import logging
import pickle
import time
from itertools import cycle
from pathlib import Path

import numpy as np

from .config import RetrieverConfig
from .query_analyzer import ParsedQuery, QueryType, analyze_query
from .hyde import expand_with_hyde
from .reranker import rerank
from .answer_builder import RetrievalResult, generate_answer

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Simple round-robin key rotation for query-time embedding (1 call per query)
# ---------------------------------------------------------------------------

class _QueryKeyRotator:
    def __init__(self, keys: list[str]) -> None:
        self._iter = cycle(keys)

    def next(self) -> str:
        return next(self._iter)


# ---------------------------------------------------------------------------
# Retriever class
# ---------------------------------------------------------------------------

class Retriever:
    """
    Stateful retriever: loads ChromaDB collection, BM25 index, and ChunkStore
    once at construction, then serves queries.

    Usage:
        r = Retriever(config, store)
        result = r.retrieve("how to replace the toner cartridge in printer?")
        print(result.answer)
    """

    def __init__(self, config: RetrieverConfig, store) -> None:
        self._cfg   = config
        self._store = store
        self._key_rotator = _QueryKeyRotator(config.gemini_api_keys)

        # Load ChromaDB
        import chromadb
        chroma_path = config.vectordb_dir / "chroma"
        client = chromadb.PersistentClient(path=str(chroma_path))
        self._collection = client.get_collection(config.collection_name)
        log.info("Retriever: ChromaDB collection '%s' (%d vectors)",
                 config.collection_name, self._collection.count())

        # Load BM25
        bm25_path = config.bm25_dir / "corpus_index.pkl"
        self._bm25 = pickle.load(open(bm25_path, "rb"))
        self._corpus_map: list[dict] = json.loads(
            (config.bm25_dir / "corpus_map.json").read_text()
        )
        log.info("Retriever: BM25 loaded (%d docs)", len(self._corpus_map))

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def retrieve(self, query: str) -> RetrievalResult:
        """
        Main entry point. Returns a RetrievalResult with answer + citations.
        """
        cfg = self._cfg
        parsed = analyze_query(query)

        log.info("Query type: %s | manual: %s | page: %s | query: %r",
                 parsed.query_type.value, parsed.manual_name, parsed.page, query[:60])

        # ── PATH A: Page / metadata lookup ─────────────────────────────────
        if parsed.query_type == QueryType.PAGE_METADATA:
            return self._page_path(query, parsed)

        # ── PATH B: Semantic paths (EXACT_CODE / NORMAL / VAGUE) ───────────
        return self._semantic_path(query, parsed)

    # ------------------------------------------------------------------
    # Path A: Page lookup (ChunkStore only, 0 embedding calls)
    # ------------------------------------------------------------------

    def _page_path(self, query: str, parsed: ParsedQuery) -> RetrievalResult:
        page = parsed.page
        manual = parsed.manual_name

        # Scan in-memory ChunkStore
        if manual:
            candidates = self._store.get_all_chunks(manual)
        else:
            candidates = self._store.get_corpus()

        page_chunks = [
            c for c in candidates
            if page in (c.get("pages") or [])
        ]

        if not page_chunks:
            log.info("Page %d not found in ChunkStore — falling back to Chroma filter", page)
            page_chunks = self._chroma_page_fallback(page, manual)

        if not page_chunks:
            return RetrievalResult(
                answer="Not found in the provided documents.",
                sources=[], figures=[], query_type=QueryType.PAGE_METADATA.value, chunks_used=0,
            )

        log.info("Page %d: found %d chunks", page, len(page_chunks))
        # Tag _manual_name for answer_builder
        for c in page_chunks:
            if "_manual_name" not in c:
                c["_manual_name"] = manual or ""

        scores = [1.0] * len(page_chunks)  # exact match, all score max
        return generate_answer(
            query, page_chunks[:self._cfg.generator_top_k],
            scores[:self._cfg.generator_top_k],
            QueryType.PAGE_METADATA.value, self._cfg,
        )

    def _chroma_page_fallback(self, page: int, manual: str | None) -> list[dict]:
        """Chroma metadata filter fallback for page lookup."""
        where: dict = {"pages": {"$contains": str(page)}}
        if manual:
            where = {"$and": [where, {"manual_name": manual}]}
        try:
            results = self._collection.query(
                query_embeddings=[[0.0] * self._cfg.embedding_dims],  # dummy
                n_results=10,
                where=where,
                include=["metadatas"],
            )
            metas = results["metadatas"][0] if results["metadatas"] else []
            return [self._store.get_chunk(m["manual_name"], m["chunk_id"]) for m in metas]
        except Exception as exc:
            log.warning("Chroma page fallback failed: %s", exc)
            return []

    # ------------------------------------------------------------------
    # Path B: Semantic retrieval
    # ------------------------------------------------------------------

    def _semantic_path(self, query: str, parsed: ParsedQuery) -> RetrievalResult:
        cfg = self._cfg

        # Step 1: Embed query (with HyDE if vague and enabled)
        embed_input = query
        if parsed.query_type == QueryType.VAGUE and cfg.use_hyde:
            embed_input = expand_with_hyde(query, cfg)

        embed_text = f"task: search result | query: {embed_input}"
        query_vec = self._embed_query(embed_text)

        # Step 2: Dense search (Chroma)
        dense_hits = self._dense_search(query_vec, parsed.manual_name)

        # Step 3: BM25 search
        bm25_hits = self._bm25_search(query, parsed.manual_name)

        # Step 4: RRF fusion → top-8
        rrf_hits = self._rrf_fuse(dense_hits, bm25_hits)

        log.info("RRF: %d candidates after fusion", len(rrf_hits))

        # Step 5: LLM rerank → scored list
        scored = rerank(query, rrf_hits, self._store, cfg)

        # Step 6: Relevance threshold filter
        passing = [
            (meta, score) for meta, score in scored
            if score >= cfg.relevance_threshold
        ]
        passing = passing[:cfg.generator_top_k]

        if not passing:
            log.info("All chunks below threshold %.2f — returning 'Not found'", cfg.relevance_threshold)
            return RetrievalResult(
                answer="Not found in the provided documents.",
                sources=[], figures=[], query_type=parsed.query_type.value, chunks_used=0,
            )

        log.info("Passing threshold: %d chunks (scores: %s)",
                 len(passing), [f"{s:.2f}" for _, s in passing])

        # Step 7: Load full chunks from ChunkStore
        full_chunks = []
        final_scores = []
        for meta, score in passing:
            try:
                chunk = self._store.get_chunk(meta["manual_name"], meta["chunk_id"])
                chunk["_manual_name"] = meta["manual_name"]
                full_chunks.append(chunk)
                final_scores.append(score)
            except KeyError as exc:
                log.warning("Could not load chunk %s: %s", meta, exc)

        # Step 8: Generate answer
        return generate_answer(
            query, full_chunks, final_scores,
            parsed.query_type.value, cfg,
        )

    # ------------------------------------------------------------------
    # Internal: embedding
    # ------------------------------------------------------------------

    def _embed_query(self, text: str) -> list[float]:
        """Embed a single query text using Gemini embedding-2 with key rotation."""
        from google import genai
        from google.genai import types as T

        for attempt in range(len(self._cfg.gemini_api_keys) + 1):
            api_key = self._key_rotator.next()
            try:
                client = genai.Client(api_key=api_key)
                result = client.models.embed_content(
                    model=self._cfg.embedding_model,
                    contents=[T.Content(parts=[T.Part(text=text)])],
                    config=T.EmbedContentConfig(
                        output_dimensionality=self._cfg.embedding_dims
                    ),
                )
                return list(result.embeddings[0].values)
            except Exception as exc:
                if "429" in str(exc) and attempt < len(self._cfg.gemini_api_keys):
                    log.warning("Embed key hit 429 — rotating to next key")
                    time.sleep(1.0)
                else:
                    raise

        raise RuntimeError("All Gemini keys exhausted during query embedding")

    # ------------------------------------------------------------------
    # Internal: dense search
    # ------------------------------------------------------------------

    def _dense_search(self, query_vec: list[float], manual: str | None) -> list[dict]:
        """Chroma dense search. Returns list of metadata dicts."""
        where = None
        if manual:
            where = {"manual_name": manual}

        kwargs = dict(
            query_embeddings=[query_vec],
            n_results=self._cfg.dense_candidates,
            include=["metadatas", "distances"],
        )
        if where:
            kwargs["where"] = where

        results = self._collection.query(**kwargs)
        metas = results["metadatas"][0] if results["metadatas"] else []
        log.debug("Dense: %d hits", len(metas))
        return metas

    # ------------------------------------------------------------------
    # Internal: BM25 search
    # ------------------------------------------------------------------

    def _bm25_search(self, query: str, manual: str | None) -> list[dict]:
        """BM25 search. Returns list of corpus_map dicts."""
        tokens = query.lower().split()
        scores = self._bm25.get_scores(tokens)

        # Score all, filter by manual if specified
        indexed = [
            (score, entry)
            for score, entry in zip(scores, self._corpus_map)
            if score > 0 and (manual is None or entry["manual_name"] == manual)
        ]
        indexed.sort(key=lambda x: x[0], reverse=True)
        top = [entry for _, entry in indexed[: self._cfg.bm25_candidates]]

        log.debug("BM25: %d hits", len(top))
        return top

    # ------------------------------------------------------------------
    # Internal: RRF fusion
    # ------------------------------------------------------------------

    def _rrf_fuse(
        self,
        dense_hits: list[dict],
        bm25_hits:  list[dict],
    ) -> list[dict]:
        """
        Reciprocal Rank Fusion.
        Returns top-cfg.rrf_candidates unique (manual_name, chunk_id) metas.
        """
        k = self._cfg.rrf_k
        scores: dict[tuple, float] = {}
        meta_map: dict[tuple, dict] = {}

        for rank, meta in enumerate(dense_hits, 1):
            key = (meta["manual_name"], meta["chunk_id"])
            scores[key]   = scores.get(key, 0.0) + 1.0 / (k + rank)
            meta_map[key] = meta   # keep Chroma metadata (richer)

        for rank, entry in enumerate(bm25_hits, 1):
            key = (entry["manual_name"], entry["chunk_id"])
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            if key not in meta_map:
                # BM25-only hit: build minimal metadata dict
                meta_map[key] = {
                    "manual_name": entry["manual_name"],
                    "chunk_id":    entry["chunk_id"],
                    "heading":     entry.get("heading", ""),
                    "pages":       "",
                    "section_id":  "",
                }

        sorted_keys = sorted(scores, key=lambda k: scores[k], reverse=True)
        return [meta_map[key] for key in sorted_keys[: self._cfg.rrf_candidates]]
