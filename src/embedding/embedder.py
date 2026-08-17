"""
src/embedding/embedder.py
─────────────────────────
SemanticEmbedder: loads chunks from all manuals and generates
text embeddings using gemini-embedding-2 with:

  • 5-key round-robin rotation
  • Adaptive batching (50 → 100 on success)
  • Exponential-backoff retry on 429 / 500 errors
  • Key rotation on per-key failure before global retry
  • Resume support: skips manuals already embedded
  • Saves per-manual embeddings to data/embeddings/{manual}/
"""
from __future__ import annotations

import json
import logging
import re
import time
from itertools import cycle
from pathlib import Path
from typing import Iterator

import numpy as np

from .config import EmbeddingConfig

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Key rotator
# ---------------------------------------------------------------------------

class _RpmBudget:
    """
    Sliding-window request counter.
    Limits total requests within any 60-second window to max_rpm.

    The free-tier quota for gemini-embedding-2 is:
      100 requests / minute / user / project / model
    Each Content object in batchEmbedContents counts as 1 request.
    So a batch of 50 chunks = 50 requests consumed.
    """

    def __init__(self, max_rpm: int = 90) -> None:  # 90 = safe margin below 100
        self._max_rpm = max_rpm
        self._window: list[float] = []  # timestamps of recent requests

    def consume(self, n: int) -> None:
        """
        Block until there is budget for `n` more requests.
        Removes timestamps older than 60s, then waits if needed.
        """
        while True:
            now = time.monotonic()
            # Drop timestamps outside the 60s window
            self._window = [t for t in self._window if now - t < 60.0]
            available = self._max_rpm - len(self._window)
            if available >= n:
                break
            need_to_wait = 60.0 - (now - self._window[0]) + 1.0  # +1s buffer
            log.info(
                "RPM budget: used %d/%d — waiting %.1fs for window to slide",
                len(self._window), self._max_rpm, need_to_wait,
            )
            time.sleep(max(need_to_wait, 1.0))

        # Record these n requests as consumed
        now = time.monotonic()
        self._window.extend([now] * n)


# ---------------------------------------------------------------------------
# Per-key RPM budget manager
# ---------------------------------------------------------------------------

class _KeyBudgetManager:
    """
    Manages 5 separate API keys, each with its own 100 RPM quota
    (5 accounts = 5 independent rate limits = ~500 RPM total).

    On each call, picks the key with the most available budget.
    If all keys are exhausted, blocks until any key has capacity.

    Each Content object in batchEmbedContents = 1 request consumed.
    We track 90/100 RPM per key as a safety margin.
    """

    RPM_PER_KEY = 90          # safe margin (actual limit = 100)
    WINDOW_S    = 60.0

    def __init__(self, keys: list[str]) -> None:
        self._keys = list(keys)
        # per-key sliding window of request timestamps
        self._windows: dict[str, list[float]] = {k: [] for k in self._keys}

    def acquire(self, n: int) -> str:
        """
        Block until some key has at least `n` budget available.
        Returns the key to use and records the consumption.
        """
        while True:
            now = time.monotonic()
            best_key = None
            best_avail = -1

            for key in self._keys:
                # Slide the window
                self._windows[key] = [
                    t for t in self._windows[key] if now - t < self.WINDOW_S
                ]
                avail = self.RPM_PER_KEY - len(self._windows[key])
                if avail >= n and avail > best_avail:
                    best_avail = avail
                    best_key = key

            if best_key is not None:
                # Consume budget on chosen key
                now = time.monotonic()
                self._windows[best_key].extend([now] * n)
                return best_key

            # All keys exhausted: find how long until the earliest slot frees up
            earliest = min(
                self._windows[k][0]
                for k in self._keys
                if self._windows[k]
            )
            wait = (earliest + self.WINDOW_S) - time.monotonic() + 1.0
            log.info(
                "All %d keys at capacity — waiting %.1fs for quota to free",
                len(self._keys), wait,
            )
            time.sleep(max(wait, 1.0))

    def release_back(self, key: str, n: int) -> None:
        """
        On a 429 error, remove the n timestamps we added for this key
        so we don't double-penalise the budget.
        """
        if self._windows[key]:
            self._windows[key] = self._windows[key][:-n]

    def available(self) -> dict[str, int]:
        """Return available budget per key (for logging)."""
        now = time.monotonic()
        return {
            k: self.RPM_PER_KEY - len([t for t in self._windows[k] if now - t < self.WINDOW_S])
            for k in self._keys
        }


# ---------------------------------------------------------------------------
# SemanticEmbedder
# ---------------------------------------------------------------------------

class SemanticEmbedder:
    """
    Embeds all chunks across all manuals using gemini-embedding-2.

    Output per manual (data/embeddings/{manual_name}/):
        embeddings.npy       – float32 array (N_chunks × 1536)
        chunk_ids.json       – ordered list of chunk_ids
        manual_meta.json     – document_id, model, timestamp, dims
    """

    def __init__(self, config: EmbeddingConfig | None = None) -> None:
        self._cfg = config or EmbeddingConfig()
        if not self._cfg.api_keys:
            raise ValueError("No API keys configured. Set GEMINI_API_KEYS before embedding.")
        self._budget = _KeyBudgetManager(self._cfg.api_keys)
        self._current_batch_size = self._cfg.initial_batch_size
        self._consecutive_successes = 0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def embed_all(self) -> dict[str, int]:
        """
        Embed all manuals found in cfg.chunks_dir.
        Returns {manual_name: chunks_embedded} summary dict.
        """
        chunks_dir = Path(self._cfg.chunks_dir)
        manual_dirs = sorted([d for d in chunks_dir.iterdir() if d.is_dir()])

        if not manual_dirs:
            raise FileNotFoundError(f"No manual directories found in {chunks_dir}")

        summary: dict[str, int] = {}
        for manual_dir in manual_dirs:
            name = manual_dir.name
            manifest_path = manual_dir / "chunks.json"
            if not manifest_path.exists():
                log.warning("Skipping %s — no chunks.json", name)
                continue

            out_dir = Path(self._cfg.embeddings_dir) / name
            if self._is_already_embedded(out_dir):
                log.info("Skipping %s — already embedded", name)
                summary[name] = self._load_existing_count(out_dir)
                continue

            log.info("=" * 60)
            log.info("Embedding: %s", name)
            log.info("=" * 60)
            n = self.embed_manual(manifest_path, out_dir, name)
            summary[name] = n

        return summary

    def embed_manual(
        self,
        manifest_path: Path,
        output_dir: Path,
        manual_name: str,
    ) -> int:
        """Embed one manual. Returns number of chunks embedded."""
        manifest = json.load(open(manifest_path, encoding="utf-8"))
        chunks = manifest["chunks"]
        document_id = manifest.get("document_id", "")

        log.info("%s: %d chunks to embed", manual_name, len(chunks))

        # Prepare text inputs (asymmetric doc format)
        texts = [self._format_doc(c) for c in chunks]
        chunk_ids = [c["chunk_id"] for c in chunks]

        # Embed in batches
        all_embeddings: list[list[float]] = []
        total = len(texts)
        pos = 0

        while pos < total:
            batch_texts = texts[pos : pos + self._current_batch_size]
            batch_ids = chunk_ids[pos : pos + self._current_batch_size]

            log.info(
                "  Batch %d–%d / %d  (batch_size=%d)",
                pos + 1,
                min(pos + self._current_batch_size, total),
                total,
                self._current_batch_size,
            )

            batch_embeddings = self._embed_batch_with_retry(batch_texts)
            all_embeddings.extend(batch_embeddings)
            pos += self._current_batch_size

            # Adaptive batch size ramp
            self._consecutive_successes += 1
            if (
                self._current_batch_size < self._cfg.max_batch_size
                and self._consecutive_successes >= self._cfg.batch_size_ramp_after
            ):
                self._current_batch_size = self._cfg.max_batch_size
                log.info("  Ramped batch size to %d", self._current_batch_size)

            # Polite delay between batches
            if pos < total:
                time.sleep(self._cfg.inter_batch_delay_s)

        # Persist
        output_dir.mkdir(parents=True, exist_ok=True)
        self._save(output_dir, chunk_ids, all_embeddings, document_id, manual_name)

        log.info("%s: saved %d embeddings → %s", manual_name, len(all_embeddings), output_dir)
        return len(all_embeddings)

    # ------------------------------------------------------------------
    # Internal: formatting
    # ------------------------------------------------------------------

    def _format_doc(self, chunk: dict) -> str:
        """Format chunk as asymmetric retrieval document."""
        heading = chunk.get("heading") or "none"
        text = chunk.get("text") or ""

        # Truncate if over budget
        words = text.split()
        if len(words) > self._cfg.max_text_words:
            text = " ".join(words[: self._cfg.max_text_words])
            log.debug("Truncated chunk %s to %d words", chunk.get("chunk_id"), self._cfg.max_text_words)

        return self._cfg.doc_prefix_template.format(heading=heading, text=text)

    # ------------------------------------------------------------------
    # Internal: embedding with retry + key rotation
    # ------------------------------------------------------------------

    def _embed_batch_with_retry(self, texts: list[str]) -> list[list[float]]:
        """
        Embed one batch:
          1. Acquire best available key (per-key RPM budget manager)
          2. Call API with that key
          3. On 429: release budget back, parse retryDelay, wait, retry with different key
        """
        for attempt in range(1, self._cfg.max_retries + 1):
            # Pick the key with most available budget
            api_key = self._budget.acquire(len(texts))

            avail = self._budget.available()
            key_idx = self._cfg.api_keys.index(api_key) + 1
            log.debug("Using key %d, budget per key: %s", key_idx, avail)

            try:
                result = self._call_api(api_key, texts)
                # Small polite delay
                time.sleep(self._cfg.inter_batch_delay_s)
                return result
            except Exception as exc:
                err_str = str(exc)
                is_rate_limit = "429" in err_str or "quota" in err_str.lower() or "rate" in err_str.lower()
                is_server_err = "500" in err_str or "503" in err_str

                if attempt == self._cfg.max_retries:
                    log.error("Batch failed after %d attempts: %s", attempt, exc)
                    raise

                if is_rate_limit or is_server_err:
                    self._consecutive_successes = 0

                    # Release the budget back for this key (it didn't actually serve the requests)
                    self._budget.release_back(api_key, len(texts))

                    retry_s = self._parse_retry_delay(err_str)
                    wait = min(max(retry_s + 5.0, self._cfg.retry_base_delay_s), self._cfg.retry_max_delay_s)

                    log.warning(
                        "Attempt %d/%d: key-%d hit %s. API retry-after=%.0fs, waiting %.1fs…",
                        attempt, self._cfg.max_retries, key_idx,
                        "rate-limit" if is_rate_limit else "server-error",
                        retry_s, wait,
                    )
                    time.sleep(wait)
                else:
                    log.error("Non-retryable error: %s", exc)
                    raise

        raise RuntimeError("Unreachable")  # pragma: no cover

    @staticmethod
    def _parse_retry_delay(err_str: str) -> float:
        """
        Extract retry delay in seconds from the 429 error message.
        The API includes e.g. 'Please retry in 31.6s' and 'retryDelay: 31s'.
        Returns 35.0 as a safe default if not parseable.
        """
        # Try 'Please retry in X.Xs'
        m = re.search(r'retry in (\d+\.?\d*)', err_str, re.IGNORECASE)
        if m:
            return float(m.group(1))
        # Try 'retryDelay.*Xs'
        m = re.search(r"retryDelay[\"']?:\s*[\"']?(\d+)", err_str)
        if m:
            return float(m.group(1))
        return 35.0  # safe default (just over 31s window)

    def _call_api(self, api_key: str, texts: list[str]) -> list[list[float]]:
        """
        Single API call to gemini-embedding-2.

        IMPORTANT: Passing a plain list of strings to `contents` causes the SDK
        to treat them as parts of ONE aggregated embedding (1 output for N texts).
        To get N separate embeddings for N texts, each text must be wrapped in
        its own Content object.
        """
        from google import genai
        from google.genai import types as T

        client = genai.Client(api_key=api_key)

        # Wrap each text as its own Content → N inputs → N embeddings
        contents = [T.Content(parts=[T.Part(text=t)]) for t in texts]

        result = client.models.embed_content(
            model=self._cfg.model,
            contents=contents,
            config=T.EmbedContentConfig(
                output_dimensionality=self._cfg.output_dimensionality,
            ),
        )

        if len(result.embeddings) != len(texts):
            raise RuntimeError(
                f"API returned {len(result.embeddings)} embeddings for {len(texts)} inputs. "
                "Unexpected SDK behaviour."
            )

        return [list(e.values) for e in result.embeddings]

    # ------------------------------------------------------------------
    # Internal: persistence
    # ------------------------------------------------------------------

    def _save(
        self,
        output_dir: Path,
        chunk_ids: list[str],
        embeddings: list[list[float]],
        document_id: str,
        manual_name: str,
    ) -> None:
        arr = np.array(embeddings, dtype=np.float32)
        np.save(output_dir / "embeddings.npy", arr)

        (output_dir / "chunk_ids.json").write_text(
            json.dumps(chunk_ids, indent=2), encoding="utf-8"
        )

        meta = {
            "manual_name": manual_name,
            "document_id": document_id,
            "model": self._cfg.model,
            "output_dimensionality": self._cfg.output_dimensionality,
            "num_chunks": len(chunk_ids),
            "embedding_shape": list(arr.shape),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        (output_dir / "manual_meta.json").write_text(
            json.dumps(meta, indent=2), encoding="utf-8"
        )

    def _is_already_embedded(self, output_dir: Path) -> bool:
        return (
            (output_dir / "embeddings.npy").exists()
            and (output_dir / "chunk_ids.json").exists()
            and (output_dir / "manual_meta.json").exists()
        )

    def _load_existing_count(self, output_dir: Path) -> int:
        try:
            meta = json.loads((output_dir / "manual_meta.json").read_text())
            return meta.get("num_chunks", 0)
        except Exception:
            return 0
