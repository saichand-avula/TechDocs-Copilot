"""
src/retrieval/reranker.py
──────────────────────────
LLM reranker using Sarvam sarvam-105b.

Input : up to 8 candidate chunks (metadata from ChromaDB/BM25)
Output: list of (chunk_meta, normalized_score) sorted desc by score

Score format: LLM returns integer 1–10 per chunk.
  - 1  = completely irrelevant
  - 10 = directly answers the query
Normalized to 0.0–1.0 by dividing by 10.

Why 1–10 integers, not 0.0–1.0 floats:
  LLMs produce poorly calibrated continuous probabilities.
  Ordinal integer judgment is more stable and consistent.
"""
from __future__ import annotations

import json
import logging
import re

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_RANKER_PROMPT = """\
You are a document relevance judge for a technical manual Q&A system.

Query: {query}

Below are document chunks retrieved for this query.
For each chunk, return a relevance score from 1 (completely irrelevant) \
to 10 (directly contains the answer).

Scoring guide:
  9–10 : chunk directly and fully answers the query
  7–8  : chunk is closely related and partially answers
  5–6  : chunk is topically relevant but doesn't answer directly
  3–4  : chunk is distantly related
  1–2  : chunk is irrelevant

Be strict. Only give high scores to chunks that CONTAIN the actual answer.

Chunks:
{formatted_chunks}

Return ONLY a JSON array, one object per chunk, in the SAME order as above:
[{{"id": 1, "score": 8}}, {{"id": 2, "score": 3}}, ...]
No explanation. JSON only.\
"""


# ---------------------------------------------------------------------------
# Chunk snippet formatter
# ---------------------------------------------------------------------------

def _format_snippet(idx: int, meta: dict, chunk: dict) -> str:
    """
    Build a structured 800–1000 char snippet for the reranker.
    Includes: manual, section path, pages, heading, text, table preview.
    """
    manual  = meta.get("manual_name", "")
    heading = meta.get("heading", "—")
    pages   = meta.get("pages", "")
    section = meta.get("section_id", "")

    text  = (chunk.get("text") or "")[:700]
    tables = chunk.get("tables") or []
    table_preview = ""
    if tables:
        first_table = tables[0].get("markdown", "")
        # First 3 rows of table (split by newline)
        rows = [r for r in first_table.split("\n") if r.strip()][:4]
        table_preview = "\nTable preview:\n" + "\n".join(rows)

    return (
        f"[{idx}]\n"
        f"Manual  : {manual}\n"
        f"Section : {section}\n"
        f"Pages   : {pages}\n"
        f"Heading : {heading}\n"
        f"Text    : {text}"
        f"{table_preview}"
    )


# ---------------------------------------------------------------------------
# Main reranker
# ---------------------------------------------------------------------------

def rerank(
    query:      str,
    candidates: list[dict],   # list of ChromaDB metadata dicts
    store,                    # ChunkStore
    config,
) -> list[tuple[dict, float]]:
    """
    Rank up to 8 candidates by relevance to query.

    Returns: list of (chunk_meta, normalized_score 0.0–1.0), sorted desc.
    Falls back to original order with uniform scores on LLM failure.
    """
    if not candidates:
        return []

    # Build structured snippets
    snippets = []
    for i, meta in enumerate(candidates, 1):
        try:
            chunk = store.get_chunk(meta["manual_name"], meta["chunk_id"])
        except KeyError:
            chunk = {}
        snippets.append(_format_snippet(i, meta, chunk))

    formatted = "\n\n".join(snippets)
    prompt = _RANKER_PROMPT.format(query=query, formatted_chunks=formatted)

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.sarvam_api_key,
            base_url=config.sarvam_base_url,
        )
        response = client.chat.completions.create(
            model=config.sarvam_model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,   # sarvam-105b is a reasoning model: needs budget for CoT
            temperature=0,
        )
        # sarvam-105b: answer is in content AFTER reasoning phase completes.
        # If content is None the model ran out of tokens mid-reasoning.
        raw = (
            response.choices[0].message.content
            or response.choices[0].message.reasoning_content
            or ""
        ).strip()
        scores_data = _parse_scores(raw, len(candidates))

        # Build (meta, normalized_score) pairs
        id_to_meta = {i + 1: meta for i, meta in enumerate(candidates)}
        result = [
            (id_to_meta[item["id"]], item["score"] / 10.0)
            for item in scores_data
            if item["id"] in id_to_meta
        ]
        result.sort(key=lambda x: x[1], reverse=True)

        log.info(
            "Reranker: %d candidates → scores %s",
            len(candidates),
            [f"{s:.1f}" for _, s in result],
        )
        return result

    except Exception as exc:
        log.warning("Reranker failed (%s) — returning original order with 0.6 score", exc)
        # Fallback: return original order, all above threshold
        return [(meta, 0.6) for meta in candidates]


# ---------------------------------------------------------------------------
# JSON parsing (robust)
# ---------------------------------------------------------------------------

def _parse_scores(raw: str, n_expected: int) -> list[dict]:
    """
    Parse LLM output to list of {id, score} dicts.
    Tries strict JSON first, then regex extraction as fallback.
    """
    # Strip markdown fences if present
    clean = re.sub(r"```(?:json)?|```", "", raw).strip()

    try:
        data = json.loads(clean)
        if isinstance(data, list) and all("id" in d and "score" in d for d in data):
            # Clamp scores to valid range
            return [{"id": d["id"], "score": max(1, min(10, int(d["score"])))} for d in data]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass

    # Regex fallback: extract all {"id": N, "score": M} patterns
    pattern = re.compile(r'"id"\s*:\s*(\d+).*?"score"\s*:\s*(\d+)', re.DOTALL)
    matches = pattern.findall(clean)
    if matches:
        return [{"id": int(m[0]), "score": max(1, min(10, int(m[1])))} for m in matches]

    # Last resort: return uniform scores in original order
    log.warning("Could not parse reranker output: %r", raw[:200])
    return [{"id": i + 1, "score": 6} for i in range(n_expected)]
