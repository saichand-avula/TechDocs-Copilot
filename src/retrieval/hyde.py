"""
src/retrieval/hyde.py
──────────────────────
HyDE (Hypothetical Document Embeddings) — used only for VAGUE queries.

Sends the vague query to Sarvam LLM → gets a hypothetical technical passage
→ embeds THAT passage instead of the raw query → much better retrieval signal.

NOT used for: exact codes, page lookups, normal queries.
"""
from __future__ import annotations

import logging

log = logging.getLogger(__name__)

_HYDE_PROMPT = (
    "You are a technical documentation assistant.\n"
    "Write a short technical documentation passage (2–3 sentences) "
    "that would DIRECTLY answer this question or describe this topic.\n"
    "Use precise technical language. Be specific.\n\n"
    "Question/Topic: {query}\n\n"
    "Passage:"
)


def expand_with_hyde(query: str, config) -> str:
    """
    Generate a hypothetical document for the given vague query.
    Returns the hypothetical text (to be embedded, not the original query).

    If Sarvam call fails, falls back to the original query (graceful degradation).
    """
    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.sarvam_api_key,
            base_url=config.sarvam_base_url,
        )
        response = client.chat.completions.create(
            model=config.sarvam_model,
            messages=[
                {"role": "user", "content": _HYDE_PROMPT.format(query=query)}
            ],
            max_tokens=1500,   # sarvam-105b reasoning model: needs room for CoT + output
            temperature=0.3,
        )
        # sarvam-105b: answer in content after reasoning completes
        hypothesis = (
            response.choices[0].message.content
            or response.choices[0].message.reasoning_content
            or ""
        ).strip()
        log.info("HyDE expansion: %r → %r", query, hypothesis[:80])
        return hypothesis

    except Exception as exc:
        log.warning("HyDE failed (%s) — falling back to raw query", exc)
        return query  # graceful fallback: embed the original query
