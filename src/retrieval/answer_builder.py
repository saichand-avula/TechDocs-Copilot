"""
src/retrieval/answer_builder.py
─────────────────────────────────
Assembles context from passing chunks and calls Sarvam LLM for generation.
Returns RetrievalResult with answer text, inline citations, sources, and figure paths.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class FigureInfo:
    fig_id:     str
    image_path: str
    caption:    str | None = None
    page:       int | None = None


@dataclass
class RetrievalResult:
    answer:      str
    sources:     list[dict]       = field(default_factory=list)
    figures:     list[FigureInfo] = field(default_factory=list)
    query_type:  str              = "normal"
    chunks_used: int              = 0

    def __str__(self) -> str:
        return self.answer


# ---------------------------------------------------------------------------
# System prompt
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a precise technical documentation assistant for industrial and consumer equipment manuals.

Rules:
1. Answer using ONLY the information in the provided context chunks.
2. Cite every factual claim with the chunk number in brackets: [1], [2], etc.
3. Format numbered procedures as numbered steps.
4. If a table contains the answer, read it carefully and cite it.
5. If the answer is not present in the context, respond exactly:
   "Not found in the provided documents."
6. Never invent part numbers, procedures, or specifications.
7. Be concise but complete. Do not pad.\
"""

# ---------------------------------------------------------------------------
# Context assembler
# ---------------------------------------------------------------------------

def _assemble_context(chunks: list[dict], scores: list[float]) -> tuple[str, list[dict], list[str]]:
    """
    Build the numbered context string, source list, and figure paths.

    chunks: full chunk dicts from ChunkStore (with _manual_name tagged)
    scores: normalized relevance scores (0–1), same order as chunks
    """
    parts:   list[str]  = []
    sources: list[dict] = []
    figures: list[str]  = []

    for i, (chunk, score) in enumerate(zip(chunks, scores), 1):
        manual  = chunk.get("_manual_name", chunk.get("manual_name", ""))
        heading = chunk.get("heading") or "—"
        pages   = chunk.get("pages") or []
        text    = (chunk.get("text") or "")[:1500]

        # Include tables (first table full markdown, up to 500 chars)
        tables = chunk.get("tables") or []
        table_str = ""
        if tables:
            md = tables[0].get("markdown", "")[:500]
            if md:
                table_str = f"\n\nTable:\n{md}"

        pages_str = ", ".join(str(p) for p in pages[:4])

        part = (
            f"[{i}] {manual} | {heading} | Pages: {pages_str}\n"
            f"{text}"
            f"{table_str}"
        )
        parts.append(part)

        sources.append({
            "citation_num": i,
            "manual_name":  manual,
            "heading":      heading,
            "pages":        pages,
            "chunk_id":     chunk.get("chunk_id", ""),
            "score":        round(score, 2),
        })

        # Collect unique figures (each is a dict: id/image_path/caption/page)
        seen_paths: set[str] = {f.image_path for f in figures}
        for fig in (chunk.get("figures") or []):
            if isinstance(fig, dict):
                path = fig.get("image_path") or ""
                if path and path not in seen_paths:
                    figures.append(FigureInfo(
                        fig_id     = fig.get("id") or fig.get("figure_number") or path,
                        image_path = path,
                        caption    = fig.get("caption") or None,
                        page       = fig.get("page") or None,
                    ))
                    seen_paths.add(path)
            elif isinstance(fig, str) and fig not in seen_paths:
                # Fallback: bare path string
                figures.append(FigureInfo(fig_id=fig, image_path=fig))
                seen_paths.add(fig)

    context = "\n\n---\n\n".join(parts)
    return context, sources, figures


# ---------------------------------------------------------------------------
# Citation footer
# ---------------------------------------------------------------------------

def _build_citation_footer(sources: list[dict]) -> str:
    lines = ["\n\nSources:"]
    for s in sources:
        pages = s.get("pages") or []
        if len(pages) >= 2:
            pages_str = f"{pages[0]}–{pages[-1]}"
        elif pages:
            pages_str = str(pages[0])
        else:
            pages_str = "—"
        lines.append(
            f'[{s["citation_num"]}] {s["manual_name"]} — '
            f'"{s["heading"]}" — Pages {pages_str}'
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main generation call
# ---------------------------------------------------------------------------

def generate_answer(
    query:      str,
    chunks:     list[dict],     # full chunks from ChunkStore
    scores:     list[float],    # normalized scores, same order
    query_type: str,
    config,
) -> RetrievalResult:
    """
    Assemble context from chunks and generate a grounded answer via Sarvam.
    """
    if not chunks:
        return RetrievalResult(
            answer="Not found in the provided documents.",
            sources=[], figures=[], query_type=query_type, chunks_used=0,
        )

    context, sources, figures = _assemble_context(chunks, scores)

    user_prompt = f"Context:\n{context}\n\nQuestion: {query}"

    try:
        from openai import OpenAI
        client = OpenAI(
            api_key=config.sarvam_api_key,
            base_url=config.sarvam_base_url,
        )
        response = client.chat.completions.create(
            model=config.sarvam_model,
            messages=[
                {"role": "system", "content": _SYSTEM_PROMPT},
                {"role": "user",   "content": user_prompt},
            ],
            max_tokens=4096,   # sarvam-105b reasoning model needs large token budget
            temperature=0.1,
        )
        # sarvam-105b: final answer in content; reasoning in reasoning_content
        answer_text = (
            response.choices[0].message.content
            or response.choices[0].message.reasoning_content
            or ""
        ).strip()
        if not answer_text:
            answer_text = "Not found in the provided documents."

    except Exception as exc:
        log.error("Answer generation failed: %s", exc)
        answer_text = f"Generation failed: {exc}"

    citation_footer = _build_citation_footer(sources)
    full_answer = answer_text + citation_footer

    return RetrievalResult(
        answer=full_answer,
        sources=sources,
        figures=figures,
        query_type=query_type,
        chunks_used=len(chunks),
    )
