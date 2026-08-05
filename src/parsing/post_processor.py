"""
post_processor.py
=================
Semantic enricher for parsed document output.

Runs AFTER the parser as a fast, standalone pass over JSON files.
Parser stays frozen — this is pure read-modify-write on the JSON.

Passes applied (in order)
--------------------------
Pass 0  Fix 2  — Merge split admonitions  (heading + icon-figure + paragraph → admonition)
Pass 1  Fix 1  — TOC table detection      (dot-leader tables → type "toc")
Pass 2         — Caption / admonition / heading-role / procedure-step detection
Pass 3         — Caption ↔ figure/table spatial linking
Pass 4  Fix 3  — Figure normalization     (tiny icons → decorative, store width/height)
Pass 5  Fix 4  — URL normalization        (all block types, including www. prefix)
Pass 6  Fix 6  — Table title consistency  (extract title + table_number from neighbours)
Pass 7         — Figure/table number extraction from caption text  (Should 12/13)
Pass 8         — Table rows + cols from markdown  (Should 9)
Pass 9  Fix 5  — Hyperlink hint flagging  (anchor phrases in this manual)
End            — Propagate image_hash / width / height from figure_metadata.json (Should 7/8)
End            — Write page_stats into each page JSON  (Should 10)

Usage
-----
from src.parsing.post_processor import PostProcessor
pp = PostProcessor()
stats = pp.enrich(Path("data/parsed/docling_parser/printer_manual"))
print(stats)
"""

from __future__ import annotations

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# Regex patterns
# ---------------------------------------------------------------------------

# Caption patterns — match "Figure 1-1 ...", "Fig. 2", "Table 3-2 ..."
# Named groups: kind, num
_CAPTION_RE = re.compile(
    r"^(?P<kind>Figure|Fig\.?|Table|Chart|Diagram|Photo|Image)\s+"
    r"(?P<num>\d+(?:-\d+)?)\b",
    re.IGNORECASE,
)

# Admonition keyword → severity mapping
_ADMONITION_KEYWORDS: Dict[str, str] = {
    "warning":   "warning",
    "caution":   "caution",
    "note":      "note",
    "tip":       "tip",
    "important": "important",
    "danger":    "warning",
    "notice":    "note",
}

# Admonition detection: starts with keyword (optionally followed by punctuation/space)
_ADMONITION_RE = re.compile(
    r"^(?P<kw>warning|caution|note|tip|important|danger|notice)[!:.\s]",
    re.IGNORECASE,
)

# Standalone admonition keyword line (e.g. just "WARNING" or "CAUTION:")
_ADMONITION_HEADER_RE = re.compile(
    r"^(?P<kw>warning|caution|note|tip|important|danger|notice)[!:.]?\s*$",
    re.IGNORECASE,
)

# Procedure step: content starts with a number followed by . or )
_PROCEDURE_RE = re.compile(r"^\d{1,2}[.)]\s+")

# Heading role thresholds
_HEADING_ROLES = {1: "chapter", 2: "section", 3: "subsection"}

# URL / reference detection (Fix 4 — extended to www. prefix)
_URL_RE = re.compile(r"^(?:https?://|www\.)\S+", re.IGNORECASE)

# Hyperlink hint phrases (Fix 5 — pragmatic flagging for this manual's anchor text)
_HYPERLINK_HINTS = re.compile(
    r"(?:view\s+(?:a\s+)?video|click\s+(?:here|the\s+link)|"
    r"see\s+figure\s+\d|refer\s+to\s+figure|click\s+below|"
    r"click\s+the\s+following)",
    re.IGNORECASE,
)

# TOC dot-leader detection (Fix 1)
_DOT_LEADER_RE  = re.compile(r"\.{4,}")              # 4+ consecutive dots
_TOC_PAGE_NUM_RE = re.compile(r"\|\s*\d+\s*\|?\s*$") # trailing page-number cell

# Figure normalization thresholds
_TINY_AREA   = 2_000.0   # pt² below which a figure is "tiny/decorative"
_LARGE_RATIO = 8.0        # dominant must be at least this many times larger
_MERGE_DIST  = 400.0      # max centroid-to-centroid distance (pt) for grouping

# Admonition split-merge: max vertical distance (pt) between blocks to merge
_ADMON_MERGE_GAP = 80.0


# ---------------------------------------------------------------------------
# PostProcessor
# ---------------------------------------------------------------------------

class PostProcessor:
    """
    Enrich parsed page JSON files in-place with semantic metadata.

    Parameters
    ----------
    logger : logging.Logger, optional
    """

    def __init__(self, logger: Optional[logging.Logger] = None) -> None:
        self.logger = logger or logging.getLogger(__name__)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def enrich(self, output_dir: Path) -> Dict[str, int]:
        """
        Enrich all page files under output_dir/pages/.

        Returns
        -------
        dict with counts of each enrichment applied across all pages.
        """
        output_dir = Path(output_dir).expanduser().resolve()
        pages_dir  = output_dir / "pages"
        figs_dir   = output_dir / "figures"

        if not pages_dir.exists():
            raise FileNotFoundError(f"pages/ directory not found: {pages_dir}")

        # Load figure metadata for hash/dimension backfill (Should 7/8)
        fig_meta = self._load_figure_meta(figs_dir)

        stats: Dict[str, int] = {
            "pages_processed":       0,
            # Pass 0 — Fix 2
            "admonitions_merged":    0,
            # Pass 1 — Fix 1
            "toc_tables":            0,
            # Pass 2
            "captions_detected":     0,
            "admonitions_detected":  0,
            "heading_roles":         0,
            "procedure_steps":       0,
            # Pass 3
            "captions_linked":       0,
            # Pass 4 — Fix 3
            "decorative_figures":    0,
            "captions_reassigned":   0,
            # Pass 5 — Fix 4
            "references":            0,
            # Pass 6 — Fix 6
            "table_titles":          0,
            # Pass 7 — Should 12/13
            "figure_numbers":        0,
            "table_numbers":         0,
            # Pass 8 — Should 9
            "table_dimensions":      0,
            # Pass 9 — Fix 5
            "hyperlink_hints":       0,
            # End — Should 7/8
            "fig_meta_backfilled":   0,
        }

        page_files = sorted(pages_dir.glob("page_*.json"))
        self.logger.info(
            "Post-processing %d page files in %s",
            len(page_files), output_dir.name,
        )

        # Accumulate all enriched page data in memory for write-back passes
        all_page_data: List[Dict[str, Any]] = []

        for pf in page_files:
            try:
                data: Dict[str, Any] = json.loads(pf.read_text(encoding="utf-8"))
                blocks: List[Dict]   = data.get("blocks", [])

                blocks, s = self._enrich_blocks(blocks, fig_meta)
                _add(stats, s)

                # Should 10 — write page_stats
                data["page_stats"] = self._compute_page_stats(blocks)
                data["blocks"]     = blocks

                pf.write_text(
                    json.dumps(data, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
                all_page_data.append(data)
                stats["pages_processed"] += 1

            except Exception as exc:
                self.logger.warning("Failed to process %s: %s", pf.name, exc)

        # ── Bug 2: Write-back figure_metadata.json and table files ───────
        self._write_figure_metadata(output_dir / "figures", all_page_data)
        self._write_table_files(output_dir / "tables", all_page_data)

        self.logger.info(
            "Done — %d pages | %d TOC tables | %d admonitions merged | "
            "%d admonitions detected | %d captions linked | "
            "%d decorative figs | %d references | %d table titles | "
            "%d hyperlink hints",
            stats["pages_processed"],
            stats["toc_tables"],
            stats["admonitions_merged"],
            stats["admonitions_detected"],
            stats["captions_linked"],
            stats["decorative_figures"],
            stats["references"],
            stats["table_titles"],
            stats["hyperlink_hints"],
        )
        return stats

    # ------------------------------------------------------------------
    # Core enrichment pipeline
    # ------------------------------------------------------------------

    def _enrich_blocks(
        self,
        blocks: List[Dict],
        fig_meta: Dict[str, Dict],
    ) -> Tuple[List[Dict], Dict[str, int]]:
        stats: Dict[str, int] = {}

        # ── Pass 0: Merge split admonitions  (Fix 2) ───────────────────
        blocks, s = self._merge_split_admonitions(blocks)
        _add(stats, s)

        # ── Pass 1: TOC table detection  (Fix 1) ───────────────────────
        s = self._detect_toc_tables(blocks)
        _add(stats, s)

        # ── Pass 2: Caption, admonition, heading role, list role ────────
        for blk in blocks:
            btype   = blk.get("type", "")
            content = (blk.get("content") or "").strip()

            # Caption detection
            if btype in ("paragraph", "caption") and _CAPTION_RE.match(content):
                blk["type"] = "caption"
                stats["captions_detected"] = stats.get("captions_detected", 0) + 1

            # Admonition detection (full inline: "NOTE: some text here")
            elif btype in ("paragraph", "unknown"):
                m = _ADMONITION_RE.match(content)
                if m:
                    kw = m.group("kw").lower()
                    blk["type"]     = "admonition"
                    blk["severity"] = _ADMONITION_KEYWORDS.get(kw, "note")
                    stats["admonitions_detected"] = stats.get("admonitions_detected", 0) + 1

            # Heading role
            if btype == "heading":
                lvl  = blk.get("level") or 2
                blk["role"] = _HEADING_ROLES.get(lvl, "subsection")
                stats["heading_roles"] = stats.get("heading_roles", 0) + 1

            # List item role
            if btype == "list_item" and content and _PROCEDURE_RE.match(content):
                blk["role"] = "procedure_step"
                stats["procedure_steps"] = stats.get("procedure_steps", 0) + 1

        # ── Pass 3: Caption ↔ figure/table linking ──────────────────────
        s = self._link_captions(blocks)
        _add(stats, s)

        # ── Pass 4: Figure normalization  (Fix 3 enhanced) ─────────────
        s = self._normalize_figures(blocks)
        _add(stats, s)

        # ── Pass 5: URL normalization  (Fix 4 extended) ─────────────────
        # Apply across all text-carrying block types, not just paragraph.
        for blk in blocks:
            if blk.get("type") in ("paragraph", "list_item", "heading", "unknown"):
                content = (blk.get("content") or "").strip()
                if content and _URL_RE.match(content):
                    blk["type"] = "reference"
                    stats["references"] = stats.get("references", 0) + 1

        # ── Pass 6: Table title consistency  (Fix 6) ────────────────────
        s = self._extract_table_titles(blocks)
        _add(stats, s)

        # ── Pass 7: Figure/table number extraction  (Should 12/13) ──────
        s = self._extract_element_numbers(blocks)
        _add(stats, s)

        # ── Pass 8: Table dimension counting  (Should 9) ────────────────
        s = self._count_table_dimensions(blocks)
        _add(stats, s)

        # ── Pass 9: Hyperlink hint flagging  (Fix 5) ────────────────────
        for blk in blocks:
            if blk.get("type") in ("paragraph", "list_item"):
                content = (blk.get("content") or "").strip()
                if content and _HYPERLINK_HINTS.search(content):
                    blk["hyperlink_hint"] = True
                    stats["hyperlink_hints"] = stats.get("hyperlink_hints", 0) + 1

        # ── End: Backfill image_hash / width / height  (Should 7/8) ────
        s = self._backfill_figure_meta(blocks, fig_meta)
        _add(stats, s)

        # ── Final: Repair dangling context links  (Bug 3B) ───────────────
        self._repair_context_links(blocks)

        return blocks, stats

    # ------------------------------------------------------------------
    # Pass 0 — Merge split admonitions  (Fix 2)
    # ------------------------------------------------------------------

    def _merge_split_admonitions(
        self, blocks: List[Dict]
    ) -> Tuple[List[Dict], Dict[str, int]]:
        """
        Detect the pattern:
            [heading/paragraph]  text = "WARNING" / "CAUTION" / etc.
            [figure]             tiny icon  (area < _TINY_AREA)      ← optional
            [paragraph]          actual admonition body text

        Collapse the three (or two, without icon) blocks into one admonition block.
        The original block IDs, reading_order, section, and bbox are preserved
        from the keyword-header block.

        IMPORTANT: the icon figure index is staged tentatively and only committed
        to `skip` after the full pattern is confirmed.  This prevents the icon
        from being silently dropped when no body paragraph follows.
        """
        stats = {"admonitions_merged": 0}
        if not blocks:
            return blocks, stats

        merged: List[Dict] = []
        skip: set = set()

        for i, blk in enumerate(blocks):
            if i in skip:
                continue

            btype   = blk.get("type", "")
            content = (blk.get("content") or "").strip()

            # Is this block a standalone admonition keyword heading?
            hm = _ADMONITION_HEADER_RE.match(content)
            is_header = (
                btype in ("heading", "paragraph", "list_item") and hm is not None
            )
            if not is_header:
                merged.append(blk)
                continue

            kw       = hm.group("kw").lower()
            severity = _ADMONITION_KEYWORDS.get(kw, "note")

            # Look ahead: optional tiny-figure icon, then body paragraph
            icon_path: Optional[str] = None
            icon_idx:  Optional[int] = None   # tentative — not added to skip yet
            body_content: Optional[str] = None
            body_idx: Optional[int] = None

            j = i + 1

            # Optional tiny icon figure — stage tentatively
            if j < len(blocks) and blocks[j].get("type") == "figure":
                fig  = blocks[j]
                bbox = fig.get("bbox")
                if bbox and self._bbox_area(bbox) < _TINY_AREA:
                    if self._vertically_close(blk.get("bbox"), bbox, _ADMON_MERGE_GAP):
                        icon_path = fig.get("image_path")
                        icon_idx  = j          # tentative — NOT yet in skip
                        j += 1

            # Body paragraph immediately following
            if j < len(blocks) and blocks[j].get("type") in ("paragraph", "unknown"):
                body_blk     = blocks[j]
                body_content = (body_blk.get("content") or "").strip()
                ref_bbox     = (blocks[i + 1].get("bbox") if icon_path else blk.get("bbox"))
                if not self._vertically_close(ref_bbox, body_blk.get("bbox"), _ADMON_MERGE_GAP * 2):
                    # Body is too far away; do not merge
                    body_content = None
                else:
                    body_idx = j

            if body_content is None:
                # Pattern not complete — leave the header block unchanged.
                # The icon figure (if staged) is NOT added to skip, so it
                # will be emitted normally in the next iteration.
                merged.append(blk)
                continue

            # ── Pattern confirmed: commit both icon and body to skip ──────
            if icon_idx is not None:
                skip.add(icon_idx)
            skip.add(body_idx)

            # Build merged admonition
            admon: Dict[str, Any] = {
                "type":          "admonition",
                "id":            blk["id"],
                "reading_order": blk["reading_order"],
                "severity":      severity,
                "content":       f"{kw.upper()}: {body_content}",
            }
            if blk.get("section"):
                admon["section"] = blk["section"]
            if blk.get("bbox"):
                admon["bbox"] = blk["bbox"]
            if icon_path:
                admon["icon_path"] = icon_path

            merged.append(admon)
            stats["admonitions_merged"] += 1

        return merged, stats

    # ------------------------------------------------------------------
    # Pass 1 — TOC table detection  (Fix 1)
    # ------------------------------------------------------------------

    def _detect_toc_tables(self, blocks: List[Dict]) -> Dict[str, int]:
        """
        Reclassify table blocks whose markdown matches the TOC dot-leader pattern:
          - ≥ 3 content rows contain dot-leaders (.......)
          - ≥ 2 rows end with a bare page-number cell  (| 42 |)

        Sets type → "toc" and toc → true.
        These are navigation pages and should be excluded from the chunker.
        """
        stats = {"toc_tables": 0}
        for blk in blocks:
            if blk.get("type") != "table":
                continue
            md = blk.get("markdown") or ""
            rows = [r for r in md.split("\n") if r.strip().startswith("|")]
            if len(rows) < 3:
                continue

            dot_rows  = sum(1 for r in rows if _DOT_LEADER_RE.search(r))
            pnum_rows = sum(1 for r in rows if _TOC_PAGE_NUM_RE.search(r))

            if dot_rows >= 3 and pnum_rows >= 2:
                blk["type"] = "toc"
                blk["toc"]  = True
                stats["toc_tables"] += 1

        return stats

    # ------------------------------------------------------------------
    # Pass 3 — Caption ↔ figure/table linking
    # ------------------------------------------------------------------

    def _link_captions(self, blocks: List[Dict]) -> Dict[str, int]:
        """
        For each caption block, find the spatially nearest figure or table
        (by bbox centroid distance) and create bidirectional links:
            caption["caption_for"] = figure/table id
            figure/table["caption_id"] = caption id
            figure["caption"] = caption text  (convenience copy)

        Falls back to reading-order proximity (±3 positions) when bbox is absent.
        Runs are idempotent (stale links are cleared first).
        """
        stats: Dict[str, int] = {"captions_linked": 0}

        fig_table_blocks = [b for b in blocks if b.get("type") in ("figure", "table")]

        # Clear stale links for idempotency
        for b in blocks:
            b.pop("caption_id",  None)
            b.pop("caption_for", None)

        n = len(blocks)
        for i, blk in enumerate(blocks):
            if blk.get("type") != "caption":
                continue

            cap_bbox = blk.get("bbox")
            target: Optional[Dict] = None

            if cap_bbox and fig_table_blocks:
                target = self._nearest_by_bbox(cap_bbox, fig_table_blocks)

            if target is None:
                for delta in (1, -1, 2, -2, 3, -3):
                    j = i + delta
                    if 0 <= j < n and blocks[j].get("type") in ("figure", "table"):
                        target = blocks[j]
                        break

            if target is not None:
                blk["caption_for"]   = target["id"]
                target["caption_id"] = blk["id"]
                if target.get("type") == "figure":
                    target["caption"] = blk.get("content")
                stats["captions_linked"] += 1

        return stats

    # ------------------------------------------------------------------
    # Pass 4 — Figure normalization  (Fix 3 enhanced)
    # ------------------------------------------------------------------

    def _normalize_figures(self, blocks: List[Dict]) -> Dict[str, int]:
        """
        Identify tiny figures (area < _TINY_AREA) near a dominant figure
        (area ≥ _LARGE_RATIO × tiny area, within _MERGE_DIST).

        For each identified tiny figure:
          1. Mark decorative=True
          2. Store bbox-derived width/height (in pts) for downstream filtering
          3. Reassign any mis-linked captions to the dominant figure
        """
        stats: Dict[str, int] = {"decorative_figures": 0, "captions_reassigned": 0}

        figs = [b for b in blocks if b.get("type") == "figure" and b.get("bbox")]
        if len(figs) < 2:
            return stats

        for tiny in figs:
            area = self._bbox_area(tiny["bbox"])
            if area >= _TINY_AREA:
                continue

            # Store bbox dimensions (pts) on the icon
            bbox = tiny["bbox"]
            tiny.setdefault("width",  int(abs(bbox["r"] - bbox["l"])))
            tiny.setdefault("height", int(abs(bbox["t"] - bbox["b"])))

            tiny_cx, tiny_cy = self._bbox_centroid(bbox)

            dominant: Optional[Dict] = None
            dom_area:  float         = 0.0
            for other in figs:
                if other["id"] == tiny["id"]:
                    continue
                other_area = self._bbox_area(other["bbox"])
                if other_area < area * _LARGE_RATIO:
                    continue
                ox, oy = self._bbox_centroid(other["bbox"])
                dist = ((tiny_cx - ox) ** 2 + (tiny_cy - oy) ** 2) ** 0.5
                if dist <= _MERGE_DIST and other_area > dom_area:
                    dominant = other
                    dom_area = other_area

            if dominant is None:
                continue

            tiny["decorative"] = True
            stats["decorative_figures"] += 1

            cap_id = tiny.pop("caption_id", None)
            if cap_id:
                tiny.pop("caption", None)
                cap_blk = next((b for b in blocks if b.get("id") == cap_id), None)
                if cap_blk is not None:
                    cap_blk["caption_for"] = dominant["id"]
                dominant["caption_id"] = cap_id
                dominant["caption"]    = (cap_blk or {}).get("content")
                stats["captions_reassigned"] += 1

        return stats

    # ------------------------------------------------------------------
    # Pass 6 — Table title consistency  (Fix 6)
    # ------------------------------------------------------------------

    def _extract_table_titles(self, blocks: List[Dict]) -> Dict[str, int]:
        """
        For each table/toc block that has no title yet, search ±2 neighbours
        for a caption/heading/paragraph matching a "Table N-N ..." pattern.
        Store:
            "title":        full caption text
            "table_number": "Table 2-1"
        """
        stats = {"table_titles": 0}
        n = len(blocks)

        for i, blk in enumerate(blocks):
            if blk.get("type") not in ("table", "toc"):
                continue
            if blk.get("title"):
                continue   # already populated

            for delta in (-2, -1, 1, 2):
                j = i + delta
                if not (0 <= j < n):
                    continue
                nb = blocks[j]
                if nb.get("type") not in ("caption", "heading", "paragraph"):
                    continue
                content = (nb.get("content") or "").strip()
                m = _CAPTION_RE.match(content)
                if m and m.group("kind").lower() == "table":
                    blk["title"]        = content
                    blk["table_number"] = f"{m.group('kind')} {m.group('num')}"
                    stats["table_titles"] += 1
                    break

        return stats

    # ------------------------------------------------------------------
    # Pass 7 — Figure/table number extraction  (Should 12/13)
    # ------------------------------------------------------------------

    def _extract_element_numbers(self, blocks: List[Dict]) -> Dict[str, int]:
        """
        Extract the numeric identifier from caption/title text and store it
        as a separate field.

        Figure with caption "Figure 1-6  Recycle and unpack":
            "figure_number": "Figure 1-6"
            "caption":       "Recycle and unpack"    ← prefix trimmed

        Table with title "Table 2-1 Supplies":
            "table_number": "Table 2-1"   (skip if Pass 6 already set it)
        """
        stats = {"figure_numbers": 0, "table_numbers": 0}

        for blk in blocks:
            btype = blk.get("type")

            if btype == "figure":
                caption = (blk.get("caption") or "").strip()
                if not caption:
                    continue
                m = _CAPTION_RE.match(caption)
                if m:
                    blk["figure_number"] = f"{m.group('kind')} {m.group('num')}"
                    trimmed = caption[m.end():].strip(" \t\u2003\u2002-\u2013\u2014")
                    if trimmed:
                        blk["caption"] = trimmed
                    stats["figure_numbers"] += 1

            elif btype in ("table", "toc"):
                if blk.get("table_number"):
                    continue   # already set by Pass 6
                title = (blk.get("title") or "").strip()
                if not title:
                    continue
                m = _CAPTION_RE.match(title)
                if m and m.group("kind").lower() == "table":
                    blk["table_number"] = f"{m.group('kind')} {m.group('num')}"
                    stats["table_numbers"] += 1

        return stats

    # ------------------------------------------------------------------
    # Pass 8 — Table dimension counting  (Should 9)
    # ------------------------------------------------------------------

    def _count_table_dimensions(self, blocks: List[Dict]) -> Dict[str, int]:
        """
        Parse each table's markdown to count rows and columns.
        Stores "rows": N, "cols": M.
        Skips TOC blocks — they are navigation tables, not data tables.
        """
        stats = {"table_dimensions": 0}

        for blk in blocks:
            if blk.get("type") != "table":
                continue
            md = (blk.get("markdown") or "").strip()
            if not md:
                continue

            rows_raw = [r for r in md.split("\n") if r.strip().startswith("|")]
            # Exclude markdown separator rows (all dashes/colons)
            data_rows = [
                r for r in rows_raw
                if not re.match(r"^\|[-:| ]+\|?\s*$", r.strip())
            ]
            if not data_rows:
                continue

            cols = len([c for c in data_rows[0].split("|") if c.strip()])
            blk["rows"] = len(data_rows)
            blk["cols"] = cols
            stats["table_dimensions"] += 1

        return stats

    # ------------------------------------------------------------------
    # End — Backfill figure metadata  (Should 7/8)
    # ------------------------------------------------------------------

    def _backfill_figure_meta(
        self,
        blocks: List[Dict],
        fig_meta: Dict[str, Dict],
    ) -> Dict[str, int]:
        """
        For each figure block missing image_hash / width / height,
        pull those values from the pre-loaded figure_metadata.json.

        Also renames previous_text_id / next_text_id → previous_block_id / next_block_id
        for schema generality (the neighbour is not always a text block).
        """
        stats = {"fig_meta_backfilled": 0, "block_id_renamed": 0}
        for blk in blocks:
            if blk.get("type") != "figure":
                continue

            # ── Rename previous_text_id / next_text_id (idempotent) ────
            renamed = False
            for old, new in (
                ("previous_text_id", "previous_block_id"),
                ("next_text_id",     "next_block_id"),
            ):
                if old in blk:
                    blk[new] = blk.pop(old)
                    renamed  = True
            if renamed:
                stats["block_id_renamed"] += 1

            # ── Backfill metadata from figure_metadata.json ────────────
            if not fig_meta:
                continue
            meta = fig_meta.get(blk.get("id"))
            if not meta:
                continue
            changed = False
            for field in ("image_hash", "width", "height"):
                if blk.get(field) is None and meta.get(field) is not None:
                    blk[field] = meta[field]
                    changed    = True
            if changed:
                stats["fig_meta_backfilled"] += 1

        return stats

    # ------------------------------------------------------------------
    # Should 10 — Page statistics
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_page_stats(blocks: List[Dict]) -> Dict[str, int]:
        """
        Return a statistics dict describing the block composition of a page.

        block_count = total blocks on the page (every type).
        text_blocks = ALL blocks that carry text content:
            paragraph + heading + list_item + caption + admonition + reference
            + footnote + code + unknown
        This matches the semantic expectation: any block a chunker would read.
        """
        counts: Dict[str, int] = {
            "block_count":  len(blocks),  # total blocks on page (every type)
            "text_blocks":  0,            # ALL textual block types combined
            "paragraphs":   0,
            "headings":     0,
            "list_items":   0,
            "captions":     0,
            "admonitions":  0,
            "references":   0,
            "figures":      0,
            "tables":       0,
            "toc_tables":   0,
        }
        for blk in blocks:
            t = blk.get("type", "")
            if t == "paragraph":
                counts["paragraphs"]  += 1
                counts["text_blocks"] += 1
            elif t == "heading":
                counts["headings"]    += 1
                counts["text_blocks"] += 1
            elif t == "list_item":
                counts["list_items"]  += 1
                counts["text_blocks"] += 1
            elif t == "caption":
                counts["captions"]    += 1
                counts["text_blocks"] += 1
            elif t == "admonition":
                counts["admonitions"] += 1
                counts["text_blocks"] += 1
            elif t == "reference":
                counts["references"]  += 1
                counts["text_blocks"] += 1
            elif t in ("footnote", "code", "unknown"):
                counts["text_blocks"] += 1
            elif t == "figure":
                counts["figures"]     += 1
            elif t == "table":
                counts["tables"]      += 1
            elif t == "toc":
                counts["toc_tables"]  += 1
        return counts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _bbox_area(bbox: Dict[str, float]) -> float:
        return abs(bbox["r"] - bbox["l"]) * abs(bbox["t"] - bbox["b"])

    @staticmethod
    def _bbox_centroid(bbox: Dict[str, float]) -> Tuple[float, float]:
        return (bbox["l"] + bbox["r"]) / 2, (bbox["t"] + bbox["b"]) / 2

    @staticmethod
    def _vertically_close(
        bbox_a: Optional[Dict[str, float]],
        bbox_b: Optional[Dict[str, float]],
        max_gap: float,
    ) -> bool:
        """True if the vertical distance between two bboxes is within max_gap."""
        if not bbox_a or not bbox_b:
            return True   # can't measure → assume close enough
        top_a = max(bbox_a["t"], bbox_a["b"])
        bot_a = min(bbox_a["t"], bbox_a["b"])
        top_b = max(bbox_b["t"], bbox_b["b"])
        bot_b = min(bbox_b["t"], bbox_b["b"])
        # gap is how far apart the two bboxes are vertically (0 if overlapping)
        gap = max(0.0, max(bot_a, bot_b) - min(top_a, top_b))
        return gap <= max_gap

    def _nearest_by_bbox(
        self,
        cap_bbox: Dict[str, float],
        candidates: List[Dict],
        max_dist: float = 500.0,
    ) -> Optional[Dict]:
        """
        Return the candidate block whose bbox centroid is closest to
        cap_bbox's centroid, within max_dist points.
        Skips decorative figures so tiny icons do not attract captions.
        """
        cap_cx = (cap_bbox["l"] + cap_bbox["r"]) / 2
        cap_cy = (cap_bbox["t"] + cap_bbox["b"]) / 2

        best:      Optional[Dict] = None
        best_dist: float          = float("inf")

        for blk in candidates:
            if blk.get("decorative"):
                continue
            fig_bbox = blk.get("bbox")
            if not fig_bbox:
                continue
            fig_cx = (fig_bbox["l"] + fig_bbox["r"]) / 2
            fig_cy = (fig_bbox["t"] + fig_bbox["b"]) / 2
            dist   = ((cap_cx - fig_cx) ** 2 + (cap_cy - fig_cy) ** 2) ** 0.5
            if dist < best_dist:
                best_dist = dist
                best      = blk

        return best if best_dist <= max_dist else None

    def _load_figure_meta(self, figs_dir: Path) -> Dict[str, Dict]:
        meta_file = figs_dir / "figure_metadata.json"
        if not meta_file.exists():
            return {}
        try:
            entries = json.loads(meta_file.read_text(encoding="utf-8"))
            return {e["id"]: e for e in entries if "id" in e}
        except Exception as exc:
            self.logger.warning("Could not load figure_metadata.json: %s", exc)
            return {}

    # ------------------------------------------------------------------
    # Bug 3B — Repair dangling context links
    # ------------------------------------------------------------------

    def _repair_context_links(self, blocks: List[Dict]) -> None:
        """
        Nullify previous_block_id / next_block_id on any block that points to
        an ID no longer present in the block list.

        Post-processing removes blocks (e.g. icon figures absorbed into
        admonitions, captions merged).  Any context links that pointed to those
        removed blocks would become dangling references that break referential
        integrity.  This pass is cheap — O(n) — and must run last.
        """
        live: set = {b["id"] for b in blocks}
        for b in blocks:
            if b.get("previous_block_id") not in live:
                b.pop("previous_block_id", None)
            if b.get("next_block_id") not in live:
                b.pop("next_block_id", None)

    # ------------------------------------------------------------------
    # Bug 2a — Write-back: figure_metadata.json
    # ------------------------------------------------------------------

    def _write_figure_metadata(
        self, figs_dir: Path, all_page_data: List[Dict[str, Any]]
    ) -> None:
        """
        Rebuild figure_metadata.json from the *enriched* page blocks.

        The original figure_metadata.json is written by the parser before
        enrichment, so caption / figure_number / decorative fields added by
        the post-processor are never reflected there.  This pass rebuilds the
        file from the enriched page data so both artifacts stay in sync.
        """
        if not figs_dir.exists():
            return

        meta_file = figs_dir / "figure_metadata.json"
        # Load existing entries as a base (preserves image_hash, file_path, etc.)
        existing: Dict[str, Dict] = {}
        if meta_file.exists():
            try:
                for e in json.loads(meta_file.read_text(encoding="utf-8")):
                    if "id" in e:
                        existing[e["id"]] = e
            except Exception:
                pass

        entries: List[Dict] = []
        seen: set = set()

        SYNC_FIELDS = (
            "caption", "caption_id", "figure_number", "figure_metadata",
            "decorative", "width", "height", "image_hash",
            "previous_block_id", "next_block_id", "section", "bbox",
            "reading_order", "page_number",
        )

        for data in all_page_data:
            for blk in data.get("blocks", []):
                if blk.get("type") != "figure":
                    continue
                fig_id = blk["id"]
                if fig_id in seen:
                    continue
                seen.add(fig_id)

                entry = dict(existing.get(fig_id, {}))
                entry["id"] = fig_id
                for field in SYNC_FIELDS:
                    if field in blk:
                        entry[field] = blk[field]
                    elif field in entry and field not in blk:
                        pass  # keep existing value if page block dropped the field
                entries.append(entry)

        try:
            meta_file.write_text(
                json.dumps(entries, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
            self.logger.debug(
                "Wrote figure_metadata.json with %d entries (%d updated)",
                len(entries), len(seen),
            )
        except Exception as exc:
            self.logger.warning("Could not write figure_metadata.json: %s", exc)

    # ------------------------------------------------------------------
    # Bug 2b — Write-back: table files
    # ------------------------------------------------------------------

    def _write_table_files(
        self, tbls_dir: Path, all_page_data: List[Dict[str, Any]]
    ) -> None:
        """
        Patch enriched fields back into tables/tbl_*.json files.

        The post-processor adds title, table_number, rows, cols, and toc flags
        to table blocks inside page JSON, but the individual table JSON files
        written by the parser are never updated.  This pass syncs them.
        """
        if not tbls_dir or not tbls_dir.exists():
            return

        TABLE_SYNC_FIELDS = ("title", "table_number", "rows", "cols", "toc", "type")

        updated = 0
        for data in all_page_data:
            for blk in data.get("blocks", []):
                if blk.get("type") not in ("table", "toc"):
                    continue
                tbl_file = tbls_dir / f"{blk['id']}.json"
                if not tbl_file.exists():
                    continue
                try:
                    tbl = json.loads(tbl_file.read_text(encoding="utf-8"))
                    changed = False
                    for field in TABLE_SYNC_FIELDS:
                        if field in blk and tbl.get(field) != blk[field]:
                            tbl[field] = blk[field]
                            changed = True
                    if changed:
                        tbl_file.write_text(
                            json.dumps(tbl, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        updated += 1
                except Exception as exc:
                    self.logger.warning(
                        "Could not update %s: %s", tbl_file.name, exc
                    )

        self.logger.debug("Synced %d table JSON files", updated)


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def _add(target: Dict[str, int], source: Dict[str, int]) -> None:
    """In-place merge: target[k] += source[k] for all k in source."""
    for k, v in source.items():
        target[k] = target.get(k, 0) + v
