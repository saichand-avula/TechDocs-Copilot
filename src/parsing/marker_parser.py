"""
marker_parser.py
================
MarkerParser — concrete implementation of the Parser ABC using Marker-PDF v2.x.

Schema v2.1 — Final Frozen Output
-----------------------------------
- PageBlock carries only fields relevant to its type (no nulls)
- document_id and page stored once at PageOutput level
- Unified "section" field throughout
- Figures carry previous_text_id / next_text_id for context retrieval
- Page files serialized with exclude_none=True → clean minimal JSON

Marker 2.0 block types:
    Text, SectionHeader, ListItem, ListGroup, Caption,
    Footnote, PageHeader, PageFooter, Code, TextInlineMath,
    Table, TableGroup, Figure, FigureGroup, Picture
"""

from __future__ import annotations

import hashlib
import io
import logging
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .config import ParserConfig
from .parser_interface import (
    BlockType,
    DocumentIndex,
    DocumentMetadata,
    DocumentStatistics,
    FigureBlock,
    PageBlock,
    PageIndexEntry,
    PageOutput,
    ParsedDocument,
    Parser,
    TableBlock,
    TextBlock,
)
from .utils import (
    ensure_output_dirs,
    make_document_id,
    make_figure_filename,
    make_figure_id,
    make_table_id,
    make_text_id,
    setup_logger,
    utc_now_iso,
    write_json,
)

try:
    from marker.converters.pdf import PdfConverter
    from marker.models import create_model_dict
    from marker.config.parser import ConfigParser as MarkerConfigParser
    _MARKER_AVAILABLE = True
except ImportError:
    _MARKER_AVAILABLE = False


# ---------------------------------------------------------------------------
# Block type classification
# ---------------------------------------------------------------------------

_HEADING_TYPES  = {"SectionHeader"}
_TEXT_TYPES     = {"Text", "TextInlineMath", "Footnote", "PageHeader", "PageFooter", "Code"}
_LIST_TYPES     = {"ListItem", "ListGroup"}
_TABLE_TYPES    = {"Table", "TableGroup"}
_FIGURE_TYPES   = {"Figure", "FigureGroup", "Picture"}
_CAPTION_TYPES  = {"Caption"}

_TYPE_MAP: Dict[str, BlockType] = {
    "SectionHeader":  "heading",
    "Text":           "paragraph",
    "TextInlineMath": "paragraph",
    "ListItem":       "list_item",
    "ListGroup":      "list_item",
    "Caption":        "caption",
    "Footnote":       "footnote",
    "PageHeader":     "page_header",
    "PageFooter":     "page_footer",
    "Code":           "code",
}


# ---------------------------------------------------------------------------
# MarkerParser
# ---------------------------------------------------------------------------

class MarkerParser(Parser):
    """
    Production document parser backed by Marker-PDF v2.x.

    Usage
    -----
    >>> parser = MarkerParser(config)
    >>> result = parser.parse(Path("manual.pdf"))
    >>> parser.save(result, Path("output/"))
    """

    PARSER_NAME = "MarkerParser"

    def __init__(self, config: ParserConfig) -> None:
        if not _MARKER_AVAILABLE:
            raise ImportError("marker-pdf not installed. Run: pip install 'marker-pdf[full]'")
        self.config = config
        self.logger: logging.Logger = setup_logger(
            __name__, level=config.log_level, log_file=config.log_file
        )
        self._models: Optional[dict] = None
        self._figure_images: Dict[str, object] = {}
        self._wall_start: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: Path) -> ParsedDocument:
        self._wall_start = time.time()
        pdf_path = Path(pdf_path).expanduser().resolve()
        self.logger.info("Starting parse: %s", pdf_path.name)

        cp = MarkerConfigParser({
            "output_format": "json",
            "paginate_output": True,
            "extract_images": self.config.extract_figures,
        })
        converter = PdfConverter(
            config=cp.generate_config_dict(),
            artifact_dict=self._get_models(),
            processor_list=cp.get_processors(),
            renderer=cp.get_renderer(),
        )
        rendered = converter(str(pdf_path))
        self.logger.info("Marker conversion complete.")

        doc_id = make_document_id(pdf_path)
        text_ctr = fig_ctr = tbl_ctr = 0
        self._figure_images = {}

        # Unified reading-order list: ("text"|"table"|"figure", block)
        all_blocks: List[Tuple[str, object]] = []
        current_section: Optional[str] = None

        pages_data: list = getattr(rendered, "children", []) or []
        total_pages = len(pages_data)
        self.logger.info("Walking %d pages…", total_pages)

        for page_idx, page_obj in enumerate(pages_data):
            page_no = page_idx + 1
            for block in (getattr(page_obj, "children", []) or []):
                btype = str(getattr(block, "block_type", "") or "").strip()
                block_id = str(getattr(block, "id", "") or "")
                ro = len(all_blocks)

                if btype in _HEADING_TYPES:
                    text = self._plain_text(block)
                    if not text:
                        continue
                    current_section = text
                    text_ctr += 1
                    all_blocks.append(("text", TextBlock(
                        id=make_text_id(text_ctr),
                        type="heading",
                        page=page_no,
                        section=current_section,
                        content=text,
                        block_id=block_id,
                        reading_order=ro,
                        level=2,    # Marker doesn't expose nesting depth
                    )))

                elif btype in _TEXT_TYPES or btype in _LIST_TYPES:
                    text = self._plain_text(block)
                    if not text:
                        continue
                    section = self._section_from(block) or current_section
                    text_ctr += 1
                    all_blocks.append(("text", TextBlock(
                        id=make_text_id(text_ctr),
                        type=_TYPE_MAP.get(btype, "unknown"),
                        page=page_no,
                        section=section,
                        content=text,
                        block_id=block_id,
                        reading_order=ro,
                    )))

                elif btype in _CAPTION_TYPES:
                    text = self._plain_text(block)
                    if not text:
                        continue
                    text_ctr += 1
                    all_blocks.append(("text", TextBlock(
                        id=make_text_id(text_ctr),
                        type="caption",
                        page=page_no,
                        section=self._section_from(block) or current_section,
                        content=text,
                        block_id=block_id,
                        reading_order=ro,
                    )))

                elif btype in _TABLE_TYPES and self.config.extract_tables:
                    html = str(getattr(block, "html", "") or "")
                    tbl_ctr += 1
                    all_blocks.append(("table", TableBlock(
                        id=make_table_id(tbl_ctr),
                        page=page_no,
                        section=self._section_from(block) or current_section,
                        title=self._child_caption(block),
                        markdown=self._html_to_md(html) if html else "",
                        block_id=block_id,
                        reading_order=ro,
                    )))

                elif btype in _FIGURE_TYPES and self.config.extract_figures:
                    fig_ctr += 1
                    fig_id   = make_figure_id(fig_ctr)
                    rel_path = f"figures/{make_figure_filename(fig_ctr, self.config.figure_format)}"
                    imgs: dict = getattr(block, "images", {}) or {}
                    if imgs:
                        first = next(iter(imgs.values()), None)
                        if first is not None:
                            self._figure_images[fig_id] = first
                    all_blocks.append(("figure", FigureBlock(
                        id=fig_id,
                        page=page_no,
                        section=self._section_from(block) or current_section,
                        caption=self._child_caption(block),
                        path=rel_path,
                        block_id=block_id,
                        reading_order=ro,
                        image_path=rel_path,
                    )))

        # Link figure reading context
        self._link_figure_context(all_blocks)

        text_blocks   = [b for t, b in all_blocks if t == "text"]
        table_blocks  = [b for t, b in all_blocks if t == "table"]
        figure_blocks = [b for t, b in all_blocks if t == "figure"]

        page_outputs = self._build_pages(all_blocks, doc_id)
        title = self._title(text_blocks) or pdf_path.stem

        doc_idx = DocumentIndex(
            document_id=doc_id,
            title=title,
            pages=total_pages,
            statistics=DocumentStatistics(
                text_blocks=len(text_blocks),
                tables=len(table_blocks),
                figures=len(figure_blocks),
            ),
            page_index=[
                PageIndexEntry(page=po.page, path=f"pages/page_{po.page:03d}.json")
                for po in page_outputs
            ],
        )

        metadata = DocumentMetadata(
            document_id=doc_id,
            filename=pdf_path.name,
            title=title,
            total_pages=total_pages,
            parser=self.PARSER_NAME,
            parser_version=self._marker_version(),
            parsed_at=utc_now_iso(),
        )

        self.logger.info(
            "Parsed %d text blocks, %d tables, %d figures across %d pages.",
            len(text_blocks), len(table_blocks), len(figure_blocks), total_pages,
        )
        return ParsedDocument(
            metadata=metadata,
            document=doc_idx,
            page_outputs=page_outputs,
            table_blocks=table_blocks,
            figure_blocks=figure_blocks,
        )

    def save(self, result: ParsedDocument, output_dir: Path) -> None:
        output_dir = Path(output_dir).expanduser().resolve()
        dirs = ensure_output_dirs(output_dir)
        self.logger.info("Saving output to: %s", output_dir)

        write_json(result.document, dirs["root"] / "document.json")
        self.logger.info("Wrote document.json")

        write_json(result.metadata, dirs["root"] / "metadata.json")
        self.logger.info("Wrote metadata.json")

        # Page files — clean output, no nulls
        for po in result.page_outputs:
            data = {
                "page":        po.page,
                "document_id": po.document_id,
                "blocks": [b.model_dump(exclude_none=True) for b in po.blocks],
            }
            if po.page_bbox is not None:
                data["page_bbox"] = po.page_bbox
            path = dirs["pages"] / f"page_{po.page:03d}.json"
            path.write_text(__import__("json").dumps(data, indent=2, ensure_ascii=False))
        self.logger.info("Wrote %d page files.", len(result.page_outputs))

        for tbl in result.table_blocks:
            write_json(tbl, dirs["tables"] / f"{tbl.id}.json")
        self.logger.info("Wrote %d table files.", len(result.table_blocks))

        self._save_figures(result, dirs["figures"])

        result.metadata.processing_time_seconds = round(time.time() - self._wall_start, 1)
        write_json(result.metadata, dirs["root"] / "metadata.json")
        self.logger.info("Save complete: %s", output_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_models(self) -> dict:
        if self._models is None:
            self.logger.info("Loading Marker models…")
            self._models = create_model_dict()
            self.logger.info("Marker models loaded.")
        return self._models

    def _plain_text(self, block: object) -> str:
        html = str(getattr(block, "html", "") or "")
        if html:
            return re.sub(r"\s+", " ", re.sub(r"<[^>]+>", " ", html)).strip()
        parts = [self._plain_text(c) for c in (getattr(block, "children", []) or [])]
        return " ".join(p for p in parts if p).strip()

    def _section_from(self, block: object) -> Optional[str]:
        sh = getattr(block, "section_hierarchy", None)
        if sh and isinstance(sh, dict) and sh:
            val = sh[max(sh.keys())]
            return str(val).strip() or None
        return None

    def _child_caption(self, block: object) -> Optional[str]:
        for child in (getattr(block, "children", []) or []):
            if str(getattr(child, "block_type", "")) in _CAPTION_TYPES:
                return self._plain_text(child) or None
        return None

    def _html_to_md(self, html: str) -> str:
        try:
            from html.parser import HTMLParser

            class _TP(HTMLParser):
                def __init__(self):
                    super().__init__()
                    self.rows: List[List[str]] = []
                    self._row: List[str] = []
                    self._cell: List[str] = []
                    self._in_cell = False

                def handle_starttag(self, tag, attrs):
                    if tag == "tr":       self._row = []
                    elif tag in ("td","th"): self._cell = []; self._in_cell = True

                def handle_endtag(self, tag):
                    if tag in ("td","th"):
                        self._row.append(" ".join(self._cell).strip()); self._in_cell = False
                    elif tag == "tr" and self._row:
                        self.rows.append(self._row)

                def handle_data(self, data):
                    if self._in_cell: self._cell.append(data.strip())

            p = _TP(); p.feed(html)
            if not p.rows: return html
            lines = []
            for i, row in enumerate(p.rows):
                lines.append("| " + " | ".join(row) + " |")
                if i == 0: lines.append("|" + "|".join(["---"] * len(row)) + "|")
            return "\n".join(lines)
        except Exception:
            return html

    def _title(self, text_blocks: List[TextBlock]) -> Optional[str]:
        for b in text_blocks:
            if b.type == "heading" and b.page <= 5:
                return b.content
        return None

    def _link_figure_context(self, all_blocks: List[Tuple[str, object]]) -> None:
        text_positions = [i for i, (t, _) in enumerate(all_blocks) if t == "text"]
        for pos, (btype, block) in enumerate(all_blocks):
            if btype != "figure":
                continue
            prev_pos = next(
                (text_positions[j] for j in range(len(text_positions) - 1, -1, -1)
                 if text_positions[j] < pos), None)
            next_pos = next(
                (text_positions[j] for j in range(len(text_positions))
                 if text_positions[j] > pos), None)
            block.previous_text_id = all_blocks[prev_pos][1].id if prev_pos is not None else None
            block.next_text_id     = all_blocks[next_pos][1].id if next_pos is not None else None

    def _build_pages(
        self, all_blocks: List[Tuple[str, object]], doc_id: str
    ) -> List[PageOutput]:
        pages: Dict[int, List[PageBlock]] = defaultdict(list)

        for btype, block in all_blocks:
            pno = block.page
            if btype == "text":
                pages[pno].append(PageBlock(
                    type=block.type,
                    id=block.id,
                    reading_order=block.reading_order,
                    section=block.section or None,
                    content=block.content,
                    level=block.level,
                    bbox=block.bbox,
                ))
            elif btype == "table":
                pages[pno].append(PageBlock(
                    type="table",
                    id=block.id,
                    reading_order=block.reading_order,
                    section=block.section or None,
                    markdown=block.markdown,
                ))
            elif btype == "figure":
                pages[pno].append(PageBlock(
                    type="figure",
                    id=block.id,
                    reading_order=block.reading_order,
                    section=block.section or None,
                    caption=block.caption,
                    image_path=block.image_path,
                    previous_text_id=block.previous_text_id,
                    next_text_id=block.next_text_id,
                ))

        return [PageOutput(page=pno, document_id=doc_id, blocks=pages[pno])
                for pno in sorted(pages.keys())]

    def _save_figures(self, result: ParsedDocument, figures_dir: Path) -> None:
        from PIL import Image as PILImage
        fmt = self.config.figure_format.upper().replace("JPG", "JPEG")

        for fig in result.figure_blocks:
            img_data = self._figure_images.get(fig.id)
            if img_data is not None:
                try:
                    dest = figures_dir / Path(fig.path).name
                    if isinstance(img_data, str):
                        import base64
                        raw = base64.b64decode(img_data)
                        pil = PILImage.open(io.BytesIO(raw))
                        buf = io.BytesIO(); pil.save(buf, format=fmt)
                        dest.write_bytes(buf.getvalue())
                    else:
                        buf = io.BytesIO(); img_data.save(buf, format=fmt)
                        dest.write_bytes(buf.getvalue())
                        pil = img_data
                    fig.width      = pil.width
                    fig.height     = pil.height
                    fig.image_hash = hashlib.sha256(dest.read_bytes()).hexdigest()
                    for po in result.page_outputs:
                        for pb in po.blocks:
                            if pb.id == fig.id and pb.type == "figure":
                                pb.width = fig.width; pb.height = fig.height
                                pb.image_hash = fig.image_hash
                except Exception as exc:
                    self.logger.warning("Could not save figure %s: %s", fig.id, exc)

        write_json(
            [fig.model_dump(exclude_none=True) for fig in result.figure_blocks],
            figures_dir / "figure_metadata.json",
        )
        self.logger.info("Wrote figure_metadata.json with %d entries.", len(result.figure_blocks))

    @staticmethod
    def _marker_version() -> Optional[str]:
        try:
            import importlib.metadata
            return importlib.metadata.version("marker-pdf")
        except Exception:
            return None
