"""
docling_parser.py
=================
DoclingParser — concrete implementation of the Parser ABC using IBM Docling.

Schema v2.1 — Final Frozen Output
-----------------------------------
- PageBlock carries only its own fields (no nulls for irrelevant types)
- document_id and page stored once at PageOutput level
- Unified "section" field (no parent_heading / parent_section split)
- Figures carry previous_text_id / next_text_id for context-based retrieval
- Page files serialized with exclude_none=True → clean minimal JSON
"""

from __future__ import annotations

import hashlib
import io
import logging
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

# Docling imports — isolated here intentionally
try:
    from docling.datamodel.base_models import InputFormat
    from docling.datamodel.pipeline_options import (
        AcceleratorDevice,
        AcceleratorOptions,
        EasyOcrOptions,
        PdfPipelineOptions,
        TableFormerMode,
    )
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling_core.types.doc import (
        DocItemLabel,
        DoclingDocument,
        PictureItem,
        TableItem,
        TextItem,
    )
    _DOCLING_AVAILABLE = True
except ImportError:
    _DOCLING_AVAILABLE = False


# ---------------------------------------------------------------------------
# Label helpers
# ---------------------------------------------------------------------------

_LABEL_TO_BLOCK_TYPE: Dict[str, BlockType] = {
    "title":          "heading",
    "section_header": "heading",
    "text":           "paragraph",
    "list_item":      "list_item",
    "caption":        "caption",
    "footnote":       "footnote",
    "page_header":    "page_header",
    "page_footer":    "page_footer",
    "code":           "code",
}

_HEADING_LABELS = {"title", "section_header"}


def _block_type(label: str) -> BlockType:
    return _LABEL_TO_BLOCK_TYPE.get(label.lower(), "unknown")


def _heading_level(label: str, depth: int) -> Optional[int]:
    l = label.lower()
    if l == "title":
        return 1
    if l == "section_header":
        return max(2, depth)
    return None


def _bbox(item: object) -> Optional[Dict[str, float]]:
    try:
        prov = getattr(item, "prov", None)
        if prov and len(prov) > 0:
            b = prov[0].bbox
            if b is not None:
                return {"l": float(b.l), "t": float(b.t),
                        "r": float(b.r), "b": float(b.b)}
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# DoclingParser
# ---------------------------------------------------------------------------

class DoclingParser(Parser):
    """
    Production document parser backed by IBM Docling.

    Usage
    -----
    >>> parser = DoclingParser(config)
    >>> result = parser.parse(Path("manual.pdf"))
    >>> parser.save(result, Path("output/"))
    """

    PARSER_NAME = "DoclingParser"

    def __init__(self, config: ParserConfig) -> None:
        if not _DOCLING_AVAILABLE:
            raise ImportError("docling not installed. Run: pip install 'docling[full]'")
        self.config = config
        self.logger: logging.Logger = setup_logger(
            __name__, level=config.log_level, log_file=config.log_file
        )
        self._converter: Optional[DocumentConverter] = None
        self._figure_images: Dict[str, object] = {}
        self._wall_start: float = 0.0

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def parse(self, pdf_path: Path) -> ParsedDocument:
        self._wall_start = time.time()
        pdf_path = Path(pdf_path).expanduser().resolve()
        self.logger.info("Starting parse: %s", pdf_path.name)

        doc: DoclingDocument = self._get_converter().convert(str(pdf_path)).document
        self.logger.info("Docling conversion complete.")

        doc_id = make_document_id(pdf_path)

        # ── Counters ──────────────────────────────────────────────────
        text_ctr = fig_ctr = tbl_ctr = 0
        self._figure_images = {}

        # ── Unified reading-order list: ("text"|"table"|"figure", block)
        all_blocks: List[Tuple[str, object]] = []

        # ── Section tracking ──────────────────────────────────────────
        current_section:       Optional[str] = None
        current_section_level: Optional[int] = None

        # ── Iterate Docling items ─────────────────────────────────────
        for item, depth in doc.iterate_items():
            label = str(getattr(item, "label", "")).lower()
            page_no = self._page_no(item)
            item_bbox = _bbox(item)
            ro = len(all_blocks)

            if label in _HEADING_LABELS:
                text = self._text(item)
                if not text:
                    continue
                lvl = _heading_level(label, depth)
                current_section       = text
                current_section_level = lvl
                text_ctr += 1
                all_blocks.append(("text", TextBlock(
                    id=make_text_id(text_ctr),
                    type="heading",
                    page=page_no,
                    section=current_section,
                    content=text,
                    block_id=self._block_id(item),
                    reading_order=ro,
                    level=lvl,
                    bbox=item_bbox,
                )))

            elif isinstance(item, TextItem):
                text = self._text(item)
                if not text:
                    continue
                text_ctr += 1
                all_blocks.append(("text", TextBlock(
                    id=make_text_id(text_ctr),
                    type=_block_type(label),
                    page=page_no,
                    section=current_section,
                    content=text,
                    block_id=self._block_id(item),
                    reading_order=ro,
                    bbox=item_bbox,
                )))

            elif isinstance(item, TableItem) and self.config.extract_tables:
                tbl_ctr += 1
                all_blocks.append(("table", TableBlock(
                    id=make_table_id(tbl_ctr),
                    page=page_no,
                    section=current_section,
                    markdown=self._table_md(item, doc),
                    block_id=self._block_id(item),
                    reading_order=ro,
                    bbox=item_bbox,
                )))

            elif isinstance(item, PictureItem) and self.config.extract_figures:
                fig_ctr += 1
                fig_id = make_figure_id(fig_ctr)
                rel_path = f"figures/{make_figure_filename(fig_ctr, self.config.figure_format)}"
                try:
                    pil = item.get_image(doc)
                    if pil is not None:
                        self._figure_images[fig_id] = pil
                except Exception:
                    pass
                all_blocks.append(("figure", FigureBlock(
                    id=fig_id,
                    page=page_no,
                    section=current_section,
                    caption=self._caption(item),
                    path=rel_path,
                    block_id=self._block_id(item),
                    reading_order=ro,
                    bbox=item_bbox,
                    image_path=rel_path,
                )))

        # ── Link figure reading context ────────────────────────────────
        self._link_figure_context(all_blocks)

        # ── Separate typed lists ───────────────────────────────────────
        text_blocks   = [b for t, b in all_blocks if t == "text"]
        table_blocks  = [b for t, b in all_blocks if t == "table"]
        figure_blocks = [b for t, b in all_blocks if t == "figure"]

        # ── Page outputs ───────────────────────────────────────────────
        page_bboxes = self._page_sizes(doc)
        page_outputs = self._build_pages(all_blocks, doc_id, page_bboxes)

        # ── Document index ─────────────────────────────────────────────
        total_pages = len(doc.pages) if doc.pages else 0
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
            parser_version=self._docling_version(),
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

        # 1. document.json (lightweight)
        write_json(result.document, dirs["root"] / "document.json")
        self.logger.info("Wrote document.json")

        # 2. metadata.json
        write_json(result.metadata, dirs["root"] / "metadata.json")
        self.logger.info("Wrote metadata.json")

        # 3. Page files — exclude_none=True for clean output
        for po in result.page_outputs:
            data = {
                "page":        po.page,
                "document_id": po.document_id,
                "page_bbox":   po.page_bbox,
                "blocks": [b.model_dump(exclude_none=True) for b in po.blocks],
            }
            if data["page_bbox"] is None:
                del data["page_bbox"]
            path = dirs["pages"] / f"page_{po.page:03d}.json"
            path.write_text(__import__("json").dumps(data, indent=2, ensure_ascii=False))
        self.logger.info("Wrote %d page files.", len(result.page_outputs))

        # 4. Table files
        for tbl in result.table_blocks:
            write_json(tbl, dirs["tables"] / f"{tbl.id}.json")
        self.logger.info("Wrote %d table files.", len(result.table_blocks))

        # 5. Figures (images + metadata)
        self._save_figures(result, dirs["figures"])

        # 6. Update processing_time_seconds
        result.metadata.processing_time_seconds = round(time.time() - self._wall_start, 1)
        write_json(result.metadata, dirs["root"] / "metadata.json")

        self.logger.info("Save complete: %s", output_dir)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _get_converter(self) -> "DocumentConverter":
        if self._converter is not None:
            return self._converter
        self.logger.info("Initialising Docling DocumentConverter…")
        pipeline_options = PdfPipelineOptions(
            accelerator_options=AcceleratorOptions(
                num_threads=4, device=AcceleratorDevice.CPU,
            )
        )
        pipeline_options.do_ocr             = self.config.do_ocr
        pipeline_options.do_table_structure = self.config.do_table_structure
        pipeline_options.generate_picture_images = self.config.extract_figures
        if self.config.do_table_structure:
            pipeline_options.table_structure_options.mode = TableFormerMode.ACCURATE
            self.logger.info("Table structure mode: ACCURATE")
        if self.config.do_ocr:
            pipeline_options.ocr_options = EasyOcrOptions(force_full_page_ocr=False)
        self._converter = DocumentConverter(
            format_options={InputFormat.PDF: PdfFormatOption(pipeline_options=pipeline_options)}
        )
        self.logger.info("DocumentConverter ready (device=CPU).")
        return self._converter

    def _page_no(self, item: object) -> int:
        try:
            prov = getattr(item, "prov", None)
            if prov and len(prov) > 0:
                return int(prov[0].page_no)
        except Exception:
            pass
        return 1

    def _block_id(self, item: object) -> Optional[str]:
        sr = getattr(item, "self_ref", None)
        return str(sr) if sr else None

    def _text(self, item: object) -> str:
        try:
            return (item.text or "").strip()
        except Exception:
            pass
        try:
            return item.export_to_markdown().strip()
        except Exception:
            return ""

    def _table_md(self, item: "TableItem", doc: "DoclingDocument") -> str:
        try:
            return item.export_to_markdown(doc).strip()
        except TypeError:
            try:
                return item.export_to_markdown().strip()
            except Exception:
                return ""

    def _caption(self, item: object) -> Optional[str]:
        for cap_ref in (getattr(item, "captions", None) or []):
            try:
                t = getattr(cap_ref.resolve(), "text", None)
                if t:
                    return t.strip()
            except Exception:
                pass
        return None

    def _link_figure_context(self, all_blocks: List[Tuple[str, object]]) -> None:
        """
        Fill previous_block_id / next_block_id on every FigureBlock.

        Uses the *immediate* reading-order neighbours (position pos-1 and pos+1)
        regardless of block type.  The schema contract is:

            previous_block_id = the block immediately before this one
            next_block_id     = the block immediately after this one

        Previously this searched only through text-typed blocks, which violated
        the contract and produced dangling refs after post-processing removed
        non-text blocks.
        """
        for pos, (btype, block) in enumerate(all_blocks):
            if btype != "figure":
                continue
            block.previous_block_id = (
                all_blocks[pos - 1][1].id if pos > 0 else None
            )
            block.next_block_id = (
                all_blocks[pos + 1][1].id if pos < len(all_blocks) - 1 else None
            )

    def _page_sizes(self, doc: "DoclingDocument") -> Dict[int, Dict[str, float]]:
        """Extract {page_no: {width, height}} from Docling page objects."""
        sizes: Dict[int, Dict[str, float]] = {}
        try:
            for page_no, page_obj in (doc.pages or {}).items():
                size = getattr(page_obj, "size", None)
                if size is not None:
                    sizes[int(page_no)] = {
                        "width":  float(getattr(size, "width",  0)),
                        "height": float(getattr(size, "height", 0)),
                    }
        except Exception:
            pass
        return sizes

    def _build_pages(
        self,
        all_blocks: List[Tuple[str, object]],
        doc_id: str,
        page_bboxes: Dict[int, Dict[str, float]],
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
                    level=block.level,       # None for non-headings → excluded by exclude_none
                    bbox=block.bbox,
                ))
            elif btype == "table":
                pages[pno].append(PageBlock(
                    type="table",
                    id=block.id,
                    reading_order=block.reading_order,
                    section=block.section or None,
                    bbox=block.bbox,
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
                    bbox=block.bbox,
                    previous_block_id=block.previous_block_id,
                    next_block_id=block.next_block_id,
                ))

        return [
            PageOutput(
                page=pno,
                document_id=doc_id,
                page_bbox=page_bboxes.get(pno),
                blocks=pages[pno],
            )
            for pno in sorted(pages.keys())
        ]

    def _save_figures(self, result: ParsedDocument, figures_dir: Path) -> None:
        fmt = self.config.figure_format.upper()
        if fmt == "JPG":
            fmt = "JPEG"

        for fig in result.figure_blocks:
            pil = self._figure_images.get(fig.id)
            if pil is not None:
                try:
                    dest = figures_dir / Path(fig.path).name
                    buf  = io.BytesIO()
                    pil.save(buf, format=fmt)
                    img_bytes = buf.getvalue()
                    dest.write_bytes(img_bytes)
                    fig.width      = pil.width
                    fig.height     = pil.height
                    fig.image_hash = hashlib.sha256(img_bytes).hexdigest()
                    # Backfill into page block
                    for po in result.page_outputs:
                        for pb in po.blocks:
                            if pb.id == fig.id and pb.type == "figure":
                                pb.width      = fig.width
                                pb.height     = fig.height
                                pb.image_hash = fig.image_hash
                except Exception as exc:
                    self.logger.warning("Could not save figure %s: %s", fig.id, exc)

        write_json(
            [fig.model_dump(exclude_none=True) for fig in result.figure_blocks],
            figures_dir / "figure_metadata.json",
        )
        self.logger.info("Wrote figure_metadata.json with %d entries.", len(result.figure_blocks))

    def _title(self, text_blocks: List[TextBlock]) -> Optional[str]:
        for b in text_blocks:
            if b.type == "heading" and b.page <= 5:
                return b.content
        return None

    @staticmethod
    def _docling_version() -> Optional[str]:
        try:
            import importlib.metadata
            return importlib.metadata.version("docling")
        except Exception:
            return None
