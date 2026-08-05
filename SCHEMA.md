# TechDocs Copilot — Parser Schema v1.0

> **Status: FROZEN**
> Do not add fields to this schema without updating this document.
> The next stage is the structure-aware chunker — stop here.

---

## Project Data Layout

```
data/
├── raw/
│   └── manuals/                     ← original PDF files (source of truth, never modified)
│
├── parsed/
│   ├── docling_raw/                 ← Stage 1: raw Docling parser output (frozen snapshot)
│   │   └── printer_manual/
│   │       ├── document.json
│   │       ├── metadata.json
│   │       ├── pages/
│   │       ├── tables/
│   │       └── figures/
│   │
│   └── parser_v1/                   ← Stage 2: post-processed, enriched output (Schema v1.0)
│       └── printer_manual/
│           ├── document.json
│           ├── metadata.json
│           ├── pages/               ← page_stats, all block enrichments applied
│           ├── tables/
│           └── figures/
│
├── chunks/                          ← Stage 3: chunker output (next stage)
├── embeddings/                      ← Stage 4: embedding vectors
├── vectordb/                        ← Stage 5: vector database index
├── evaluation/                      ← Evaluation results and metrics
└── experiments/                     ← Experiment logs and ablations
```

**Reproducibility guarantee:** `docling_raw/` is never overwritten after the initial parse run. If the chunker or embedding model changes, only stages 3–5 need to be re-run. If the post-processor changes, only `parser_v1/` needs to be regenerated from `docling_raw/`.

### Two-stage workflow

```bash
# Stage 1 — Parse  (run once, ~4 minutes, output frozen)
python scripts/run_parser.py \
    --pdf  data/raw/manuals/01_HP_LaserJet_Repair_Manual_770p.pdf \
    --output data/parsed/docling_raw/printer_manual

# Stage 2 — Post-process  (fast, re-runnable, ~17 seconds)
cp -R data/parsed/docling_raw/printer_manual \
      data/parsed/parser_v1/printer_manual

python scripts/run_post_processor.py \
    --output data/parsed/parser_v1/printer_manual
```

---

## Per-Document Output Layout (inside parser_v1/<document>/)

```
<document>/
├── document.json            ← lightweight document index
├── metadata.json            ← parser provenance + timing
├── pages/
│   ├── page_001.json        ← ordered blocks + page_stats
│   ├── page_002.json
│   └── ...
├── tables/
│   └── tbl_XXXX.json        ← full table with markdown + bbox
└── figures/
    ├── figure_metadata.json ← enriched figure metadata list
    ├── figure_0001.png
    └── ...
```

---

## metadata.json

Parser provenance. Written at parse time; `processing_time_seconds` backfilled after save.

| Field | Type | Description |
|---|---|---|
| `document_id` | string | SHA-256 prefix of the PDF filename. Stable across re-parses of the same file. |
| `filename` | string | Original PDF filename. |
| `title` | string\|null | First heading found in pages 1–5. |
| `total_pages` | int | Total PDF pages (including blank/image-only). |
| `parser` | string | Parser implementation name (`"DoclingParser"`). |
| `parser_version` | string\|null | Installed `docling` package version. |
| `parsed_at` | string | UTC ISO-8601 timestamp of parse start. |
| `processing_time_seconds` | float\|null | Wall-clock parse + save time in seconds. |

---

## document.json

Lightweight index. Does **not** contain block text — use page files for content.

| Field | Type | Description |
|---|---|---|
| `document_id` | string | Same as metadata. |
| `title` | string | Document title. |
| `pages` | int | Total PDF pages. |
| `statistics.text_blocks` | int | Total text blocks across all pages. |
| `statistics.tables` | int | Total table blocks. |
| `statistics.figures` | int | Total figure blocks. |
| `page_index` | array | `[{page: N, path: "pages/page_NNN.json"}]` — navigation map. |

---

## pages/page_NNN.json

One file per parsed page. Primary consumption unit for the chunker.

### Page envelope

| Field | Type | Description |
|---|---|---|
| `page` | int | 1-based page number. |
| `document_id` | string | Parent document ID. |
| `page_bbox` | object\|null | `{width, height}` in PDF points (1 pt = 1/72 in). US Letter = 594.72 × 792.0. |
| `blocks` | array | Reading-order list of `PageBlock` objects. |
| `page_stats` | object | Block composition summary. |

### page_stats

| Field | Type | Description |
|---|---|---|
| `block_count` | int | **Total blocks on this page** (all types). |
| `text_blocks` | int | **All textual blocks** — paragraph + heading + list_item + caption + admonition + reference + footnote + code + unknown. Any block a chunker would read. |
| `paragraphs` | int | `type == "paragraph"` count. |
| `headings` | int | `type == "heading"` count. |
| `list_items` | int | `type == "list_item"` count. |
| `captions` | int | `type == "caption"` count. |
| `admonitions` | int | `type == "admonition"` count. |
| `references` | int | `type == "reference"` count. |
| `figures` | int | `type == "figure"` count (including decorative). |
| `tables` | int | `type == "table"` count. |
| `toc_tables` | int | `type == "toc"` count. |

> **Invariant:** `block_count == text_blocks + figures + tables + toc_tables`

---

## Block Types

| `type` | Description |
|---|---|
| `paragraph` | Body text. |
| `heading` | Section heading. Carries `level` and `role`. |
| `list_item` | Single item from a bulleted or numbered list. Numbered items matching `^\d+[.)]\s+` carry `role: "procedure_step"`. |
| `caption` | Figure or table caption. Carries `caption_for`. `content` = original PDF text, number prefix included. |
| `admonition` | Safety or informational callout. Carries `severity`. |
| `figure` | Extracted image. |
| `table` | Extracted data table rendered as markdown. |
| `toc` | Table of Contents navigation block (dot-leader pattern). Excluded from chunking. |
| `reference` | Standalone URL line (`https://`, `http://`, or `www.` prefix). |
| `footnote` | Page footnote. |
| `code` | Code or pre-formatted text. |
| `unknown` | Unclassified by Docling. Treat as paragraph. |

---

## Common Fields (all block types)

| Field | Type | Description |
|---|---|---|
| `type` | string | Block type (see above). |
| `id` | string | Document-scoped unique ID. Format: `txt_XXXX`, `fig_XXXX`, `tbl_XXXX`. |
| `reading_order` | int | 0-based global reading-order index. Monotonically increasing across the full document. |
| `section` | string\|null | Human-readable text of the nearest ancestor heading at parse time. Example: `"Removal and replacement procedures"`. |
| `bbox` | object\|null | `{l, t, r, b}` bounding box in PDF points. Origin bottom-left; `t > b`. |

---

## Text Block Fields

Applies to: `paragraph`, `heading`, `list_item`, `caption`, `admonition`, `reference`, `footnote`, `code`, `unknown`.

| Field | Type | Description |
|---|---|---|
| `content` | string | Verbatim text from the PDF. Never synthesized. |
| `level` | int\|null | **Headings only.** 1 = document title, 2 = section, 3 = subsection. |
| `role` | string\|null | Sub-role. Headings: `"chapter"` / `"section"` / `"subsection"`. Numbered list items: `"procedure_step"`. |
| `severity` | string\|null | **Admonitions only.** `"warning"` / `"caution"` / `"note"` / `"tip"` / `"important"`. |
| `icon_path` | string\|null | **Merged admonitions only.** Relative path to the icon figure merged into this block. |
| `caption_for` | string\|null | **Caption blocks only.** ID of the figure or table this caption describes. |
| `hyperlink_hint` | bool\|null | `true` when the block contains anchor text that likely had a PDF hyperlink (e.g. "view a video"). Target URL is not preserved — flag only. |

---

## Figure Block Fields

| Field | Type | Description |
|---|---|---|
| `image_path` | string | Relative path to the saved PNG. |
| `figure_number` | string\|null | Structured identifier extracted from caption. Example: `"Figure 1-6"`. Preserved exactly as in the PDF. |
| `caption` | string\|null | **Descriptive text only** — figure number prefix stripped. Example: `"Open the toner-cartridge door"`. |
| `caption_id` | string\|null | ID of the `caption` block. Bidirectional with `caption.caption_for`. |
| `decorative` | bool\|null | `true` if likely a small icon (bbox area < 2,000 pt² near a dominant figure). Not a definitive classification — use `width`/`height` for filtering. |
| `width` | int\|null | Image width in pixels. |
| `height` | int\|null | Image height in pixels. |
| `image_hash` | string\|null | SHA-256 hex of PNG bytes. Use to detect duplicate images. |
| `previous_block_id` | string\|null | ID of the nearest preceding block in reading order (any type). |
| `next_block_id` | string\|null | ID of the nearest following block in reading order (any type). |

### Caption convention

| Location | Content |
|---|---|
| `figure.caption` | Descriptive text only (number stripped). For display and retrieval. |
| `caption.content` | Original PDF text including number (e.g. `"Figure 1-4 Open the toner-cartridge door"`). For exact reproduction. |

---

## Table Block Fields

| Field | Type | Description |
|---|---|---|
| `markdown` | string | Full table as GitHub-flavored markdown. |
| `title` | string\|null | Title from adjacent caption. Full text including number. |
| `table_number` | string\|null | Structured number from title. Example: `"Table 2-1"`. Preserved exactly as in the PDF. |
| `caption_id` | string\|null | ID of the `caption` block that describes this table. |
| `rows` | int\|null | Data row count (separator rows excluded). |
| `cols` | int\|null | Column count (from first data row). |
| `toc` | bool\|null | `true` only on `type == "toc"` blocks — navigation table, not data. |

---

## figures/figure_metadata.json

Array of enriched figure records. One entry per figure across the whole document.

| Field | Type | Description |
|---|---|---|
| `id` | string | `fig_XXXX` |
| `page` | int | 1-based page number. |
| `section` | string\|null | Nearest ancestor heading. |
| `caption` | string\|null | Descriptive caption (number stripped). |
| `figure_number` | string\|null | Structured identifier. |
| `path` | string | Relative path to PNG. |
| `image_path` | string | Alias for `path`. |
| `image_hash` | string\|null | SHA-256 of PNG bytes. |
| `width` | int\|null | Pixels. |
| `height` | int\|null | Pixels. |
| `bbox` | object\|null | `{l, t, r, b}` in PDF points. |
| `reading_order` | int | Global reading-order index. |

---

## Known Behaviours (Not Bugs)

### 1. Inline procedural notes vs. boxed admonitions

```json
{"type": "list_item", "content": "NOTE: HP recommends ..."}
```
vs.
```json
{"type": "admonition", "severity": "note", "content": "NOTE: ..."}
```

Both are correct. Docling classifies **visually boxed** admonitions as `admonition`. An inline note appearing as an indented item inside a procedure step list stays as `list_item`. The post-processor does not forcibly normalize these — it preserves Docling's layout-based classification.

### 2. Decorative figures

Marked `decorative: true` when:
- bbox area < 2,000 pt² **and**
- A dominant figure ≥ 8× larger exists within 400 pt centroid distance.

The parser makes no claim about icon identity. `decorative` means "probably not a semantic figure." Use `width < 20 and height < 20` for programmatic filtering.

### 3. 8 missing pages

Pages 2, 10, 12, 721, 725, 729, 741, 766 produce no JSON (blank or background-image-only in the source PDF). Confirmed by manual inspection. No content is lost.

### 4. TOC tables

Pages 7, 8, 9, 11 are Table of Contents pages. Tables on these pages are classified `type: "toc"`. Exclude from chunking with `block["type"] != "toc"`.

---

## ID Conventions

| Prefix | Scope | Format |
|---|---|---|
| `doc_` | document | `doc_` + first 8 chars of SHA-256 of filename |
| `txt_` | text blocks | `txt_` + zero-padded 4-digit counter (per document) |
| `fig_` | figures | `fig_` + zero-padded 4-digit counter (per document) |
| `tbl_` | tables | `tbl_` + zero-padded 4-digit counter (per document) |

IDs are **document-scoped** — unique within one document, not globally unique across documents.

---

## Changelog

| Version | Date | Changes |
|---|---|---|
| v1.0 | 2026-08-05 | Frozen. 6 must-fix bugs resolved, 7 should-fix enrichments added. `block_count` in `page_stats`. `previous_block_id`/`next_block_id` replaces `previous_text_id`/`next_text_id`. Caption convention documented. Known behaviours documented. |
