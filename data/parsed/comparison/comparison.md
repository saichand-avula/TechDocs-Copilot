# Parser Comparison — HP LaserJet Repair Manual (770 pages)
## Schema v2.0 — Enriched Metadata & Reading-Order Blocks

Both parsers now output the same **schema v2.0** structure with enriched metadata.
Comparison is run on the same document: `01_HP_LaserJet_Repair_Manual_770p.pdf`

---

## 1. Quick Summary

| Dimension | Docling | Marker |
|---|---|---|
| **Speed** | **219s** | 272s |
| **Text blocks** | **5,639** | 3,031 |
| **List items** | **2,491** | ~1 |
| **Tables** | **94** | 72 |
| **Table accuracy** | **Clean cells** | Merged columns |
| **Figures detected** | **2,045** | 800 |
| **Figures saved (PNG)** | **2,045** | 800 |
| **Page coverage** | **762/770 (99%)** | 699/770 (91%) |
| **Section context** | **Human-readable text** | Raw block IDs |
| **Setup complexity** | pip only | pip + brew llama.cpp |

**Verdict: Docling wins on every meaningful dimension.**

---

## 2. Schema v2.0 — What Both Parsers Now Output

Both parsers share the same canonical output structure:

```
output_dir/
    document.json           ← lightweight index (stats + page_index only)
    metadata.json           ← parser info + processing_time_seconds
    pages/
        page_001.json       ← ordered blocks list (text inline, fig/tbl as refs)
        page_002.json
        …
    tables/
        tbl_0001.json       ← full table with markdown + bbox + parent_section
        …
    figures/
        figure_metadata.json ← all figures with hash + dimensions
        figure_0001.png
        …
```

### document.json (lightweight)
```json
{
  "document_id": "doc_7e4fe272",
  "title": "Service Manual: Repair",
  "pages": 770,
  "statistics": {"text_blocks": 5639, "tables": 94, "figures": 2045},
  "page_index": [{"page": 1, "path": "pages/page_001.json"}, …]
}
```
No paragraph duplication. Pure lightweight index.

### page_031.json (reading-order blocks)
```json
{
  "page_number": 31,
  "blocks": [
    {"type": "list_item", "id": "txt_0299", "reading_order": 351,
     "parent_heading": "Post service test", "content": "…",
     "previous_block_id": "txt_0298", "next_block_id": "txt_0300"},
    {"type": "figure", "id": "fig_0046", "reading_order": 354,
     "image_path": "figures/figure_0046.png", "caption": "…"},
    {"type": "table",  "id": "tbl_0012", "reading_order": 355,
     "parent_section": "Post service test", "markdown": "| … |"}
  ]
}
```

### figure_metadata entry
```json
{
  "id": "fig_0001",
  "page": 1,
  "reading_order": 0,
  "parent_section": null,
  "caption": null,
  "width": 67,
  "height": 65,
  "image_hash": "50a1692c7d5fd1079c3119c585baf2a7f17d5a06b55cafb7506a070667c974a5",
  "previous_text_block": null,
  "next_text_block": "txt_0001"
}
```

---

## 3. Text Extraction

| Metric | Docling | Marker |
|---|---|---|
| Total text blocks | **5,639** | 3,031 |
| Paragraph blocks | ~2,600 | ~2,900 |
| **List items** | **2,491** | **≈ 1** |
| Headings | 548 | 130 |
| Captions | 1,192 | 215 |

> [!IMPORTANT]
> A repair manual without list items is **unusable for RAG**. Marker treats procedure steps as paragraphs. Docling classifies them correctly as `list_item`, which is critical for step-aware chunking.

---

## 4. Table Extraction

| Metric | Docling | Marker |
|---|---|---|
| Tables detected | **94** | 72 |
| Accuracy | **Clean separated cells** | Merged/misaligned columns |
| With bbox | ✅ Yes (l, t, r, b) | ❌ Not in JSON mode |
| With parent_section | ✅ Human-readable | ⚠️ Raw block ID |

**Example (same table, page 31):**

Docling output:
```
| Component | Part Number | Description |
|-----------|-------------|-------------|
| Fuser     | RG5-7028    | 110V fuser  |
```

Marker output:
```
| Component Part Number | Description |
|---|---|
| Fuser RG5-7028 | 110V fuser |
```

---

## 5. Figure Extraction

| Metric | Docling | Marker |
|---|---|---|
| Figures detected | **2,045** | 800 |
| PNGs saved | **2,045** | 800 |
| Caption linked | **1,192** | 215 |
| image_hash (SHA-256) | ✅ All 2,045 | ✅ All 800 |
| Pixel dimensions | ✅ Width + Height | ✅ Width + Height |
| bbox | ✅ l, t, r, b | ❌ Not available |
| previous/next text block | ✅ | ✅ |

Docling extracts **155% more figures** (2,045 vs 800). On a diagram-heavy repair manual this is critical — every wiring diagram and part schematic is captured.

---

## 6. Section Context Quality

| Aspect | Docling | Marker |
|---|---|---|
| parent_heading format | **"Post service test"** | `/page/29/SectionHeader/14` |
| Chunker-usable | ✅ Immediately | ❌ Requires ID resolution |
| heading_level | ✅ 1 (title), 2 (section) | ⚠️ Always 2 (no depth) |

Marker's `section_hierarchy` stores internal block references (IDs like `/page/29/SectionHeader/14`), not resolved text. The chunker would need an extra lookup step to resolve these, adding complexity and potential failure points.

---

## 7. Reading Order

Both parsers build the unified `blocks` list in reading order. Each text block carries:
- `reading_order` — 0-based global position across the whole document
- `previous_block_id` / `next_block_id` — exact neighbor links

| Aspect | Docling | Marker |
|---|---|---|
| Global reading order | ✅ 0–7,778 | ✅ 0–3,902 |
| prev/next text links | ✅ | ✅ |
| Figure → nearest text | ✅ | ✅ |
| Table bbox | ✅ | ❌ |

---

## 8. Page Coverage

| | Docling | Marker |
|---|---|---|
| Pages in PDF | 770 | 770 |
| Page JSON files | **762** | **699** |

Marker produced 699 page files — **71 pages had no extractable content** (diagram-only pages where OCR produced nothing). Docling covered 762 pages (only 8 truly blank pages missed).

---

## 9. Speed (schema v2.0, CPU mode)

| | Docling | Marker |
|---|---|---|
| Parse + save | **219s** | 272s |
| Advantage | **+24% faster** | — |

Both now run on CPU (MPS disabled). Docling is still faster despite extracting 86% more text blocks and 155% more figures.

---

## 10. Head-to-Head Summary

| Dimension | Docling | Marker |
|---|---|---|
| Speed | **Faster (219s)** | Slower (272s) |
| Text richness | **5,639 blocks** | 3,031 blocks |
| List detection | **2,491 items** | ≈ 1 item |
| Table count | **94 tables** | 72 tables |
| Table accuracy | **Clean cells** | Merged columns |
| Table bbox | ✅ | ❌ |
| Figure count | **2,045** | 800 |
| Figure export | **2,045 PNGs** | 800 PNGs |
| Section context | **Human-readable** | Raw block IDs |
| Heading levels | **1, 2, 3…** | Always 2 |
| Page coverage | **762/770 (99%)** | 699/770 (91%) |
| SHA-256 hash | ✅ | ✅ |
| Pixel dimensions | ✅ | ✅ |
| Reading order | ✅ | ✅ |
| prev/next links | ✅ | ✅ |
| Setup complexity | **pip only** | pip + brew llama.cpp |

---

## 11. Recommendation

### Use Docling for all 11 manuals

Docling wins on **every meaningful dimension** for this project:

1. **More complete extraction** — 86% more text blocks, 31% more tables, 155% more figures
2. **Higher accuracy** — Table cells correctly separated, list items correctly classified
3. **RAG-ready metadata** — Section context is human-readable, immediately usable by the chunker
4. **Superior figure coverage** — 2,045 vs 800 figures; 1,192 vs 215 captions linked
5. **Better structure** — Heading levels (H1/H2/H3) vs Marker's flat H2 only
6. **Faster** — 219s vs 272s despite significantly more content extracted
7. **Simpler setup** — No llama.cpp installation required

Both parsers now output identical **schema v2.0** enriched format with reading order, parent heading, bbox, SHA-256 hash, pixel dimensions, and prev/next block links. The quality difference is purely in extraction accuracy and completeness.
