# TechDocs Copilot — Document Parsing (Stage 1)

A production-grade document parser for the **TechDocs Copilot** Multimodal RAG system.

Stage 1 converts raw technical PDFs into a structured, page-centric JSON representation using **IBM Docling** as the extraction engine, behind a clean abstract interface.

---

## Project Structure

```
TechDocs Copilot/
├── src/
│   └── parsing/
│       ├── __init__.py           # Package exports
│       ├── parser_interface.py   # Abstract Parser base + Pydantic models
│       ├── docling_parser.py     # DoclingParser implementation
│       ├── config.py             # ParserConfig (Pydantic settings)
│       └── utils.py              # Logging, JSON I/O, path helpers
├── scripts/
│   └── run_parser.py             # CLI entry point
├── dataset/
│   └── manuals/                  # Source PDFs (original location, never moved)
├── data/
│   └── parsed/                   # Parser output (auto-created)
├── requirements.txt
└── README.md
```

---

## Requirements

- Python 3.10+
- ~2–4 GB disk space for Docling AI model downloads (one-time, automatic on first run)

```bash
pip install -r requirements.txt
```

---

## Quick Start

Run the parser on the HP LaserJet manual:

```bash
python scripts/run_parser.py \
    --pdf  dataset/manuals/01_HP_LaserJet_Repair_Manual_770p.pdf \
    --output data/parsed/printer_manual
```

First run downloads Docling's layout and table-structure models from HuggingFace — this takes a few minutes.  
Subsequent runs are much faster.

---

## CLI Options

| Flag | Default | Description |
|---|---|---|
| `--pdf` | *(required)* | Path to input PDF |
| `--output` | *(required)* | Root output directory |
| `--no-tables` | off | Disable table extraction |
| `--no-figures` | off | Disable figure extraction |
| `--ocr` | off | Enable OCR (for scanned pages) |
| `--figure-format` | `png` | Image format: `png` or `jpg` |
| `--log-level` | `INFO` | `DEBUG \| INFO \| WARNING \| ERROR` |
| `--log-file` | none | Write logs to a file |

---

## Output Structure

```
data/parsed/printer_manual/
├── document.json            # Document-level index (all blocks)
├── metadata.json            # Filename, pages, title, parse timestamp
├── pages/
│   ├── page_001.json        # All blocks on page 1 (text, table refs, figure refs)
│   ├── page_002.json
│   └── ...
├── tables/
│   ├── tbl_0001.json        # Each table as Markdown
│   └── ...
└── figures/
    ├── figure_0001.png      # Extracted images as PNG
    ├── figure_0002.png
    └── figure_metadata.json # All figure metadata in one file
```

### document.json schema

```json
{
  "document_id": "doc_a3f1c2b4",
  "title": "HP LaserJet Repair Manual",
  "pages": 770,
  "text_blocks": [...],
  "tables": [...],
  "figures": [...]
}
```

### page_XXX.json schema

```json
{
  "page_number": 12,
  "text_blocks": [
    {
      "id": "txt_0042",
      "type": "paragraph",
      "page": 12,
      "section": "Removing the Fuser",
      "content": "Turn off the printer before...",
      "block_id": "#/texts/41"
    }
  ],
  "table_refs": ["tbl_0003"],
  "figure_refs": ["fig_0007"]
}
```

### table_XXXX.json schema

```json
{
  "id": "tbl_0003",
  "page": 42,
  "section": "Torque Specifications",
  "title": "Torque Specifications",
  "markdown": "| Screw | Torque |\n|---|---|\n| M3 | 0.5 Nm |",
  "block_id": "#/tables/2"
}
```

---

## Architecture

```
Parser (ABC)
└── DoclingParser
      ├── config:  ParserConfig
      ├── parse()  → ParsedDocument
      └── save()   → disk output

ParsedDocument
├── metadata:  DocumentMetadata
├── document:  DocumentOutput   (index of all blocks)
└── page_outputs: List[PageOutput]
```

The `Parser` ABC means any future parser (PyMuPDF, Marker, Unstructured) can be added by:
1. Subclassing `Parser`
2. Implementing `parse()` and `save()`
3. Returning the same Pydantic models

Zero changes to downstream pipeline stages.

---

## OCR Flag

The HP LaserJet manual is a digital PDF — OCR is **not** needed.  
For scanned manuals (e.g., older equipment), pass `--ocr` to activate EasyOCR.

Check `dataset/data_analysis/dataset_inventory.pdf` for the OCR column in the dataset inventory.

---

## Running All Manuals (Batch)

```bash
for pdf in dataset/manuals/*.pdf; do
  name=$(basename "$pdf" .pdf)
  python scripts/run_parser.py \
    --pdf "$pdf" \
    --output "data/parsed/$name"
done
```
