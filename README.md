# TechDocs Copilot — Multimodal RAG for Technical Manuals

A production-grade Retrieval-Augmented Generation (RAG) system that answers questions from technical equipment manuals. It handles part numbers, step-by-step procedures, page lookups, and vague queries — with grounded answers, inline citations, and relevant figures.

---

## What it does

```
Ask: "How do I replace the toner cartridge in the printer?"

╔══════════════════════════════════════════════════════════════╗
║  ANSWER                                                      ║
╠══════════════════════════════════════════════════════════════╣
║ To replace the toner cartridge:                              ║
║ 1. Turn the printer power off [1].                           ║
║ 2. Disconnect the power cable [1].                           ║
║ 3. Press the cartridge-door-release button [2].              ║
║ 4. Remove the old toner cartridge [2].                       ║
║ 5. Unpack and install the replacement cartridge [2].         ║
╚══════════════════════════════════════════════════════════════╝
╔══════════════════════════════════════════════════════════════╗
║  SOURCES                                                     ║
╠══════════════════════════════════════════════════════════════╣
║ [1] printer_manual  —  "Power off procedure"  —  Pages 20   ║
║ [2] printer_manual  —  "Toner cartridge removal"  —  Page 21║
╚══════════════════════════════════════════════════════════════╝
╔══════════════════════════════════════════════════════════════╗
║  RELATED FIGURES (2)                                         ║
╠══════════════════════════════════════════════════════════════╣
║   🖼️  fig_0029  (page 21)                                    ║
║      └─ figures/figure_0029.png                              ║
║   🖼️  fig_0030  (page 22)                                    ║
║      └─ figures/figure_0030.png                              ║
╚══════════════════════════════════════════════════════════════╝
```

---

## Supported manuals (11 manuals, 2,141 chunks)

| Manual | Domain |
|---|---|
| HP LaserJet Enterprise — Printer | Office equipment |
| Atlas Copco — Compressor | Industrial |
| Cummins — Generator | Power |
| Goulds Pump | Industrial |
| Haier — Air Conditioner | HVAC |
| Hospira — Infusion Pump | Medical |
| Hyundai — CNC Machine | Manufacturing |
| Merrychef — Commercial Oven | Foodservice |
| Nellcor — Patient Monitor | Medical |
| APC — UPS | Power |
| Whirlpool — Dishwasher | Consumer |

---

## Architecture

```
PDF Manuals
    │
    ▼ Stage 1: Parser (IBM Docling)
Structured JSON (text blocks, tables, figures, pages)
    │
    ▼ Stage 2: Chunker (semantic, layout-aware)
Chunks (heading-aware, table-intact, figure-linked)
    │
    ▼ Stage 3: Embedder (gemini-embedding-2, 1536-dim)
Float32 embeddings per manual
    │
    ▼ Stage 4: Indexer
ChromaDB (dense) + BM25Okapi (sparse) + ChunkStore (O(1) lookup)
    │
    ▼ Stage 5: Retriever
4-way query classifier → dense + BM25 → RRF → LLM rerank → generate
    │
    ▼
Grounded Answer + Inline Citations [1][2] + Figure paths
```

---

## Prerequisites

```bash
Python 3.10+
pip install -r requirements.txt
```

**API keys needed:**
- 5 × Gemini API keys (free tier) — for embeddings
- 1 × [Sarvam AI](https://sarvam.ai) API key — for reranking + generation

---

## Installation

```bash
git clone <repo>
cd "TechDocs Copilot"
pip install -r requirements.txt
```

---

## Quick start — Ask questions now

The indexes are already built. Just run:

```bash
python3 scripts/run_retriever.py
```

The Sarvam key is pre-configured. You'll see:

```
Loading ChunkStore and indexes… ready.

╔══════════════════════════════════════════════════════╗
║         TechDocs Copilot — Retrieval CLI             ║
╠══════════════════════════════════════════════════════╣
║  Model      : sarvam-105b                            ║
║  Threshold  : 0.5                                    ║
║  Top-K      : 5                                      ║
║  HyDE       : ON                                     ║
╚══════════════════════════════════════════════════════╝

Query >
```

### Example queries to try

```
# Exact page lookup (no embedding, instant)
Query > summarize page 3 of printer manual
Query > show me page 25 of haier manual

# Part number / code lookup (BM25 primary)
Query > CF289A
Query > Error E42

# Normal procedure question
Query > how to replace the toner cartridge in the printer?
Query > what is the oil change interval for the Cummins generator?
Query > how to calibrate the nellcor monitor?

# Short / vague (HyDE expansion kicks in)
Query > cartridge error
Query > pump cavitation
Query > infusion alarm
```

### Session commands

| Command | Effect |
|---|---|
| `/hyde off` | Disable HyDE expansion for A/B comparison |
| `/hyde on` | Re-enable HyDE |
| `/threshold 0.6` | Tighten relevance filter (fewer, higher-quality chunks) |
| `/threshold 0.3` | Loosen filter (more results) |
| `/type` | Show how the next query is classified before answering |
| `/quit` | Exit |

### CLI options

```bash
python3 scripts/run_retriever.py --help

  --sarvam-key KEY      Sarvam API key (or set SARVAM_API_KEY env var)
  --threshold 0.5       Relevance threshold, default 0.5
  --top-k 5             Max chunks to generator, default 5
  --no-hyde             Disable HyDE expansion
  --log-level INFO      DEBUG | INFO | WARNING | ERROR
```

---

## How the retrieval works

Each query is classified into one of 4 types:

| Type | Example | Path |
|---|---|---|
| **Page/metadata** | `"page 25 of printer manual"` | ChunkStore scan → generate (1 API call) |
| **Exact code** | `"CF289A"`, `"Error E42"` | BM25 + dense → RRF → rerank → generate (3 calls) |
| **Normal** | `"how to replace the toner?"` | Dense + BM25 → RRF → rerank → generate (3 calls) |
| **Vague** | `"cartridge error"` | HyDE → dense + BM25 → RRF → rerank → generate (4 calls) |

### Retrieval pipeline

```
Query
  ↓
Query Analyzer (regex, no API)
  ↓
[if vague] HyDE: Sarvam generates a hypothetical doc → embed that
  ↓
Dense search: Chroma top-15
+
BM25 search: top-15
  ↓
RRF fusion → top-8
  ↓
Sarvam reranker: score each chunk 1–10
  ↓
Relevance threshold filter (default ≥ 5/10)
  ↓
max 5 chunks → Sarvam answer generation
  ↓
Answer + citations + figure paths
```

---

## Running from scratch (pipeline stages)

If you want to rebuild from raw PDFs:

### Stage 1: Parse all PDFs

```bash
python3 scripts/run_parser.py \
    --pdf dataset/manuals/01_HP_LaserJet_Repair_Manual_770p.pdf \
    --output data/parsed/printer_manual
```

### Stage 2: Chunk all manuals

```bash
python3 scripts/run_chunker.py
```

### Stage 3: Embed all chunks

```bash
python3 scripts/run_embedder.py
```

> Uses 5 Gemini API keys in parallel (5 separate accounts × 90 RPM = 450 RPM total).
> Resumes automatically if interrupted.

### Stage 4: Build indexes

```bash
python3 scripts/run_indexer.py
```

Builds:
- `data/vectordb/chroma/` — ChromaDB dense index (2,141 vectors × 1536 dims)
- `data/vectordb/bm25/` — BM25Okapi sparse index (2,112 docs)

### Stage 5: Ask questions

```bash
python3 scripts/run_retriever.py
```

---

## Data directory structure

```
data/
├── chunks/
│   └── {manual_name}/
│       ├── chunks.json          # All chunks with text, tables, figure refs
│       └── manifest.json        # document_id, chunk count
├── embeddings/
│   └── {manual_name}/
│       ├── embeddings.npy       # float32 (N × 1536)
│       ├── chunk_ids.json       # ordered list of chunk_ids
│       └── manual_meta.json     # model, dims, timestamp
├── parsed/
│   └── {manual_name}/
│       ├── document.json
│       ├── metadata.json
│       ├── pages/
│       ├── tables/
│       └── figures/
└── vectordb/
    ├── chroma/                  # ChromaDB persistent store
    └── bm25/
        ├── corpus_index.pkl     # BM25Okapi fitted model
        └── corpus_map.json      # chunk_id → manual_name, heading mapping
```

---

## Citation format

Answers include inline citations `[1]`, `[2]` and a sources block:

```
To remove the toner cartridge, open the front door [1].
Pull the cartridge tab outward [1]. Insert CF289A until it clicks [2].

Sources:
[1] printer_manual — "Toner cartridge removal" — Pages 32–33
[2] printer_manual — "Table 1-1  Part number" — Pages 21–22
```

---

## Tech stack

| Component | Technology |
|---|---|
| PDF parsing | IBM Docling |
| Chunking | Custom semantic chunker (layout-aware) |
| Embeddings | `gemini-embedding-2` (1536-dim MRL) |
| Dense index | ChromaDB (persistent) |
| Sparse index | BM25Okapi (`rank-bm25`) |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `sarvam-105b` (LLM, 1–10 scoring) |
| Generator | `sarvam-105b` (grounded, citations) |
| HyDE | `sarvam-105b` → hypothetical doc → embed |
