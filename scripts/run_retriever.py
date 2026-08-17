#!/usr/bin/env python3
"""
scripts/run_retriever.py
─────────────────────────
Interactive CLI for the TechDocs Copilot retrieval pipeline.

Usage:
    python scripts/run_retriever.py --sarvam-key YOUR_KEY
    python scripts/run_retriever.py --sarvam-key YOUR_KEY --no-hyde
    python scripts/run_retriever.py --sarvam-key YOUR_KEY --threshold 0.6

Commands during session:
    /quit or /exit   — exit
    /type            — show query type for next query (debug)
    /threshold 0.6   — change relevance threshold live
    /hyde on|off     — toggle HyDE live
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env FIRST — before any other import reads os.environ
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # env vars set directly in shell are fine too

from src.indexing.chunk_store import ChunkStore
from src.retrieval.config import RetrieverConfig
from src.retrieval.retriever import Retriever
from src.retrieval.query_analyzer import analyze_query


def _setup_logging(level: str = "WARNING") -> None:
    fmt = "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s"
    logging.basicConfig(level=getattr(logging, level.upper()), format=fmt)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="TechDocs Copilot — interactive retrieval CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument(
        "--sarvam-key", "-k",
        default=os.environ.get("SARVAM_API_KEY", ""),
        help="Sarvam API key (or set SARVAM_API_KEY env var)",
    )
    p.add_argument("--chunks-dir",   default="data/chunks",    help="Chunks directory")
    p.add_argument("--vectordb-dir", default="data/vectordb",  help="VectorDB directory")
    p.add_argument("--threshold",    type=float, default=0.5,  help="Relevance threshold (0–1)")
    p.add_argument("--top-k",        type=int,   default=5,    help="Max chunks to generator")
    p.add_argument("--no-hyde",      action="store_true",       help="Disable HyDE expansion")
    p.add_argument("--log-level",    default="WARNING",
                   choices=["DEBUG", "INFO", "WARNING", "ERROR"])
    return p.parse_args()


def _print_banner(cfg: RetrieverConfig) -> None:
    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║         TechDocs Copilot — Retrieval CLI             ║")
    print("╠══════════════════════════════════════════════════════╣")
    print(f"║  Model      : {cfg.sarvam_model:<38}║")
    print(f"║  Threshold  : {cfg.relevance_threshold:<38}║")
    print(f"║  Top-K      : {cfg.generator_top_k:<38}║")
    print(f"║  HyDE       : {'ON' if cfg.use_hyde else 'OFF':<38}║")
    print("╠══════════════════════════════════════════════════════╣")
    print("║  Commands: /quit  /hyde on|off  /threshold 0.6       ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()


def _handle_command(cmd: str, retriever: Retriever, cfg: RetrieverConfig) -> bool:
    """Handle slash commands. Returns True if handled."""
    parts = cmd.strip().split()
    name  = parts[0].lower()

    if name in ("/quit", "/exit", "/q"):
        print("Goodbye.")
        sys.exit(0)

    elif name == "/hyde":
        if len(parts) > 1 and parts[1].lower() in ("on", "off"):
            cfg.use_hyde = parts[1].lower() == "on"
            print(f"HyDE: {'ON' if cfg.use_hyde else 'OFF'}")
        else:
            print(f"HyDE is currently {'ON' if cfg.use_hyde else 'OFF'}. Use: /hyde on|off")
        return True

    elif name == "/threshold":
        if len(parts) > 1:
            try:
                cfg.relevance_threshold = float(parts[1])
                print(f"Threshold set to {cfg.relevance_threshold}")
            except ValueError:
                print("Usage: /threshold 0.6")
        return True

    elif name == "/type":
        return False  # Signal: show type for next query

    return False


def main() -> None:
    args = parse_args()
    _setup_logging(args.log_level)

    # Validate Sarvam key
    sarvam_key = args.sarvam_key
    if not sarvam_key:
        sarvam_key = input("Enter your Sarvam API key: ").strip()
    if not sarvam_key:
        print("Error: Sarvam API key is required.")
        sys.exit(1)

    # Build config
    cfg = RetrieverConfig(
        sarvam_api_key      = sarvam_key,
        chunks_dir          = Path(args.chunks_dir),
        vectordb_dir        = Path(args.vectordb_dir),
        bm25_dir            = Path(args.vectordb_dir) / "bm25",
        relevance_threshold = args.threshold,
        generator_top_k     = args.top_k,
        use_hyde            = not args.no_hyde,
    )

    # Load ChunkStore and Retriever
    print("Loading ChunkStore and indexes…", end=" ", flush=True)
    store = ChunkStore(cfg.chunks_dir)
    store.warm()
    retriever = Retriever(cfg, store)
    print("ready.\n")

    _print_banner(cfg)

    show_type_next = False

    while True:
        try:
            query = input("Query > ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not query:
            continue

        # Handle slash commands
        if query.startswith("/"):
            if query.lower() == "/type":
                show_type_next = True
                print("(will show query type for next query)")
                continue
            _handle_command(query, retriever, cfg)
            continue

        # Show query type if requested
        if show_type_next:
            parsed = analyze_query(query)
            print(f"  Query type : {parsed.query_type.value}")
            print(f"  Manual     : {parsed.manual_name}")
            print(f"  Page       : {parsed.page}")
            print(f"  HyDE?      : {parsed.use_hyde and cfg.use_hyde}")
            show_type_next = False

        # ── Run retrieval ──────────────────────────────────────────────────
        print("\nRetrieving\u2026")
        try:
            result = retriever.retrieve(query)
            _print_result(result)
        except Exception as exc:
            print(f"\n❌ Error: {exc}\n")


def _print_result(result) -> None:
    """Pretty-print a RetrievalResult with ANSWER / SOURCES / RELATED FIGURES."""
    W = 62

    # Separate answer body from the embedded Sources footer
    answer_body = result.answer
    if "\n\nSources:" in answer_body:
        answer_body = answer_body.split("\n\nSources:")[0]

    # ── ANSWER ──────────────────────────────────────────────────────────────
    print()
    print("╔" + "═" * W + "╗")
    print("║  ANSWER" + " " * (W - 7) + "║")
    print("╠" + "═" * W + "╣")
    for raw_line in answer_body.strip().splitlines():
        line = raw_line
        while len(line) > W - 2:
            print("║ " + line[:W - 2] + " ║")
            line = "  " + line[W - 2:]
        print("║ " + line.ljust(W - 2) + " ║")
    print("╚" + "═" * W + "╝")

    # ── SOURCES ─────────────────────────────────────────────────────────────
    if result.sources:
        print()
        print("╔" + "═" * W + "╗")
        print("║  SOURCES" + " " * (W - 8) + "║")
        print("╠" + "═" * W + "╣")
        for s in result.sources:
            pages = s.get("pages") or []
            pages_str = f"{pages[0]}–{pages[-1]}" if len(pages) >= 2 else (str(pages[0]) if pages else "—")
            line = f"[{s['citation_num']}] {s['manual_name']}  —  \"{s['heading']}\"  —  Pages {pages_str}"
            print("║ " + line[:W - 2].ljust(W - 2) + " ║")
        print("╚" + "═" * W + "╝")

    # ── RELATED FIGURES ─────────────────────────────────────────────────────
    if result.figures:
        print()
        label = f"  RELATED FIGURES ({len(result.figures)})"
        print("╔" + "═" * W + "╗")
        print("║" + label + " " * (W - len(label)) + "║")
        print("╠" + "═" * W + "╣")
        for fig in result.figures:
            fig_id   = getattr(fig, "fig_id",     str(fig))
            img_path = getattr(fig, "image_path", str(fig))
            caption  = getattr(fig, "caption",    None)
            page     = getattr(fig, "page",       None)

            id_line = f"  🖼️   {fig_id}" + (f"  (page {page})" if page else "")
            print("║ " + id_line[:W - 2].ljust(W - 2) + " ║")
            print("║ " + f"     └─ {img_path}"[:W - 2].ljust(W - 2) + " ║")
            if caption:
                print("║ " + f"     Caption: {caption}"[:W - 2].ljust(W - 2) + " ║")
        print("╚" + "═" * W + "╝")

    print(f"\n  [{result.query_type} path | {result.chunks_used} chunk(s) used]\n")


if __name__ == "__main__":
    main()
