"""
utils.py
========
Shared utility functions used across the parsing package.

Includes:
    - Logging setup (console + optional file handler)
    - JSON serialisation helpers
    - Path management helpers
    - Document-ID generation
"""

from __future__ import annotations

import hashlib
import json
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from pydantic import BaseModel


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------


def setup_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[Path] = None,
) -> logging.Logger:
    """
    Configure and return a named logger.

    Parameters
    ----------
    name : str
        Logger name (usually __name__ of the calling module).
    level : str
        Logging level string: DEBUG | INFO | WARNING | ERROR.
    log_file : Path, optional
        If provided, a FileHandler is added alongside the StreamHandler.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    # Avoid adding duplicate handlers if the logger is re-used.
    if logger.handlers:
        return logger

    logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    fmt = logging.Formatter(
        fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Console handler
    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    # Optional file handler
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file, encoding="utf-8")
        fh.setFormatter(fmt)
        logger.addHandler(fh)

    return logger


# ---------------------------------------------------------------------------
# JSON helpers
# ---------------------------------------------------------------------------


def write_json(data: Any, path: Path, indent: int = 2) -> None:
    """
    Serialise *data* to JSON and write it to *path*.

    Handles Pydantic BaseModel instances and plain dicts/lists.

    Parameters
    ----------
    data : Any
        Data to serialise.
    path : Path
        Destination file path (parent directories are created automatically).
    indent : int
        JSON indentation level.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(data, BaseModel):
        payload = data.model_dump(mode="json")
    elif isinstance(data, list) and data and isinstance(data[0], BaseModel):
        payload = [item.model_dump(mode="json") for item in data]
    else:
        payload = data

    with path.open("w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=indent, ensure_ascii=False)


def load_json(path: Path) -> Any:
    """Load and return a JSON file as a Python object."""
    with Path(path).open("r", encoding="utf-8") as fh:
        return json.load(fh)


# ---------------------------------------------------------------------------
# ID / timestamp helpers
# ---------------------------------------------------------------------------


def make_document_id(pdf_path: Path) -> str:
    """
    Generate a stable, short document ID from the PDF filename.

    Uses the first 8 chars of the SHA-256 hash of the filename stem,
    prefixed with 'doc_'.

    Parameters
    ----------
    pdf_path : Path
        Path to the PDF.

    Returns
    -------
    str
        e.g. 'doc_a3f1c2b4'
    """
    digest = hashlib.sha256(pdf_path.stem.encode()).hexdigest()[:8]
    return f"doc_{digest}"


def utc_now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(tz=timezone.utc).isoformat()


# ---------------------------------------------------------------------------
# Counter helpers for sequential IDs
# ---------------------------------------------------------------------------


def make_text_id(n: int) -> str:
    """e.g. 1 → 'txt_0001'"""
    return f"txt_{n:04d}"


def make_table_id(n: int) -> str:
    """e.g. 1 → 'tbl_0001'"""
    return f"tbl_{n:04d}"


def make_figure_id(n: int) -> str:
    """e.g. 1 → 'fig_0001'"""
    return f"fig_{n:04d}"


def make_figure_filename(n: int, fmt: str = "png") -> str:
    """e.g. 1 → 'figure_0001.png'"""
    return f"figure_{n:04d}.{fmt}"


# ---------------------------------------------------------------------------
# Output-directory helpers
# ---------------------------------------------------------------------------


def ensure_output_dirs(output_dir: Path) -> dict[str, Path]:
    """
    Create the canonical subdirectory tree under *output_dir* and return
    a dict mapping shorthand names to their absolute Paths.

    Structure created:
        output_dir/
            pages/
            tables/
            figures/

    Returns
    -------
    dict with keys: 'root', 'pages', 'tables', 'figures'
    """
    dirs: dict[str, Path] = {
        "root": output_dir,
        "pages": output_dir / "pages",
        "tables": output_dir / "tables",
        "figures": output_dir / "figures",
    }
    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)
    return dirs
