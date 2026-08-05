"""
config.py
=========
ParserConfig — all runtime settings for the document parser.

Use Pydantic BaseSettings so values can be overridden via environment variables
or passed directly when constructing DoclingParser.

No paths are hardcoded — everything flows through this config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ParserConfig(BaseModel):
    """
    Configuration for a single parser run.

    All path fields accept str or Path; they are normalised to Path objects
    by the validator.
    """

    # ------------------------------------------------------------------ #
    # I/O                                                                   #
    # ------------------------------------------------------------------ #
    pdf_path: Path = Field(
        ...,
        description="Path to the input PDF file.",
    )
    output_dir: Path = Field(
        ...,
        description="Root directory where parsed output will be written.",
    )

    # ------------------------------------------------------------------ #
    # Extraction toggles                                                    #
    # ------------------------------------------------------------------ #
    extract_tables: bool = Field(
        True,
        description="Whether to extract tables from the PDF.",
    )
    extract_figures: bool = Field(
        True,
        description="Whether to extract figures/images from the PDF.",
    )
    figure_format: Literal["png", "jpg"] = Field(
        "png",
        description="Image format used when saving extracted figures.",
    )

    # ------------------------------------------------------------------ #
    # Docling options                                                        #
    # ------------------------------------------------------------------ #
    do_ocr: bool = Field(
        False,
        description=(
            "Enable OCR for scanned / image-based pages. "
            "Set True for documents flagged 'OCR Needed' in the dataset inventory."
        ),
    )
    do_table_structure: bool = Field(
        True,
        description="Use Docling's table structure model to parse table cells.",
    )

    # ------------------------------------------------------------------ #
    # Logging                                                               #
    # ------------------------------------------------------------------ #
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR"] = Field(
        "INFO",
        description="Python logging level.",
    )
    log_file: Optional[Path] = Field(
        None,
        description="If set, logs are also written to this file.",
    )

    # ------------------------------------------------------------------ #
    # Validators                                                            #
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _validate_paths(self) -> "ParserConfig":
        """Resolve paths and verify the PDF exists."""
        self.pdf_path = self.pdf_path.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()

        if not self.pdf_path.exists():
            raise FileNotFoundError(f"PDF not found: {self.pdf_path}")
        if not self.pdf_path.suffix.lower() == ".pdf":
            raise ValueError(f"Expected a .pdf file, got: {self.pdf_path.suffix}")

        return self

    class Config:
        arbitrary_types_allowed = True
