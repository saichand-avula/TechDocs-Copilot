"""
config.py
=========
ChunkerConfig — all runtime settings for the semantic chunking stage.

Mirrors the style of src/parsing/config.py.
No paths are hardcoded — everything flows through this config.
"""

from __future__ import annotations

from pathlib import Path
from typing import Literal, Optional

from pydantic import BaseModel, Field, model_validator


class ChunkerConfig(BaseModel):
    """
    Configuration for a single chunker run.

    All path fields accept str or Path; they are normalised to Path objects
    by the validator.
    """

    # ------------------------------------------------------------------ #
    # I/O                                                                  #
    # ------------------------------------------------------------------ #
    parsed_dir: Path = Field(
        ...,
        description="Path to a parser_v1/<doc> directory (must contain document.json).",
    )
    output_dir: Path = Field(
        ...,
        description="Root directory where chunk output will be written.",
    )

    # ------------------------------------------------------------------ #
    # Planner thresholds                                                   #
    # ------------------------------------------------------------------ #
    max_tokens: int = Field(
        512,
        ge=50,
        description=(
            "Overflow threshold (R4.1). Chunks projected to exceed this word count "
            "are candidates for splitting."
        ),
    )
    short_sibling_threshold: int = Field(
        150,
        ge=10,
        description=(
            "Token count below which adjacent H2 sibling sections are merged (R3.3). "
            "Both siblings must be under this threshold."
        ),
    )

    # ------------------------------------------------------------------ #
    # Validator thresholds                                                  #
    # ------------------------------------------------------------------ #
    min_chunk_tokens: int = Field(
        20,
        ge=1,
        description="Validator: flag chunks with fewer tokens than this as 'too_small'.",
    )
    max_chunk_tokens: int = Field(
        800,
        ge=100,
        description="Validator: flag chunks with more tokens than this as 'too_large'.",
    )
    score_threshold: float = Field(
        70.0,
        ge=0.0,
        le=100.0,
        description=(
            "Chunks with chunk_score below this value are flagged for LLM review. "
            "Default 70.0 means any chunk with 3+ warnings or 1 error is escalated."
        ),
    )

    # ------------------------------------------------------------------ #
    # Logging                                                              #
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
    # Validators                                                           #
    # ------------------------------------------------------------------ #
    @model_validator(mode="after")
    def _validate_paths(self) -> "ChunkerConfig":
        """Resolve paths and verify parsed_dir contains a document.json."""
        self.parsed_dir = self.parsed_dir.expanduser().resolve()
        self.output_dir = self.output_dir.expanduser().resolve()

        doc_json = self.parsed_dir / "document.json"
        if not self.parsed_dir.is_dir():
            raise FileNotFoundError(f"parsed_dir not found: {self.parsed_dir}")
        if not doc_json.exists():
            raise FileNotFoundError(
                f"document.json not found in parsed_dir: {self.parsed_dir}. "
                "Make sure this is a valid parser_v1/<doc> directory."
            )
        return self

    class Config:
        arbitrary_types_allowed = True
