"""
TechDocs Copilot — Chunking Package (Stage 3)
"""
from .config import ChunkerConfig
from .models import (
    Chunk,
    ChunkManifest,
    ChunkPlan,
    NormalizedBlock,
    PlannedChunk,
    ValidationFlag,
)
from .normalizer import Normalizer
from .chunk_planner import ChunkPlanner
from .chunk_validator import ChunkValidator
from .chunk_builder import ChunkBuilder
from .chunker import SemanticChunker

__all__ = [
    "ChunkerConfig",
    "SemanticChunker",
    "Normalizer",
    "ChunkPlanner",
    "ChunkValidator",
    "ChunkBuilder",
    "NormalizedBlock",
    "PlannedChunk",
    "ChunkPlan",
    "ValidationFlag",
    "Chunk",
    "ChunkManifest",
]
