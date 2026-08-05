"""
TechDocs Copilot — Parsing Package
"""
from .parser_interface import Parser, TextBlock, TableBlock, FigureBlock, PageOutput, ParsedDocument
from .docling_parser import DoclingParser
from .marker_parser import MarkerParser
from .config import ParserConfig

__all__ = [
    "Parser",
    "DoclingParser",
    "MarkerParser",
    "ParserConfig",
    "TextBlock",
    "TableBlock",
    "FigureBlock",
    "PageOutput",
    "ParsedDocument",
]
