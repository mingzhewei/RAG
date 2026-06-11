"""Shared typed data structures for the RAG pipeline."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


class OperationCancelled(RuntimeError):
    """Raised when a long-running operation is asked to stop."""


@dataclass
class ParsedSegment:
    """Parsed logical segment from a source document."""

    content: str
    content_type: str
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TextChunk:
    """Chunk ready for embedding and indexing."""

    content: str
    content_type: str
    chunk_index: int
    page_number: int | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class SearchResult:
    """Unified search result from semantic, keyword, or hybrid search."""

    chunk_id: str
    document_id: str
    content: str
    score: float
    source: str
    file_path: str
    file_type: str
    page_number: int | None = None
    sensor_model: str | None = None
    manufacturer: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class ImportErrorItem:
    """One failed import item."""

    path: Path
    error: str


@dataclass
class ImportReport:
    """Summary of one import run."""

    scanned: int = 0
    imported: int = 0
    updated: int = 0
    skipped: int = 0
    failed: int = 0
    errors: list[ImportErrorItem] = field(default_factory=list)
