"""ChromaDB-backed vector storage."""

from __future__ import annotations

from typing import Any

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.types import SearchResult
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)


class VectorStore:
    """Persistent local vector store built on ChromaDB."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize Chroma client and collection."""
        self.settings = settings or get_settings()
        self.settings.ensure_directories()
        try:
            import chromadb
        except ImportError as exc:
            raise RuntimeError("chromadb is required for vector storage.") from exc

        self.client = chromadb.PersistentClient(path=str(self.settings.chroma_path))
        self.collection = self.client.get_or_create_collection(
            name=self.settings.chroma_collection,
            metadata={"hnsw:space": "cosine"},
        )

    def add_chunks(
        self,
        chunk_ids: list[str],
        documents: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Add or update embedded chunks in ChromaDB."""
        if not chunk_ids:
            return
        try:
            self.collection.upsert(
                ids=chunk_ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=[_clean_metadata(item) for item in metadatas],
            )
        except Exception as exc:
            raise RuntimeError(f"Failed to upsert chunks into ChromaDB: {exc}") from exc

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run semantic search in ChromaDB."""
        try:
            where = _clean_metadata(filters or {}) or None
            result = self.collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
                where=where,
                include=["documents", "metadatas", "distances"],
            )
        except Exception as exc:
            raise RuntimeError(f"ChromaDB semantic query failed: {exc}") from exc
        return self._to_results(result)

    def delete_by_document_id(self, document_id: str) -> None:
        """Delete all vectors belonging to one document."""
        try:
            self.collection.delete(where={"document_id": document_id})
        except Exception as exc:
            logger.warning("Failed to delete vectors for document %s: %s", document_id, exc)

    def count(self) -> int:
        """Return number of chunks in the collection."""
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def _to_results(self, result: dict[str, Any]) -> list[SearchResult]:
        """Convert raw ChromaDB response into unified search results."""
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]
        search_results: list[SearchResult] = []
        for chunk_id, content, metadata, distance in zip(
            ids,
            documents,
            metadatas,
            distances,
            strict=False,
        ):
            score = 1.0 / (1.0 + float(distance or 0.0))
            search_results.append(
                SearchResult(
                    chunk_id=chunk_id,
                    document_id=str(metadata.get("document_id", "")),
                    content=content or "",
                    score=score,
                    source=str(metadata.get("source_label", "")),
                    file_path=str(metadata.get("file_path", "")),
                    file_type=str(metadata.get("file_type", "")),
                    page_number=_maybe_int(metadata.get("page_number")),
                    sensor_model=metadata.get("sensor_model"),
                    manufacturer=metadata.get("manufacturer"),
                    metadata=dict(metadata),
                )
            )
        return search_results


def _clean_metadata(metadata: dict[str, Any]) -> dict[str, str | int | float | bool]:
    """Keep only Chroma-compatible scalar metadata values."""
    cleaned: dict[str, str | int | float | bool] = {}
    for key, value in metadata.items():
        if value is None:
            continue
        if isinstance(value, (str, int, float, bool)):
            cleaned[key] = value
        else:
            cleaned[key] = str(value)
    return cleaned


def _maybe_int(value: Any) -> int | None:
    """Best-effort integer conversion."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None

