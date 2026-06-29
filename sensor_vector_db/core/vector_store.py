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
            cleaned_metadatas = [_clean_metadata(item) for item in metadatas]
            for batch_ids, batch_documents, batch_embeddings, batch_metadatas in self._iter_aligned_batches(
                chunk_ids,
                documents,
                embeddings,
                cleaned_metadatas,
            ):
                self.collection.upsert(
                    ids=batch_ids,
                    documents=batch_documents,
                    embeddings=batch_embeddings,
                    metadatas=batch_metadatas,
                )
        except Exception as exc:
            first_id = chunk_ids[0] if chunk_ids else "N/A"
            file_path = metadatas[0].get("file_path", "unknown") if metadatas else "unknown"
            raise RuntimeError(
                f"Failed to upsert {len(chunk_ids)} chunks into ChromaDB "
                f"(first_id={first_id}, file={file_path}): {exc}"
            ) from exc

    def update_metadata(
        self,
        chunk_ids: list[str],
        metadatas: list[dict[str, Any]],
    ) -> None:
        """Update chunk metadata in place without recomputing embeddings."""
        if not chunk_ids:
            return
        try:
            cleaned_metadatas = [_clean_metadata(item) for item in metadatas]
            for batch_ids, batch_metadatas in self._iter_aligned_batches(
                chunk_ids,
                cleaned_metadatas,
            ):
                self.collection.update(
                    ids=batch_ids,
                    metadatas=batch_metadatas,
                )
        except Exception as exc:
            first_id = chunk_ids[0] if chunk_ids else "N/A"
            raise RuntimeError(
                f"Failed to update metadata for {len(chunk_ids)} chunks in ChromaDB "
                f"(first_id={first_id}): {exc}"
            ) from exc

    def get_embeddings(self, chunk_ids: list[str]) -> dict[str, list[float]]:
        """Return stored embeddings keyed by chunk ID."""
        if not chunk_ids:
            return {}
        try:
            result = self.collection.get(ids=chunk_ids, include=["embeddings"])
        except Exception as exc:
            raise RuntimeError(f"Failed to read embeddings from ChromaDB: {exc}") from exc

        ids = result.get("ids") or []
        embeddings = result.get("embeddings")
        if embeddings is None:
            embeddings = []
        return {
            chunk_id: _embedding_to_list(embedding)
            for chunk_id, embedding in zip(ids, embeddings, strict=False)
        }

    def get_existing_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return chunk IDs that are already present in ChromaDB."""
        if not chunk_ids:
            return set()
        existing: set[str] = set()
        try:
            for batch in _batches(chunk_ids, 512):
                result = self.collection.get(ids=batch, include=["metadatas"])
                existing.update(str(chunk_id) for chunk_id in result.get("ids") or [])
        except Exception as exc:
            raise RuntimeError(f"Failed to read existing chunks from ChromaDB: {exc}") from exc
        return existing

    def query(
        self,
        query_embedding: list[float],
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run semantic search in ChromaDB."""
        try:
            where = _build_where_clause(filters)
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

    def delete_stale_for_file(
        self,
        file_path: str,
        keep_chunk_ids: set[str],
        document_id: str | None = None,
    ) -> None:
        """Delete vectors for a source file that do not belong to current chunks.

        Tries file_path metadata first, then falls back to document_id if provided
        to ensure orphan vectors are cleaned even when file_path metadata is missing.
        """
        stale_ids: set[str] = set()
        try:
            result = self.collection.get(where={"file_path": file_path}, include=["metadatas"])
            stale_ids.update(
                str(chunk_id)
                for chunk_id in result.get("ids") or []
                if str(chunk_id) not in keep_chunk_ids
            )
        except Exception as exc:
            logger.warning("Failed to query stale vectors by file_path %s: %s", file_path, exc)

        if document_id:
            try:
                result = self.collection.get(
                    where={"document_id": document_id}, include=["metadatas"]
                )
                stale_ids.update(
                    str(chunk_id)
                    for chunk_id in result.get("ids") or []
                    if str(chunk_id) not in keep_chunk_ids
                )
            except Exception as exc:
                logger.warning(
                    "Failed to query stale vectors by document_id %s: %s", document_id, exc
                )

        if stale_ids:
            try:
                self.collection.delete(ids=list(stale_ids))
            except Exception as exc:
                logger.warning("Failed to delete %d stale vectors: %s", len(stale_ids), exc)

    def count(self) -> int:
        """Return number of chunks in the collection."""
        try:
            return int(self.collection.count())
        except Exception:
            return 0

    def max_batch_size(self) -> int:
        """Return the largest mutation batch accepted by the current Chroma client."""
        get_max_batch_size = getattr(self.client, "get_max_batch_size", None)
        if callable(get_max_batch_size):
            try:
                value = int(get_max_batch_size())
                if value > 0:
                    return value
            except Exception:
                logger.warning("Failed to read Chroma max batch size", exc_info=True)
        return 512

    def _iter_aligned_batches(self, *sequences: list[Any]):
        """Yield aligned slices across one or more equally-sized sequences."""
        if not sequences:
            return
        length = len(sequences[0])
        if any(len(sequence) != length for sequence in sequences):
            raise ValueError("All aligned sequences must have the same length.")
        batch_size = self.max_batch_size()
        for start in range(0, length, batch_size):
            end = start + batch_size
            yield tuple(sequence[start:end] for sequence in sequences)

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
            score = _cosine_distance_to_score(distance)
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


def _cosine_distance_to_score(distance: Any) -> float:
    """Convert cosine distance to a [0, 1] similarity score.

    ChromaDB cosine distance is in [0, 2] where 0 = identical, 2 = opposite.
    We convert this to similarity = 1 - (distance / 2), giving [0, 1] range
    that matches the BM25 normalized score distribution.
    """
    dist = float(distance or 0.0)
    return max(0.0, min(1.0, 1.0 - dist / 2.0))


def _build_where_clause(filters: dict[str, Any] | None) -> dict[str, Any] | None:
    """Build a ChromaDB-compatible where clause from scalar metadata filters.

    ChromaDB requires the top-level ``where`` mapping to contain exactly one
    operator, so multiple equality conditions must be combined explicitly with
    ``$and``. A single condition is passed through unchanged, and empty filters
    return ``None``.
    """
    cleaned = _clean_metadata(filters or {})
    if not cleaned:
        return None
    if len(cleaned) == 1:
        return cleaned
    return {"$and": [{key: value} for key, value in cleaned.items()]}


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


def _batches(items: list[str], size: int):
    """Yield fixed-size batches."""
    step = max(1, size)
    for index in range(0, len(items), step):
        yield items[index : index + step]


def _embedding_to_list(embedding: Any) -> list[float]:
    """Convert a stored embedding into a JSON-compatible float list."""
    if hasattr(embedding, "tolist"):
        embedding = embedding.tolist()
    return [float(value) for value in embedding]


def _maybe_int(value: Any) -> int | None:
    """Best-effort integer conversion."""
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None
