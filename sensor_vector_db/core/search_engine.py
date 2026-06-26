"""Semantic, keyword, and hybrid search."""

from __future__ import annotations

import json
import re
from typing import Any

from sqlalchemy import select

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.embedding import BaseEmbedding, create_embedding_provider
from sensor_vector_db.core.types import SearchResult
from sensor_vector_db.core.vector_store import VectorStore
from sensor_vector_db.models.database import Document, DocumentChunk, session_scope
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)

BM25_MAX_CHUNKS = 50000


class SearchEngine:
    """Search engine combining Chroma semantic search with BM25 keyword search."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize search engine."""
        self.settings = settings or get_settings()
        self.embedding = embedding or create_embedding_provider(self.settings)
        self.vector_store = vector_store or VectorStore(self.settings)

    def search(
        self,
        query: str,
        mode: str = "hybrid",
        top_k: int | None = None,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run semantic, keyword, or hybrid search."""
        clean_query = query.strip()
        if not clean_query:
            return []
        limit = top_k or self.settings.search_top_k
        self._warn_if_import_active()
        mode = mode.lower()
        if mode == "semantic":
            return self.semantic_search(clean_query, limit, filters)
        if mode == "keyword":
            return self.keyword_search(clean_query, limit, filters)
        return self.hybrid_search(clean_query, limit, filters)

    def _warn_if_import_active(self) -> None:
        """Log a one-shot warning when a search runs during an active import.

        Import workers compete for the shared embedding model, so searches
        may be slower than usual while files are being vectorised.
        """
        if not hasattr(self, "_import_warned"):
            try:
                from sensor_vector_db.models.database import ImportJob as _ImportJob
                from sqlalchemy import select as _select

                with session_scope(self.settings) as session:
                    active = session.execute(
                        _select(_ImportJob.id).where(
                            _ImportJob.status.in_(("running", "stopping"))
                        )
                    ).first()
                if active:
                    logger.info(
                        "检索请求在导入进行中发起——模型正忙，响应可能变慢"
                    )
            except Exception:
                pass
            self._import_warned = True  # type: ignore[attr-defined]

    def semantic_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run vector similarity search.

        Vectors from documents that are still being imported
        (status != 'imported') are excluded so that partial or
        uncommitted chunks never leak into search results.
        """
        embedding = self.embedding.embed_query(query)
        vector_filters = _metadata_filters(filters)
        # Fetch extra results so the post-filter still yields top_k.
        results = self.vector_store.query(embedding, top_k * 3, vector_filters)
        return _filter_imported_only(self.settings, results)[:top_k]

    def keyword_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Run BM25 keyword search over stored chunks."""
        try:
            from rank_bm25 import BM25Okapi
        except ImportError as exc:
            raise RuntimeError("rank-bm25 is required for keyword search.") from exc

        rows = self._load_candidate_chunks(filters, limit=BM25_MAX_CHUNKS)
        if not rows:
            return []
        tokenized = [tokenize_for_search(row["content"]) for row in rows]
        query_tokens = tokenize_for_search(query)
        if not query_tokens:
            return []
        bm25 = BM25Okapi(tokenized)
        scores = bm25.get_scores(query_tokens)
        max_score = max(float(score) for score in scores) if len(scores) else 0.0
        ranked = sorted(
            zip(rows, scores, strict=False),
            key=lambda item: float(item[1]),
            reverse=True,
        )[:top_k]
        results: list[SearchResult] = []
        for row, score in ranked:
            normalized = float(score) / max_score if max_score > 0 else 0.0
            if normalized <= 0:
                continue
            results.append(_row_to_search_result(row, normalized))
        return results

    def hybrid_search(
        self,
        query: str,
        top_k: int,
        filters: dict[str, Any] | None = None,
    ) -> list[SearchResult]:
        """Fuse semantic and keyword scores."""
        semantic = self.semantic_search(query, top_k * 3, filters)
        keyword = self.keyword_search(query, top_k * 3, filters)
        combined: dict[str, SearchResult] = {}
        semantic_weight = self.settings.semantic_weight
        bm25_weight = self.settings.bm25_weight
        total_weight = semantic_weight + bm25_weight or 1.0

        for result in semantic:
            item = _copy_result(result)
            item.score = (result.score * semantic_weight) / total_weight
            combined[item.chunk_id] = item

        for result in keyword:
            if result.chunk_id in combined:
                combined[result.chunk_id].score += (result.score * bm25_weight) / total_weight
            else:
                item = _copy_result(result)
                item.score = (result.score * bm25_weight) / total_weight
                combined[item.chunk_id] = item

        return sorted(combined.values(), key=lambda item: item.score, reverse=True)[:top_k]

    def _load_candidate_chunks(
        self,
        filters: dict[str, Any] | None = None,
        limit: int | None = None,
    ) -> list[dict[str, Any]]:
        """Load chunks from SQLite for BM25 indexing.

        Args:
            filters: Metadata filters to apply.
            limit: Maximum chunks to load (guards memory usage). Logs warning if truncated.
        """
        clauses = []
        clauses.append(Document.status == "imported")
        if filters:
            if filters.get("file_type"):
                clauses.append(Document.file_type == filters["file_type"])
            if filters.get("manufacturer"):
                clauses.append(Document.manufacturer == filters["manufacturer"])
            if filters.get("sensor_model"):
                clauses.append(Document.sensor_model == filters["sensor_model"])
        with session_scope(self.settings) as session:
            statement = (
                select(DocumentChunk, Document)
                .join(Document, Document.id == DocumentChunk.document_id)
                .where(*clauses)
            )
            if limit:
                statement = statement.limit(limit + 1)
            all_results = session.execute(statement).all()
            truncated = limit and len(all_results) > limit
            if truncated:
                logger.warning(
                    "BM25 keyword search truncated to %d chunks (total matched: >%d). "
                    "Consider adding filters to narrow the search scope.",
                    limit,
                    limit,
                )
                all_results = all_results[:limit]
            rows = []
            for chunk, document in all_results:
                metadata = json.loads(chunk.metadata_json or "{}")
                rows.append(
                    {
                        "chunk_id": chunk.id,
                        "document_id": document.id,
                        "content": chunk.content,
                        "source": chunk.source_label,
                        "file_path": document.file_path,
                        "file_type": document.file_type,
                        "page_number": chunk.page_number,
                        "sensor_model": document.sensor_model,
                        "manufacturer": document.manufacturer,
                        "metadata": metadata,
                    }
                )
            return rows


def tokenize_for_search(text: str) -> list[str]:
    """Tokenize mixed Chinese/English text for BM25."""
    try:
        import jieba

        tokens = list(jieba.cut(text.lower()))
    except ImportError:
        tokens = re.findall(r"[\u4e00-\u9fff]|[a-z0-9_./%+\-]+", text.lower())
    return [token.strip() for token in tokens if token.strip() and not token.isspace()]


def _row_to_search_result(row: dict[str, Any], score: float) -> SearchResult:
    """Convert a SQLite row dictionary to SearchResult."""
    return SearchResult(
        chunk_id=row["chunk_id"],
        document_id=row["document_id"],
        content=row["content"],
        score=score,
        source=row["source"],
        file_path=row["file_path"],
        file_type=row["file_type"],
        page_number=row["page_number"],
        sensor_model=row["sensor_model"],
        manufacturer=row["manufacturer"],
        metadata=row["metadata"],
    )


def _copy_result(result: SearchResult) -> SearchResult:
    """Copy a search result before mutating score."""
    return SearchResult(
        chunk_id=result.chunk_id,
        document_id=result.document_id,
        content=result.content,
        score=result.score,
        source=result.source,
        file_path=result.file_path,
        file_type=result.file_type,
        page_number=result.page_number,
        sensor_model=result.sensor_model,
        manufacturer=result.manufacturer,
        metadata=dict(result.metadata),
    )


def _metadata_filters(filters: dict[str, Any] | None) -> dict[str, Any]:
    """Convert UI filters to Chroma scalar metadata filters."""
    if not filters:
        return {}
    return {
        key: value
        for key, value in filters.items()
        if key in {"file_type", "manufacturer", "sensor_model"} and value
    }


def _filter_imported_only(
    settings: Settings,
    results: list[SearchResult],
) -> list[SearchResult]:
    """Keep only search results whose document has status='imported'.

    During import, vectors are written to ChromaDB *before* the document
    status is flipped from 'indexing' to 'imported'.  This post-filter
    ensures that chunks from partially-committed documents never appear
    in search results.

    If *every* result document is imported, the list is returned unchanged
    with zero overhead (no SQLite query).
    """
    if not results:
        return results

    # Quick-path: all document IDs that appear in the result set.
    candidate_ids = {r.document_id for r in results}

    with session_scope(settings) as session:
        imported = {
            row[0]
            for row in session.execute(
                select(Document.id).where(
                    Document.id.in_(candidate_ids),
                    Document.status == "imported",
                )
            ).all()
        }

    # If all candidates are imported, return immediately.
    if imported == candidate_ids:
        return results

    filtered = [r for r in results if r.document_id in imported]
    if len(filtered) < len(results):
        logger.debug(
            "语义搜索结果已过滤：%d/%d 条来自未完成导入的文档",
            len(results) - len(filtered),
            len(results),
        )
    return filtered
