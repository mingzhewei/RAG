"""Document import, de-duplication, indexing, and management."""

from __future__ import annotations

import json
from pathlib import Path
import threading
from typing import Any, Callable

from sqlalchemy import func as sa_func, select

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.document_processor import (
    DocumentChunker,
    DocumentParserFactory,
    MetadataExtractor,
    PDFParser,
    metadata_to_json,
)
from sensor_vector_db.core.embedding import BaseEmbedding, create_embedding_provider
from sensor_vector_db.core.index_profile import build_index_profile, profile_satisfies
from sensor_vector_db.core.types import ImportErrorItem, ImportReport, OperationCancelled, TextChunk
from sensor_vector_db.core.vector_store import VectorStore
from sensor_vector_db.models.database import Document, DocumentChunk, session_scope, utc_now
from sensor_vector_db.utils.file_utils import (
    get_file_exclusion_reason,
    get_file_info,
    iter_supported_files,
)
from sensor_vector_db.utils.hash_utils import calculate_file_md5
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)
ProgressCallback = Callable[..., None]

# Module-level lock so that only one DocumentManager instance performs
# startup recovery at a time.  Without this, concurrent __init__ calls
# (UI thread + import worker thread) would both try to recover the same
# orphan documents simultaneously.
_RECOVERY_LOCK = threading.Lock()


class DocumentManager:
    """Manage local document import and indexed metadata."""

    def __init__(
        self,
        settings: Settings | None = None,
        embedding: BaseEmbedding | None = None,
        vector_store: VectorStore | None = None,
    ) -> None:
        """Initialize manager and indexing dependencies."""
        self.settings = settings or get_settings()
        self.parser_factory = DocumentParserFactory(self.settings)
        self.chunker = DocumentChunker(self.settings)
        self.metadata_extractor = MetadataExtractor()
        self.embedding = embedding or create_embedding_provider(self.settings)
        self.vector_store = vector_store or VectorStore(self.settings)
        # Recover (NOT blindly delete) indexing documents left by force-kill.
        # Documents with valid chunk checkpoints are preserved so the next
        # import job can resume them instead of re-parsing from scratch.
        self.recover_indexing_documents()

    def recover_indexing_documents(self) -> dict:
        """Recover documents stuck in ``status='indexing'`` after a force-kill.

        Recovery strategy (most to least ideal):

        1. **Chunk checkpoint valid → keep it.**
           The document has chunk rows in SQLite; the next import job can
           skip parsing and only redo the missing vectors via
           ``_resume_indexing_document()``.

        2. **Chunk checkpoint missing or empty → delete it.**
           Without persisted chunks, there is nothing to resume.  The
           document is removed from both SQLite and ChromaDB.  It will be
           re-processed from the source file next time.

        3. **ChromaDB vector cleanup.**
           For kept documents, orphan Chroma vectors (written before
           ``os._exit``) that belong to no current chunk row are deleted.

        Returns:
            Dict with keys ``kept``, ``deleted``, and ``vectors_cleaned``.
        """
        from sensor_vector_db.utils.logger import get_logger as _get_log

        report = {"kept": 0, "deleted": 0, "vectors_cleaned": 0}

        # Acquire the module-level lock so concurrent DocumentManager
        # instances (UI thread + import workers) don't race on recovery.
        if not _RECOVERY_LOCK.acquire(blocking=False):
            _get_log(__name__).debug("恢复逻辑正由其他实例执行，跳过")
            return report

        try:
            # Batch-load orphans AND their chunk counts in two queries
            # instead of one session per document.
            with session_scope(self.settings) as session:
                orphans = session.execute(
                    select(Document).where(Document.status == "indexing")
                ).scalars().all()
                if not orphans:
                    return report

                orphan_ids = [d.id for d in orphans]
                # Build a dict {document_id: chunk_count} in a single query
                chunk_count_rows = session.execute(
                    select(
                        DocumentChunk.document_id,
                        sa_func.count(),
                    )
                    .where(DocumentChunk.document_id.in_(orphan_ids))
                    .group_by(DocumentChunk.document_id)
                ).all()
                chunk_counts = {row[0]: row[1] for row in chunk_count_rows}

            _get_log(__name__).warning(
                "发现 %d 个未完成的索引文档（上次可能被强制退出），正在分析恢复策略…",
                len(orphans),
            )

            for doc in orphans:
                doc_id = doc.id
                chunk_count = chunk_counts.get(doc_id, 0)

                if chunk_count and chunk_count > 0:
                    # Strategy 1: checkpoint is valid — keep it for resume
                    _get_log(__name__).info(
                        "保留可恢复索引文档 id=%s path=%s chunks=%d（下次导入将从断点恢复）",
                        doc_id,
                        doc.file_path,
                        chunk_count,
                    )
                    # Clean partial ChromaDB vectors for this document that have
                    # no matching chunk row, keeping only valid checkpoints.
                    try:
                        cleaned = self._prune_orphan_chroma_vectors(doc_id)
                        report["vectors_cleaned"] += cleaned
                    except Exception as exc:
                        _get_log(__name__).warning(
                            "清理 Chroma 孤立向量失败 doc=%s: %s", doc_id, exc
                        )
                    report["kept"] += 1
                else:
                    # Strategy 2: no chunks — truly orphaned, safe to delete
                    _get_log(__name__).info(
                        "清理无效索引文档 id=%s path=%s（无可用分块断点，将重新处理）",
                        doc_id,
                        doc.file_path,
                    )
                    try:
                        self.vector_store.delete_by_document_id(doc_id)
                    except Exception as exc:
                        _get_log(__name__).warning(
                            "清理 Chroma 向量失败 doc=%s: %s", doc_id, exc
                        )
                    with session_scope(self.settings) as session:
                        doomed = session.get(Document, doc_id)
                        if doomed:
                            session.delete(doomed)
                    report["deleted"] += 1

            if report["kept"] or report["deleted"]:
                _get_log(__name__).info(
                    "索引文档恢复完成：保留 %d 个（可断点续传），删除 %d 个（无断点），清理孤立向量 %d 个",
                    report["kept"],
                    report["deleted"],
                    report["vectors_cleaned"],
                )
            return report
        finally:
            _RECOVERY_LOCK.release()

    def _prune_orphan_chroma_vectors(self, document_id: str) -> int:
        """Delete ChromaDB vectors for a document that lack a matching
        SQLite chunk row, leaving only valid checkpoints intact.

        Returns the number of vectors removed.
        """
        with session_scope(self.settings) as session:
            valid_chunk_ids = {
                row[0]
                for row in session.execute(
                    select(DocumentChunk.id).where(
                        DocumentChunk.document_id == document_id
                    )
                ).all()
            }

        if not valid_chunk_ids:
            return 0

        try:
            result = self.vector_store.collection.get(
                where={"document_id": document_id},
                include=["metadatas"],
            )
            chroma_ids = set(result.get("ids") or [])
        except Exception:
            return 0

        orphan_ids = chroma_ids - valid_chunk_ids
        if orphan_ids:
            try:
                self.vector_store.collection.delete(ids=list(orphan_ids))
                return len(orphan_ids)
            except Exception:
                pass
        return 0

    def import_path(
        self,
        path: str | Path,
        progress_callback: ProgressCallback | None = None,
    ) -> ImportReport:
        """Import one file or all supported files under a directory."""
        self._emit_progress(progress_callback, 0, 0, str(path), "扫描", "正在扫描支持的文件")
        files = iter_supported_files(path)
        report = ImportReport(scanned=len(files))
        self._emit_progress(
            progress_callback,
            0,
            len(files),
            str(path),
            "扫描完成",
            f"发现 {len(files)} 个支持文件",
        )
        for index, file_path in enumerate(files, start=1):
            try:
                self._emit_progress(
                    progress_callback,
                    index,
                    len(files),
                    str(file_path),
                    "准备处理",
                    f"准备导入 {file_path.name}",
                )
                status = self.import_file(
                    file_path,
                    progress_callback=progress_callback,
                    current_index=index,
                    total_files=len(files),
                )
                if status == "imported":
                    report.imported += 1
                elif status == "updated":
                    report.updated += 1
                else:
                    report.skipped += 1
            except Exception as exc:
                logger.exception("Failed to import %s", file_path)
                report.failed += 1
                report.errors.append(ImportErrorItem(file_path, str(exc)))
                self._emit_progress(
                    progress_callback,
                    index,
                    len(files),
                    str(file_path),
                    "失败",
                    str(exc),
                    level="error",
                )
        return report

    def import_file(
        self,
        path: str | Path,
        progress_callback: ProgressCallback | None = None,
        cancel_callback: Callable[[], None] | None = None,
        current_index: int = 1,
        total_files: int = 1,
    ) -> str:
        """Import or update one supported file."""
        self._check_cancelled(cancel_callback)
        file_path = Path(path).resolve()
        exclusion_reason = get_file_exclusion_reason(file_path)
        if exclusion_reason:
            raise ValueError(f"File is excluded from import: {exclusion_reason}")
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "计算哈希",
            "正在计算文件哈希用于去重",
        )
        file_hash = calculate_file_md5(file_path)
        file_info = get_file_info(file_path)
        target_profile = build_index_profile(self.settings, file_info.file_type)
        existing_document_id: str | None = None
        existing_snapshot: dict[str, Any] | None = None
        reuse_source_snapshot: dict[str, Any] | None = None
        resume_document_id: str | None = None
        with session_scope(self.settings) as session:
            existing = session.execute(
                select(Document).where(Document.file_path == str(file_path))
            ).scalar_one_or_none()
            if (
                existing
                and existing.file_hash == file_hash
                and existing.status == "imported"
                and profile_satisfies(existing.index_profile, target_profile)
            ):
                self._emit_progress(
                    progress_callback,
                    current_index,
                    total_files,
                    str(file_path),
                    "跳过",
                    "文件未变化，且当前索引配置已满足目标，跳过入库",
                )
                return "skipped"
            if existing:
                existing_document_id = existing.id
                if (
                    existing.file_hash == file_hash
                    and existing.status == "indexing"
                    and profile_satisfies(existing.index_profile, target_profile)
                ):
                    resume_document_id = existing.id
                    self._emit_progress(
                        progress_callback,
                        current_index,
                        total_files,
                        str(file_path),
                        "恢复未完成索引",
                        "发现同一文件上次已完成分块，准备补齐缺失向量",
                    )
                elif existing.status == "imported":
                    existing_snapshot = self._snapshot_document(session, existing)
                    self._emit_progress(
                        progress_callback,
                        current_index,
                        total_files,
                        str(file_path),
                        "更新",
                        "检测到文件变化，准备生成新向量并替换旧索引",
                    )
                else:
                    self._emit_progress(
                        progress_callback,
                        current_index,
                        total_files,
                        str(file_path),
                        "重建未完成索引",
                        "上次索引没有可复用分块，准备重新解析并重建向量",
                    )
                status = "updated"
            else:
                status = "imported"
            reusable = self._find_reusable_document(session, file_hash, target_profile, existing_document_id)
            if reusable:
                reuse_source_snapshot = self._snapshot_document(session, reusable)

        if resume_document_id:
            resumed_status = self._resume_indexing_document(
                document_id=resume_document_id,
                file_path=file_path,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                current_index=current_index,
                total_files=total_files,
            )
            if resumed_status:
                return resumed_status

        if reuse_source_snapshot:
            reused_status = self._try_reuse_indexed_document(
                file_path=file_path,
                file_hash=file_hash,
                file_info=file_info,
                source_snapshot=reuse_source_snapshot,
                target_profile=target_profile,
                existing_document_id=existing_document_id,
                existing_snapshot=existing_snapshot,
                progress_callback=progress_callback,
                current_index=current_index,
                total_files=total_files,
            )
            if reused_status:
                return reused_status

        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "解析文档",
            "正在解析正文、表格或 OCR 文本",
        )

        # For PDF files, use page-batched parsing so each page batch is written
        # to a SQLite checkpoint immediately, keeping peak memory bounded.
        if file_info.file_type == "pdf" and self.settings.pdf_page_batch_size > 0:
            return self._import_pdf_batched(
                file_path=file_path,
                file_hash=file_hash,
                file_info=file_info,
                target_profile=target_profile,
                existing_document_id=existing_document_id,
                existing_snapshot=existing_snapshot,
                status=status,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                current_index=current_index,
                total_files=total_files,
            )

        segments = self.parser_factory.parse(file_path, cancel_callback=cancel_callback)
        self._check_cancelled(cancel_callback)
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "文档分块",
            f"解析得到 {len(segments)} 个片段，正在分块",
        )
        chunks = self.chunker.chunk(segments)
        if not chunks:
            raise RuntimeError("No indexable text was extracted from the file.")
        metadata = self.metadata_extractor.extract(file_path, segments)

        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "写入元数据",
            f"正在写入 SQLite checkpoint，chunk 数：{len(chunks)}",
        )
        document_id, chunk_ids, documents, metadatas = self._create_indexing_checkpoint(
            file_path=file_path,
            file_hash=file_hash,
            file_info=file_info,
            metadata=metadata,
            target_profile=target_profile,
            chunks=chunks,
            existing_document_id=existing_document_id,
        )
        try:
            self._write_missing_vectors(
                chunk_ids=chunk_ids,
                documents=documents,
                metadatas=metadatas,
                file_path=file_path,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                current_index=current_index,
                total_files=total_files,
            )
        except OperationCancelled:
            raise
        except Exception:
            self.delete_document(document_id)
            if existing_snapshot:
                try:
                    self._restore_document_snapshot(existing_snapshot)
                except Exception:
                    logger.exception("Failed to restore previous document after vector write failure")
            raise
        if existing_document_id:
            self.vector_store.delete_by_document_id(existing_document_id)
        self._delete_stale_vectors(str(file_path), set(chunk_ids), document_id)
        self._mark_document_imported(document_id)
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "完成文件",
            f"{file_path.name} 已{status}",
        )
        return status

    def _resume_indexing_document(
        self,
        document_id: str,
        file_path: Path,
        progress_callback: ProgressCallback | None,
        cancel_callback: Callable[[], None] | None,
        current_index: int,
        total_files: int,
    ) -> str | None:
        """Finish an interrupted file import from persisted chunk rows."""
        checkpoint = self._load_indexing_checkpoint(document_id, file_path)
        if not checkpoint:
            return None

        chunk_ids = checkpoint["chunk_ids"]
        documents = checkpoint["documents"]
        metadatas = checkpoint["metadatas"]
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "恢复文件断点",
            f"复用已解析的 {len(chunk_ids)} 个 chunk，补齐缺失向量",
        )
        self._write_missing_vectors(
            chunk_ids=chunk_ids,
            documents=documents,
            metadatas=metadatas,
            file_path=file_path,
            progress_callback=progress_callback,
            cancel_callback=cancel_callback,
            current_index=current_index,
            total_files=total_files,
        )
        self._delete_stale_vectors(str(file_path), set(chunk_ids), document_id)
        self._mark_document_imported(document_id)
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "完成文件",
            f"{file_path.name} 已从文件断点恢复完成",
        )
        return "updated"

    def _import_pdf_batched(
        self,
        file_path: Path,
        file_hash: str,
        file_info: Any,
        target_profile: str,
        existing_document_id: str | None,
        existing_snapshot: dict[str, Any] | None,
        status: str,
        progress_callback: ProgressCallback | None,
        cancel_callback: Callable[[], None] | None,
        current_index: int,
        total_files: int,
    ) -> str:
        """Import a PDF by parsing page batches and writing incremental checkpoints.

        Each batch of ``pdf_page_batch_size`` pages is parsed, chunked, and
        persisted to SQLite immediately. Vectors are written after the full
        parse so the embedding provider sees realistic batch sizes, but the
        chunk rows are durable on disk from the moment a page batch finishes.
        This means a very large PDF that is interrupted mid-parse can resume
        from the existing chunk checkpoint rather than re-parsing from the start.
        """
        pdf_parser = PDFParser(self.settings)
        all_segments: list = []
        all_chunks: list = []
        document_id: str | None = None

        try:
            batch_num = 0
            for page_batch in pdf_parser.iter_page_batches(file_path, cancel_callback=cancel_callback):
                self._check_cancelled(cancel_callback)
                batch_num += 1
                batch_chunks = self.chunker.chunk(page_batch)
                if not batch_chunks:
                    continue

                # On the first batch create the document row and first chunk rows.
                # On subsequent batches only append more chunk rows.
                if document_id is None:
                    metadata = self.metadata_extractor.extract(file_path, page_batch)
                    self._emit_progress(
                        progress_callback,
                        current_index,
                        total_files,
                        str(file_path),
                        "写入元数据",
                        f"正在写入 SQLite checkpoint（第 {batch_num} 批分块）",
                    )
                    document_id, _chunk_ids, _docs, _metas = self._create_indexing_checkpoint(
                        file_path=file_path,
                        file_hash=file_hash,
                        file_info=file_info,
                        metadata=metadata,
                        target_profile=target_profile,
                        chunks=batch_chunks,
                        existing_document_id=existing_document_id,
                    )
                    # After first batch existing_document_id is consumed; the
                    # new document row owns the path from now on.
                    existing_document_id = None
                else:
                    self._emit_progress(
                        progress_callback,
                        current_index,
                        total_files,
                        str(file_path),
                        "写入元数据",
                        f"追加 SQLite checkpoint（第 {batch_num} 批分块）",
                    )
                    self._append_chunks_to_document(document_id, batch_chunks)

                all_segments.extend(page_batch)
                all_chunks.extend(batch_chunks)

            if not all_chunks:
                raise RuntimeError("No indexable text was extracted from the file.")

            if document_id is None:
                # File produced no valid batches at all — fall back to a single
                # empty checkpoint so downstream clean-up has a document_id.
                metadata = self.metadata_extractor.extract(file_path, [])
                document_id, _chunk_ids, _docs, _metas = self._create_indexing_checkpoint(
                    file_path=file_path,
                    file_hash=file_hash,
                    file_info=file_info,
                    metadata=metadata,
                    target_profile=target_profile,
                    chunks=[],
                    existing_document_id=existing_document_id,
                )
                raise RuntimeError("No indexable text was extracted from the file.")

            # Re-derive better metadata from first five segments of the full parse.
            if len(all_segments) > 5:
                full_metadata = self.metadata_extractor.extract(file_path, all_segments)
                self._update_document_metadata(document_id, full_metadata)

        except OperationCancelled:
            raise
        except Exception:
            if document_id:
                self.delete_document(document_id)
                if existing_snapshot:
                    try:
                        self._restore_document_snapshot(existing_snapshot)
                    except Exception:
                        logger.exception(
                            "Failed to restore previous document after batched PDF import failure"
                        )
            raise

        # Reload all chunk IDs / content / metadatas from SQLite so the vector
        # write loop works exactly like the non-batched path.
        checkpoint = self._load_indexing_checkpoint(document_id, file_path)
        if not checkpoint:
            self.delete_document(document_id)
            raise RuntimeError("Failed to reload PDF checkpoint for vectorization.")

        try:
            self._write_missing_vectors(
                chunk_ids=checkpoint["chunk_ids"],
                documents=checkpoint["documents"],
                metadatas=checkpoint["metadatas"],
                file_path=file_path,
                progress_callback=progress_callback,
                cancel_callback=cancel_callback,
                current_index=current_index,
                total_files=total_files,
            )
        except OperationCancelled:
            raise
        except Exception:
            self.delete_document(document_id)
            if existing_snapshot:
                try:
                    self._restore_document_snapshot(existing_snapshot)
                except Exception:
                    logger.exception(
                        "Failed to restore previous document after PDF vector write failure"
                    )
            raise

        self._delete_stale_vectors(str(file_path), set(checkpoint["chunk_ids"]), document_id)
        self._mark_document_imported(document_id)
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "完成文件",
            f"{file_path.name} 已{status}（分批解析，共 {len(all_chunks)} 个 chunk）",
        )
        return status

    def _append_chunks_to_document(self, document_id: str, chunks: list) -> None:
        """Append additional chunk rows to an existing indexing document."""
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if not document:
                raise RuntimeError(
                    f"Document {document_id} disappeared while appending PDF chunks."
                )
            # Determine the starting chunk index from existing rows.
            max_index_row = session.execute(
                select(sa_func.max(DocumentChunk.chunk_index)).where(
                    DocumentChunk.document_id == document_id
                )
            ).scalar_one_or_none()
            start_index = (max_index_row + 1) if max_index_row is not None else 0
            for offset, chunk in enumerate(chunks):
                chunk.chunk_index = start_index + offset
            self._persist_chunks(session, document, chunks)

    def _update_document_metadata(self, document_id: str, metadata: dict[str, Any]) -> None:
        """Overwrite document-level metadata fields after a full parse."""
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if not document:
                return
            if metadata.get("manufacturer"):
                document.manufacturer = metadata["manufacturer"]
            if metadata.get("sensor_model"):
                document.sensor_model = metadata["sensor_model"]
            document.metadata_json = metadata_to_json(metadata)

    def _load_indexing_checkpoint(self, document_id: str, file_path: Path) -> dict[str, Any] | None:
        """Load persisted chunks for an interrupted indexing document."""
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if (
                not document
                or document.status != "indexing"
                or document.file_path != str(file_path)
            ):
                return None
            chunks = session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document.id)
                .order_by(DocumentChunk.chunk_index)
            ).scalars().all()
            if not chunks:
                return None
            return {
                "chunk_ids": [chunk.id for chunk in chunks],
                "documents": [chunk.content for chunk in chunks],
                "metadatas": [self._chunk_metadata(document, chunk) for chunk in chunks],
            }

    def _create_indexing_checkpoint(
        self,
        file_path: Path,
        file_hash: str,
        file_info: Any,
        metadata: dict[str, Any],
        target_profile: str,
        chunks: list[TextChunk],
        existing_document_id: str | None,
    ) -> tuple[str, list[str], list[str], list[dict]]:
        """Persist parsed chunks before vectorization so the file can resume."""
        with session_scope(self.settings) as session:
            if existing_document_id:
                existing = session.get(Document, existing_document_id)
                if not existing:
                    existing = session.execute(
                        select(Document).where(Document.file_path == str(file_path))
                    ).scalar_one_or_none()
                if existing:
                    session.delete(existing)
                    session.flush()
            document = Document(
                file_path=str(file_path),
                filename=file_path.name,
                file_type=file_info.file_type,
                file_hash=file_hash,
                size_bytes=file_info.size_bytes,
                created_at=file_info.created_at,
                modified_at=file_info.modified_at,
                imported_at=utc_now(),
                status="indexing",
                manufacturer=metadata.get("manufacturer"),
                sensor_model=metadata.get("sensor_model"),
                tags=metadata.get("tags"),
                notes=metadata.get("notes"),
                metadata_json=metadata_to_json(metadata),
                index_profile=target_profile,
            )
            session.add(document)
            session.flush()
            db_chunks = self._persist_chunks(session, document, chunks)
            chunk_ids = [chunk.id for chunk in db_chunks]
            documents = [chunk.content for chunk in db_chunks]
            metadatas = [self._chunk_metadata(document, chunk) for chunk in db_chunks]
            return document.id, chunk_ids, documents, metadatas

    def _write_missing_vectors(
        self,
        chunk_ids: list[str],
        documents: list[str],
        metadatas: list[dict],
        file_path: Path,
        progress_callback: ProgressCallback | None,
        cancel_callback: Callable[[], None] | None,
        current_index: int,
        total_files: int,
    ) -> None:
        """Embed and upsert only chunks that are missing from ChromaDB."""
        self._check_cancelled(cancel_callback)
        existing_ids = self._existing_vector_ids(chunk_ids)
        missing_positions = [
            index for index, chunk_id in enumerate(chunk_ids) if chunk_id not in existing_ids
        ]
        if not missing_positions:
            self._emit_progress(
                progress_callback,
                current_index,
                total_files,
                str(file_path),
                "向量已完整",
                f"{file_path.name} 的 {len(chunk_ids)} 个向量已存在，正在刷新元数据",
            )
            self._update_vector_metadata(chunk_ids, metadatas)
            return

        batch_size = max(1, int(self.settings.embedding_batch_size or 1))
        completed = 0
        for batch_positions in _batches(missing_positions, batch_size):
            self._check_cancelled(cancel_callback)
            start = completed + 1
            end = completed + len(batch_positions)
            self._emit_progress(
                progress_callback,
                current_index,
                total_files,
                str(file_path),
                "向量化",
                f"正在生成缺失向量 {start}-{end}/{len(missing_positions)}",
            )
            batch_documents = [documents[index] for index in batch_positions]
            embeddings = self.embedding.embed_texts(batch_documents)
            self._check_cancelled(cancel_callback)
            batch_chunk_ids = [chunk_ids[index] for index in batch_positions]
            batch_metadatas = [metadatas[index] for index in batch_positions]
            self._emit_progress(
                progress_callback,
                current_index,
                total_files,
                str(file_path),
                "写入向量库",
                f"正在写入 ChromaDB batch {start}-{end}/{len(missing_positions)}",
            )
            self.vector_store.add_chunks(batch_chunk_ids, batch_documents, embeddings, batch_metadatas)
            completed = end
        self._update_vector_metadata(chunk_ids, metadatas)

    def _existing_vector_ids(self, chunk_ids: list[str]) -> set[str]:
        """Return already indexed chunk IDs when the vector store supports it."""
        get_existing_ids = getattr(self.vector_store, "get_existing_ids", None)
        if callable(get_existing_ids):
            return set(get_existing_ids(chunk_ids))
        return set()

    def _update_vector_metadata(self, chunk_ids: list[str], metadatas: list[dict]) -> None:
        """Refresh vector metadata when the vector store supports it."""
        update_metadata = getattr(self.vector_store, "update_metadata", None)
        if callable(update_metadata):
            update_metadata(chunk_ids, metadatas)

    def _delete_stale_vectors(
        self, file_path: str, keep_chunk_ids: set[str], document_id: str | None = None
    ) -> None:
        """Clean old vectors for a file when the vector store supports it."""
        delete_stale_for_file = getattr(self.vector_store, "delete_stale_for_file", None)
        if callable(delete_stale_for_file):
            delete_stale_for_file(file_path, keep_chunk_ids, document_id)

    def list_documents(self) -> list[dict]:
        """Return imported documents for UI display."""
        with session_scope(self.settings) as session:
            documents = session.execute(
                select(Document)
                .where(Document.status == "imported")
                .order_by(Document.imported_at.desc())
            )
            return [
                {
                    "id": document.id,
                    "filename": document.filename,
                    "file_path": document.file_path,
                    "file_type": document.file_type,
                    "sensor_model": document.sensor_model,
                    "manufacturer": document.manufacturer,
                    "status": document.status,
                    "modified_at": document.modified_at,
                    "imported_at": document.imported_at,
                }
                for document in documents.scalars().all()
            ]

    def delete_document(self, document_id: str) -> None:
        """Delete one document and its indexed chunks."""
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if not document:
                return
            self.vector_store.delete_by_document_id(document_id)
            session.delete(document)

    def stats(self) -> dict[str, int]:
        """Return basic database statistics."""
        with session_scope(self.settings) as session:
            document_count = len(
                session.execute(
                    select(Document.id).where(Document.status == "imported")
                ).all()
            )
            indexing_count = len(
                session.execute(
                    select(Document.id).where(Document.status == "indexing")
                ).all()
            )
            chunk_count = len(
                session.execute(
                    select(DocumentChunk.id)
                    .join(Document, Document.id == DocumentChunk.document_id)
                    .where(Document.status == "imported")
                ).all()
            )
        return {
            "documents": document_count,
            "indexing": indexing_count,
            "chunks": chunk_count,
            "vectors": self.vector_store.count(),
        }

    def refresh_sensor_models(self, overwrite: bool = True) -> dict[str, int]:
        """Re-derive sensor model and manufacturer hints for imported documents.

        Reuses the chunks already stored in SQLite, so no source file is parsed
        and no embedding is recomputed. Updates the SQLite document row and the
        affected Chroma chunk metadata in place.

        Args:
            overwrite: When True, replace existing hints; when False, only fill
                fields that are currently empty.

        Returns:
            Counts of scanned and updated documents.
        """
        with session_scope(self.settings) as session:
            document_ids = [
                row[0]
                for row in session.execute(
                    select(Document.id).where(Document.status == "imported")
                ).all()
            ]
        scanned = 0
        updated = 0
        for document_id in document_ids:
            scanned += 1
            if self._refresh_document_hints(document_id, overwrite):
                updated += 1
        return {"scanned": scanned, "updated": updated}

    def _refresh_document_hints(self, document_id: str, overwrite: bool) -> bool:
        """Recompute hints for one document from its stored chunks."""
        chunk_ids_to_sync: list[str] = []
        metadatas_to_sync: list[dict] = []
        changed = False
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if not document or document.status != "imported":
                return False
            chunks = session.execute(
                select(DocumentChunk)
                .where(DocumentChunk.document_id == document_id)
                .order_by(DocumentChunk.chunk_index)
            ).scalars().all()
            if not chunks:
                return False
            evidence = "\n".join(chunk.content for chunk in chunks[:5])
            hints = self.metadata_extractor.extract_hints_from_text(evidence)
            new_model = hints.get("sensor_model")
            new_manufacturer = hints.get("manufacturer")

            if (overwrite or not document.sensor_model) and (
                new_model and new_model != document.sensor_model
            ):
                document.sensor_model = new_model
                changed = True
            if (overwrite or not document.manufacturer) and (
                new_manufacturer and new_manufacturer != document.manufacturer
            ):
                document.manufacturer = new_manufacturer
                changed = True

            if changed:
                document.metadata_json = self._merge_metadata_json(
                    document.metadata_json,
                    document.sensor_model,
                    document.manufacturer,
                )
                chunk_ids_to_sync = [chunk.id for chunk in chunks]
                metadatas_to_sync = [
                    self._chunk_metadata(document, chunk) for chunk in chunks
                ]

        if changed and chunk_ids_to_sync:
            self.vector_store.update_metadata(chunk_ids_to_sync, metadatas_to_sync)
        return changed

    @staticmethod
    def _merge_metadata_json(
        metadata_json: str | None,
        sensor_model: str | None,
        manufacturer: str | None,
    ) -> str:
        """Update stored metadata JSON with refreshed hints."""
        try:
            metadata = json.loads(metadata_json or "{}")
        except json.JSONDecodeError:
            metadata = {}
        if not isinstance(metadata, dict):
            metadata = {}
        metadata["sensor_model"] = sensor_model
        metadata["manufacturer"] = manufacturer
        return metadata_to_json(metadata)

    def _find_reusable_document(
        self,
        session,
        file_hash: str,
        target_profile: str,
        exclude_document_id: str | None = None,
    ) -> Document | None:
        """Return an already indexed document with the same file content hash."""
        statement = select(Document).where(
            Document.file_hash == file_hash,
            Document.status == "imported",
        )
        if exclude_document_id:
            statement = statement.where(Document.id != exclude_document_id)
        documents = session.execute(statement.order_by(Document.imported_at)).scalars().all()
        for document in documents:
            if not profile_satisfies(document.index_profile, target_profile):
                continue
            chunk_id = session.execute(
                select(DocumentChunk.id)
                .where(DocumentChunk.document_id == document.id)
                .limit(1)
            ).scalar_one_or_none()
            if chunk_id:
                return document
        return None

    def _try_reuse_indexed_document(
        self,
        file_path: Path,
        file_hash: str,
        file_info: Any,
        source_snapshot: dict[str, Any],
        target_profile: str,
        existing_document_id: str | None,
        existing_snapshot: dict[str, Any] | None,
        progress_callback: ProgressCallback | None,
        current_index: int,
        total_files: int,
    ) -> str | None:
        """Create a document for this path by reusing chunks and embeddings from the same hash."""
        source_document = source_snapshot["document"]
        source_chunks = source_snapshot["chunks"]
        if not source_chunks:
            return None

        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "复用文件哈希",
            f"文件哈希已入库，复用 {source_document['filename']} 的分块和向量",
        )

        document_id: str | None = None
        source_to_target: dict[str, str] = {}
        target_documents: dict[str, str] = {}
        target_metadatas: dict[str, dict] = {}

        try:
            with session_scope(self.settings) as session:
                if existing_document_id:
                    existing = session.get(Document, existing_document_id)
                    if not existing:
                        existing = session.execute(
                            select(Document).where(Document.file_path == str(file_path))
                        ).scalar_one_or_none()
                    if existing:
                        session.delete(existing)
                        session.flush()

                document = Document(
                    file_path=str(file_path),
                    filename=file_path.name,
                    file_type=file_info.file_type,
                    file_hash=file_hash,
                    size_bytes=file_info.size_bytes,
                    created_at=file_info.created_at,
                    modified_at=file_info.modified_at,
                    imported_at=utc_now(),
                    status="indexing",
                    manufacturer=source_document.get("manufacturer"),
                    sensor_model=source_document.get("sensor_model"),
                    tags=source_document.get("tags"),
                    notes=source_document.get("notes"),
                    metadata_json=self._reused_metadata_json(
                        file_path,
                        file_info,
                        file_hash,
                        source_document,
                    ),
                    index_profile=source_document.get("index_profile") or target_profile,
                )
                session.add(document)
                session.flush()
                document_id = document.id

                for source_chunk in source_chunks:
                    db_chunk = DocumentChunk(
                        document_id=document.id,
                        chunk_index=source_chunk["chunk_index"],
                        content=source_chunk["content"],
                        content_type=source_chunk["content_type"],
                        page_number=source_chunk["page_number"],
                        source_label=self._source_label(
                            document.filename,
                            source_chunk["page_number"],
                            source_chunk["chunk_index"],
                        ),
                        metadata_json=source_chunk["metadata_json"],
                    )
                    session.add(db_chunk)
                    session.flush()
                    source_to_target[source_chunk["id"]] = db_chunk.id
                    target_documents[db_chunk.id] = db_chunk.content
                    target_metadatas[db_chunk.id] = self._chunk_metadata(document, db_chunk)

            source_embeddings = self.vector_store.get_embeddings(list(source_to_target))
            missing = [
                source_id
                for source_id in source_to_target
                if source_id not in source_embeddings
            ]
            if missing:
                raise RuntimeError(
                    f"Existing vectors are incomplete for reused hash {file_hash}: {len(missing)} missing"
                )

            source_ids = list(source_to_target)
            target_ids = [source_to_target[source_id] for source_id in source_ids]
            self.vector_store.add_chunks(
                target_ids,
                [target_documents[chunk_id] for chunk_id in target_ids],
                [source_embeddings[source_id] for source_id in source_ids],
                [target_metadatas[chunk_id] for chunk_id in target_ids],
            )
        except Exception as exc:
            if document_id:
                self.delete_document(document_id)
            if existing_snapshot:
                try:
                    self._restore_document_snapshot(existing_snapshot)
                except Exception:
                    logger.exception("Failed to restore previous document after hash reuse failure")
            logger.warning("Hash reuse failed for %s, falling back to full import: %s", file_path, exc)
            return None

        if existing_document_id:
            self.vector_store.delete_by_document_id(existing_document_id)
        if document_id:
            self._mark_document_imported(document_id)

        status = "updated" if existing_document_id else "skipped"
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "完成文件",
            f"{file_path.name} 已复用相同哈希的向量",
        )
        return status

    @staticmethod
    def _reused_metadata_json(
        file_path: Path,
        file_info: Any,
        file_hash: str,
        source_document: dict[str, Any],
    ) -> str:
        """Build metadata for a duplicate path that reuses an indexed hash."""
        try:
            metadata = json.loads(source_document.get("metadata_json") or "{}")
        except json.JSONDecodeError:
            metadata = {}
        metadata.update(
            {
                "filename": file_path.name,
                "file_path": str(file_path),
                "file_type": file_info.file_type,
                "size_bytes": file_info.size_bytes,
                "created_at": file_info.created_at,
                "modified_at": file_info.modified_at,
                "file_hash": file_hash,
                "reused_from_document_id": source_document.get("id"),
                "reused_from_file_path": source_document.get("file_path"),
            }
        )
        return metadata_to_json(metadata)

    def _snapshot_document(self, session, document: Document) -> dict[str, Any]:
        """Copy a document and its chunks so an interrupted update can be restored."""
        chunks = session.execute(
            select(DocumentChunk).where(DocumentChunk.document_id == document.id)
        ).scalars().all()
        return {
            "document": {
                "id": document.id,
                "file_path": document.file_path,
                "filename": document.filename,
                "file_type": document.file_type,
                "file_hash": document.file_hash,
                "size_bytes": document.size_bytes,
                "created_at": document.created_at,
                "modified_at": document.modified_at,
                "imported_at": document.imported_at,
                "status": document.status,
                "error_message": document.error_message,
                "manufacturer": document.manufacturer,
                "sensor_model": document.sensor_model,
                "tags": document.tags,
                "notes": document.notes,
                "metadata_json": document.metadata_json,
                "index_profile": document.index_profile,
            },
            "chunks": [
                {
                    "id": chunk.id,
                    "document_id": chunk.document_id,
                    "chunk_index": chunk.chunk_index,
                    "content": chunk.content,
                    "content_type": chunk.content_type,
                    "page_number": chunk.page_number,
                    "source_label": chunk.source_label,
                    "metadata_json": chunk.metadata_json,
                    "created_at": chunk.created_at,
                }
                for chunk in chunks
            ],
        }

    def _restore_document_snapshot(self, snapshot: dict[str, Any]) -> None:
        """Restore a previously indexed document after a failed update."""
        document_data = snapshot["document"]
        snapshot_doc_id = document_data.get("id")
        snapshot_chunk_ids = {chunk["id"] for chunk in snapshot.get("chunks", [])}

        with session_scope(self.settings) as session:
            current = session.execute(
                select(Document).where(Document.file_path == document_data["file_path"])
            ).scalar_one_or_none()
            if current:
                session.delete(current)
                session.flush()

            if snapshot_doc_id:
                conflict_doc = session.get(Document, snapshot_doc_id)
                if conflict_doc:
                    session.delete(conflict_doc)
                    session.flush()

            for chunk_id in snapshot_chunk_ids:
                conflict_chunk = session.get(DocumentChunk, chunk_id)
                if conflict_chunk:
                    session.delete(conflict_chunk)
            session.flush()

            session.add(Document(**document_data))
            for chunk_data in snapshot["chunks"]:
                session.add(DocumentChunk(**chunk_data))

    def _mark_document_imported(self, document_id: str) -> None:
        """Mark a document visible only after its vectors are durable."""
        with session_scope(self.settings) as session:
            document = session.get(Document, document_id)
            if document:
                document.status = "imported"
                document.error_message = None
                document.imported_at = utc_now()

    def _persist_chunks(
        self,
        session,
        document: Document,
        chunks: list[TextChunk],
    ) -> list[DocumentChunk]:
        """Persist chunk rows and return them."""
        db_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            db_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_type=chunk.content_type,
                page_number=chunk.page_number,
                source_label=self._source_label(
                    document.filename,
                    chunk.page_number,
                    chunk.chunk_index,
                ),
                metadata_json=json.dumps(chunk.metadata, ensure_ascii=False, default=str),
            )
            session.add(db_chunk)
            db_chunks.append(db_chunk)
        session.flush()
        return db_chunks

    @staticmethod
    def _source_label(filename: str, page_number: int | None, chunk_index: int) -> str:
        """Build a compact source label for one document chunk."""
        page = f" p.{page_number}" if page_number else ""
        return f"{filename}{page} #{chunk_index}"

    @staticmethod
    def _chunk_metadata(document: Document, chunk: DocumentChunk) -> dict:
        """Build Chroma metadata for one chunk."""
        return {
            "document_id": document.id,
            "source_label": chunk.source_label,
            "file_path": document.file_path,
            "file_type": document.file_type,
            "file_hash": document.file_hash,
            "filename": document.filename,
            "page_number": chunk.page_number,
            "sensor_model": document.sensor_model,
            "manufacturer": document.manufacturer,
            "chunk_index": chunk.chunk_index,
            "content_type": chunk.content_type,
        }

    @staticmethod
    def _emit_progress(
        callback: ProgressCallback | None,
        current: int,
        total: int,
        file_path: str,
        phase: str,
        message: str,
        level: str = "info",
    ) -> None:
        """Call progress callback while keeping backward compatibility."""
        if not callback:
            return
        try:
            callback(current, total, file_path, phase, message, level)
        except TypeError:
            callback(current, total, file_path)

    @staticmethod
    def _check_cancelled(cancel_callback: Callable[[], None] | None) -> None:
        """Run an optional cancellation checkpoint."""
        if cancel_callback:
            cancel_callback()


def _batches(items: list[int], size: int):
    """Yield fixed-size batches."""
    step = max(1, size)
    for index in range(0, len(items), step):
        yield items[index : index + step]
