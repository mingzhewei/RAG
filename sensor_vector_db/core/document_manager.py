"""Document import, de-duplication, indexing, and management."""

from __future__ import annotations

from datetime import datetime
import json
from pathlib import Path
from typing import Callable

from sqlalchemy import select

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.document_processor import (
    DocumentChunker,
    DocumentParserFactory,
    MetadataExtractor,
    metadata_to_json,
)
from sensor_vector_db.core.embedding import BaseEmbedding, create_embedding_provider
from sensor_vector_db.core.types import ImportErrorItem, ImportReport, TextChunk
from sensor_vector_db.core.vector_store import VectorStore
from sensor_vector_db.models.database import Document, DocumentChunk, session_scope
from sensor_vector_db.utils.file_utils import get_file_info, iter_supported_files
from sensor_vector_db.utils.hash_utils import calculate_file_md5
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)
ProgressCallback = Callable[..., None]


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
        current_index: int = 1,
        total_files: int = 1,
    ) -> str:
        """Import or update one supported file."""
        file_path = Path(path).resolve()
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
        with session_scope(self.settings) as session:
            existing = session.execute(
                select(Document).where(Document.file_path == str(file_path))
            ).scalar_one_or_none()
            if existing and existing.file_hash == file_hash:
                self._emit_progress(
                    progress_callback,
                    current_index,
                    total_files,
                    str(file_path),
                    "跳过",
                    "文件未变化，跳过入库",
                )
                return "skipped"
            if existing:
                self._emit_progress(
                    progress_callback,
                    current_index,
                    total_files,
                    str(file_path),
                    "更新",
                    "检测到文件变化，删除旧向量并重新入库",
                )
                self.vector_store.delete_by_document_id(existing.id)
                session.delete(existing)
                session.flush()
                status = "updated"
            else:
                status = "imported"

        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "解析文档",
            "正在解析正文、表格或 OCR 文本",
        )
        segments = self.parser_factory.parse(file_path)
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

        documents = [chunk.content for chunk in chunks]
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "向量化",
            f"正在生成 {len(documents)} 个文本块的 embedding",
        )
        embeddings = self.embedding.embed_texts(documents)

        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "写入元数据",
            f"正在写入 SQLite，chunk 数：{len(chunks)}",
        )
        document_id: str | None = None
        chunk_ids: list[str] = []
        metadatas: list[dict] = []
        with session_scope(self.settings) as session:
            document = Document(
                file_path=str(file_path),
                filename=file_path.name,
                file_type=file_info.file_type,
                file_hash=file_hash,
                size_bytes=file_info.size_bytes,
                created_at=file_info.created_at,
                modified_at=file_info.modified_at,
                imported_at=datetime.utcnow(),
                status="imported",
                manufacturer=metadata.get("manufacturer"),
                sensor_model=metadata.get("sensor_model"),
                tags=metadata.get("tags"),
                notes=metadata.get("notes"),
                metadata_json=metadata_to_json(metadata),
            )
            session.add(document)
            session.flush()
            document_id = document.id
            db_chunks = self._persist_chunks(session, document, chunks)
            chunk_ids = [chunk.id for chunk in db_chunks]
            metadatas = [
                self._chunk_metadata(document, chunk)
                for chunk in db_chunks
            ]

        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "写入向量库",
            "正在写入 ChromaDB",
        )
        try:
            self.vector_store.add_chunks(chunk_ids, documents, embeddings, metadatas)
        except Exception:
            if document_id:
                self.delete_document(document_id)
            raise
        self._emit_progress(
            progress_callback,
            current_index,
            total_files,
            str(file_path),
            "完成文件",
            f"{file_path.name} 已{status}",
        )
        return status

    def list_documents(self) -> list[dict]:
        """Return imported documents for UI display."""
        with session_scope(self.settings) as session:
            documents = session.execute(select(Document).order_by(Document.imported_at.desc()))
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
            document_count = len(session.execute(select(Document.id)).all())
            chunk_count = len(session.execute(select(DocumentChunk.id)).all())
        return {
            "documents": document_count,
            "chunks": chunk_count,
            "vectors": self.vector_store.count(),
        }

    def _persist_chunks(
        self,
        session,
        document: Document,
        chunks: list[TextChunk],
    ) -> list[DocumentChunk]:
        """Persist chunk rows and return them."""
        db_chunks: list[DocumentChunk] = []
        for chunk in chunks:
            page = f" p.{chunk.page_number}" if chunk.page_number else ""
            source_label = f"{document.filename}{page} #{chunk.chunk_index}"
            db_chunk = DocumentChunk(
                document_id=document.id,
                chunk_index=chunk.chunk_index,
                content=chunk.content,
                content_type=chunk.content_type,
                page_number=chunk.page_number,
                source_label=source_label,
                metadata_json=json.dumps(chunk.metadata, ensure_ascii=False, default=str),
            )
            session.add(db_chunk)
            db_chunks.append(db_chunk)
        session.flush()
        return db_chunks

    @staticmethod
    def _chunk_metadata(document: Document, chunk: DocumentChunk) -> dict:
        """Build Chroma metadata for one chunk."""
        return {
            "document_id": document.id,
            "source_label": chunk.source_label,
            "file_path": document.file_path,
            "file_type": document.file_type,
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
