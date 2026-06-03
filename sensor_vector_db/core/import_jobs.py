"""Persistent background import job orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any

from sqlalchemy import desc, select

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.models.database import (
    Document,
    ImportJob,
    ImportJobEvent,
    ImportJobFile,
    init_database,
    session_scope,
)
from sensor_vector_db.utils.file_utils import get_file_info, iter_supported_files
from sensor_vector_db.utils.hash_utils import calculate_file_md5
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)
_RUNNING_THREADS: dict[str, threading.Thread] = {}
_THREAD_LOCK = threading.Lock()
RETRYABLE_FILE_STATUSES = {"pending", "processing", "failed"}


@dataclass
class ImportJobSnapshot:
    """Read-only snapshot for UI rendering."""

    id: str
    source_path: str
    status: str
    phase: str
    message: str | None
    current_file: str | None
    total_files: int
    current_index: int
    imported: int
    updated: int
    skipped: int
    deleted: int
    failed: int
    created_at: datetime
    started_at: datetime | None
    updated_at: datetime
    finished_at: datetime | None
    is_thread_active: bool

    @property
    def progress_ratio(self) -> float:
        """Return progress as 0-1 ratio."""
        if self.total_files <= 0:
            return 0.0
        return min(1.0, max(0.0, self.current_index / self.total_files))

    @property
    def can_resume(self) -> bool:
        """Return whether this job can be resumed by the UI."""
        return self.status in {"queued", "running", "failed", "completed_with_errors"} and not self.is_thread_active


class ImportJobManager:
    """Create, run, resume, and inspect persistent import jobs."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize manager."""
        self.settings = settings or get_settings()
        init_database(self.settings)

    def start_import(self, source_path: str | Path) -> str:
        """Create a directory sync job and start it in a background thread."""
        path_text = str(Path(source_path).resolve())
        existing = self._find_resumable_job(path_text)
        if existing:
            self.resume_job(existing)
            return existing

        with session_scope(self.settings) as session:
            job = ImportJob(
                source_path=path_text,
                status="queued",
                phase="排队",
                message="导入任务已创建",
                current_file=path_text,
            )
            session.add(job)
            session.flush()
            job_id = job.id
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    phase="排队",
                    message=f"导入任务已创建：{path_text}",
                    file_path=path_text,
                )
            )
        self.resume_job(job_id)
        return job_id

    def resume_job(self, job_id: str) -> None:
        """Resume a queued, interrupted, failed, or partially completed job."""
        with _THREAD_LOCK:
            thread = _RUNNING_THREADS.get(job_id)
            if thread and thread.is_alive():
                return
            new_thread = threading.Thread(
                target=self._run_import,
                args=(job_id,),
                name=f"import-job-{job_id[:8]}",
                daemon=True,
            )
            _RUNNING_THREADS[job_id] = new_thread
            new_thread.start()

    def list_jobs(self, limit: int = 20) -> list[ImportJobSnapshot]:
        """Return recent import jobs."""
        with session_scope(self.settings) as session:
            jobs = session.execute(
                select(ImportJob).order_by(desc(ImportJob.created_at)).limit(limit)
            ).scalars().all()
            return [self._snapshot(job) for job in jobs]

    def get_job(self, job_id: str) -> ImportJobSnapshot | None:
        """Return one import job snapshot."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            return self._snapshot(job) if job else None

    def get_events(self, job_id: str, limit: int = 80) -> list[dict[str, Any]]:
        """Return recent progress events for a job."""
        with session_scope(self.settings) as session:
            events = session.execute(
                select(ImportJobEvent)
                .where(ImportJobEvent.job_id == job_id)
                .order_by(desc(ImportJobEvent.created_at))
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "time": event.created_at.strftime("%H:%M:%S"),
                    "level": event.level,
                    "phase": event.phase,
                    "message": event.message,
                    "file_path": event.file_path,
                }
                for event in reversed(events)
            ]

    def get_file_rows(self, job_id: str, limit: int = 500) -> list[dict[str, Any]]:
        """Return per-file status rows for the UI."""
        with session_scope(self.settings) as session:
            rows = session.execute(
                select(ImportJobFile)
                .where(ImportJobFile.job_id == job_id)
                .order_by(ImportJobFile.file_path)
                .limit(limit)
            ).scalars().all()
            return [
                {
                    "status": row.status,
                    "phase": row.phase,
                    "file_path": row.file_path,
                    "hash": row.file_hash,
                    "document_id": row.document_id,
                    "message": row.message,
                    "error": row.error_message,
                    "updated_at": row.updated_at.strftime("%Y-%m-%d %H:%M:%S"),
                }
                for row in rows
            ]

    def _run_import(self, job_id: str) -> None:
        """Background thread target."""
        try:
            source_path = self._mark_started(job_id)
            manager = DocumentManager(self.settings)
            self._prepare_plan(job_id, source_path, manager)
            pending_files = self._get_retryable_files(job_id)

            for sequence, file_row_id in enumerate(pending_files, start=1):
                self._process_one_file(job_id, file_row_id, sequence, len(pending_files), manager)

            self._finish_from_file_rows(job_id)
        except Exception as exc:
            logger.exception("Import job failed: %s", job_id)
            self._mark_finished(
                job_id,
                status="failed",
                phase="任务失败",
                message=classify_error(exc, "任务"),
                report={"error": str(exc)},
            )
        finally:
            with _THREAD_LOCK:
                _RUNNING_THREADS.pop(job_id, None)

    def _prepare_plan(self, job_id: str, source_path: str, manager: DocumentManager) -> None:
        """Scan source path, plan retries/skips, and remove deleted documents."""
        self._update_job(job_id, "扫描目录", "正在扫描目录并生成导入计划", source_path)
        files = iter_supported_files(source_path)
        current_paths = {str(path.resolve()) for path in files}
        deleted_count = self._reconcile_deleted_documents(job_id, source_path, current_paths, manager)

        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.deleted = deleted_count
                job.total_files = len(files)
                job.current_index = 0
                job.updated_at = datetime.utcnow()
            for row in session.execute(
                select(ImportJobFile).where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.status == "processing",
                )
            ).scalars():
                row.status = "pending"
                row.phase = "恢复排队"
                row.message = "上次任务中断，已重新加入待处理队列"
                row.updated_at = datetime.utcnow()

        for index, file_path in enumerate(files, start=1):
            self._update_job(
                job_id,
                "计算文件状态",
                f"正在检查 {index}/{len(files)}：{file_path.name}",
                str(file_path),
                current_index=index,
                total_files=len(files),
            )
            self._upsert_file_plan(job_id, file_path)

        self._update_counts(job_id)
        self._add_event(
            job_id,
            "计划完成",
            f"计划完成：当前目录 {len(files)} 个支持文件，清理已删除文档 {deleted_count} 个",
            source_path,
        )

    def _process_one_file(
        self,
        job_id: str,
        file_row_id: str,
        sequence: int,
        total_pending: int,
        manager: DocumentManager,
    ) -> None:
        """Process one planned file and persist its final state."""
        with session_scope(self.settings) as session:
            row = session.get(ImportJobFile, file_row_id)
            if not row:
                return
            file_path = row.file_path
            row.status = "processing"
            row.phase = "开始处理"
            row.message = f"正在处理待办 {sequence}/{total_pending}"
            row.started_at = row.started_at or datetime.utcnow()
            row.updated_at = datetime.utcnow()

        def progress(
            current: int,
            total: int,
            path: str,
            phase: str,
            message: str,
            level: str = "info",
        ) -> None:
            del current, total
            self._update_file_progress(job_id, file_row_id, path, phase, message, level)

        try:
            status = manager.import_file(
                file_path,
                progress_callback=progress,
                current_index=sequence,
                total_files=total_pending,
            )
            self._mark_file_success(job_id, file_row_id, status)
        except Exception as exc:
            self._mark_file_failed(job_id, file_row_id, exc)
        finally:
            self._update_counts(job_id)

    def _upsert_file_plan(self, job_id: str, file_path: Path) -> None:
        """Create or update one file row based on current hash and document state."""
        resolved = str(file_path.resolve())
        try:
            file_hash = calculate_file_md5(file_path)
            info = get_file_info(file_path)
        except Exception as exc:
            self._upsert_error_file(job_id, resolved, exc)
            return

        with session_scope(self.settings) as session:
            existing_doc = session.execute(
                select(Document).where(Document.file_path == resolved)
            ).scalar_one_or_none()
            row = session.execute(
                select(ImportJobFile).where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.file_path == resolved,
                )
            ).scalar_one_or_none()
            if not row:
                row = ImportJobFile(job_id=job_id, file_path=resolved)
                session.add(row)

            row.file_hash = file_hash
            row.size_bytes = info.size_bytes
            row.modified_at = info.modified_at
            row.updated_at = datetime.utcnow()

            if existing_doc and existing_doc.file_hash == file_hash:
                row.status = "skipped"
                row.phase = "已存在"
                row.message = "文件未变化，已有向量，跳过"
                row.document_id = existing_doc.id
                row.error_message = None
                row.finished_at = datetime.utcnow()
            else:
                row.status = "pending"
                row.phase = "等待处理"
                row.message = "新增文件" if not existing_doc else "文件已修改，等待更新向量"
                row.document_id = existing_doc.id if existing_doc else None
                row.error_message = None
                row.finished_at = None

    def _upsert_error_file(self, job_id: str, file_path: str, exc: Exception) -> None:
        """Persist a file planning failure."""
        error = classify_error(exc, "扫描")
        with session_scope(self.settings) as session:
            row = session.execute(
                select(ImportJobFile).where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.file_path == file_path,
                )
            ).scalar_one_or_none()
            if not row:
                row = ImportJobFile(job_id=job_id, file_path=file_path)
                session.add(row)
            row.status = "failed"
            row.phase = "扫描失败"
            row.message = error
            row.error_message = error
            row.updated_at = datetime.utcnow()
            row.finished_at = datetime.utcnow()
        self._add_event(job_id, "扫描失败", error, file_path, level="error")

    def _reconcile_deleted_documents(
        self,
        job_id: str,
        source_path: str,
        current_paths: set[str],
        manager: DocumentManager,
    ) -> int:
        """Delete indexed documents whose source files no longer exist under the directory."""
        source = Path(source_path)
        if source.is_file():
            return 0
        root = str(source.resolve())
        deleted = 0
        with session_scope(self.settings) as session:
            documents = session.execute(select(Document)).scalars().all()
            stale = [
                document
                for document in documents
                if _is_under_root(document.file_path, root)
                and document.file_path not in current_paths
            ]
            stale_ids = [(document.id, document.file_path, document.filename) for document in stale]

        for document_id, file_path, filename in stale_ids:
            manager.delete_document(document_id)
            deleted += 1
            self._add_event(
                job_id,
                "删除同步",
                f"源文件已不存在，已清理旧向量：{filename}",
                file_path,
            )
        return deleted

    def _get_retryable_files(self, job_id: str) -> list[str]:
        """Return file row IDs that need import or retry."""
        with session_scope(self.settings) as session:
            rows = session.execute(
                select(ImportJobFile).where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.status.in_(RETRYABLE_FILE_STATUSES),
                ).order_by(ImportJobFile.file_path)
            ).scalars().all()
            return [row.id for row in rows]

    def _mark_started(self, job_id: str) -> str:
        """Mark a job as running and return its source path."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if not job:
                raise ValueError(f"Import job not found: {job_id}")
            job.status = "running"
            job.phase = "启动"
            job.message = "后台导入线程已启动"
            job.started_at = job.started_at or datetime.utcnow()
            job.finished_at = None
            job.updated_at = datetime.utcnow()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    phase="启动",
                    message="后台导入线程已启动或已恢复",
                    file_path=job.source_path,
                )
            )
            return job.source_path

    def _update_file_progress(
        self,
        job_id: str,
        file_row_id: str,
        file_path: str,
        phase: str,
        message: str,
        level: str,
    ) -> None:
        """Persist job-level and file-level progress."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            row = session.get(ImportJobFile, file_row_id)
            if job:
                job.status = "running"
                job.phase = phase
                job.message = message
                job.current_file = file_path
                job.updated_at = datetime.utcnow()
            if row:
                row.status = "processing"
                row.phase = phase
                row.message = message
                row.updated_at = datetime.utcnow()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level=level,
                    phase=phase,
                    message=message,
                    file_path=file_path,
                )
            )

    def _mark_file_success(self, job_id: str, file_row_id: str, status: str) -> None:
        """Mark one file as imported, updated, or skipped."""
        with session_scope(self.settings) as session:
            row = session.get(ImportJobFile, file_row_id)
            if not row:
                return
            document = session.execute(
                select(Document).where(Document.file_path == row.file_path)
            ).scalar_one_or_none()
            row.status = status
            row.phase = "完成文件"
            row.message = _success_message(status)
            row.document_id = document.id if document else row.document_id
            row.file_hash = document.file_hash if document else row.file_hash
            row.error_message = None
            row.updated_at = datetime.utcnow()
            row.finished_at = datetime.utcnow()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level="info",
                    phase="完成文件",
                    message=f"{Path(row.file_path).name}：{_success_message(status)}",
                    file_path=row.file_path,
                )
            )

    def _mark_file_failed(self, job_id: str, file_row_id: str, exc: Exception) -> None:
        """Mark one file as failed with a classified error."""
        with session_scope(self.settings) as session:
            row = session.get(ImportJobFile, file_row_id)
            if not row:
                return
            phase = row.phase or "处理"
            error = classify_error(exc, phase)
            row.status = "failed"
            row.message = error
            row.error_message = error
            row.updated_at = datetime.utcnow()
            row.finished_at = datetime.utcnow()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level="error",
                    phase="文件失败",
                    message=f"{Path(row.file_path).name}：{error}",
                    file_path=row.file_path,
                )
            )

    def _finish_from_file_rows(self, job_id: str) -> None:
        """Finish a job using persisted per-file status rows."""
        counts = self._update_counts(job_id)
        status = "completed_with_errors" if counts["failed"] else "completed"
        phase = "完成，有失败文件" if counts["failed"] else "完成"
        message = (
            f"导入 {counts['imported']}，更新 {counts['updated']}，跳过 {counts['skipped']}，"
            f"删除 {counts['deleted']}，失败 {counts['failed']}"
        )
        self._mark_finished(job_id, status, phase, message, counts)

    def _update_counts(self, job_id: str) -> dict[str, int]:
        """Recompute job counters from file rows."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            rows = session.execute(
                select(ImportJobFile).where(ImportJobFile.job_id == job_id)
            ).scalars().all()
            counts = {
                "scanned": len(rows),
                "imported": sum(1 for row in rows if row.status == "imported"),
                "updated": sum(1 for row in rows if row.status == "updated"),
                "skipped": sum(1 for row in rows if row.status == "skipped"),
                "failed": sum(1 for row in rows if row.status == "failed"),
                "deleted": job.deleted if job else 0,
            }
            if job:
                job.total_files = counts["scanned"]
                job.imported = counts["imported"]
                job.updated = counts["updated"]
                job.skipped = counts["skipped"]
                job.failed = counts["failed"]
                completed = counts["imported"] + counts["updated"] + counts["skipped"] + counts["failed"]
                job.current_index = min(completed, counts["scanned"])
                job.updated_at = datetime.utcnow()
            return counts

    def _update_job(
        self,
        job_id: str,
        phase: str,
        message: str,
        file_path: str | None = None,
        current_index: int | None = None,
        total_files: int | None = None,
    ) -> None:
        """Update job headline status."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if not job:
                return
            job.status = "running"
            job.phase = phase
            job.message = message
            job.current_file = file_path
            if current_index is not None:
                job.current_index = current_index
            if total_files is not None:
                job.total_files = total_files
            job.updated_at = datetime.utcnow()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    phase=phase,
                    message=message,
                    file_path=file_path,
                )
            )

    def _mark_finished(
        self,
        job_id: str,
        status: str,
        phase: str,
        message: str,
        report: dict[str, Any],
    ) -> None:
        """Persist final status and report."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if not job:
                return
            job.status = status
            job.phase = phase
            job.message = message
            job.finished_at = datetime.utcnow()
            job.updated_at = datetime.utcnow()
            job.report_json = json.dumps(report, ensure_ascii=False, default=str)
            job.imported = int(report.get("imported", job.imported or 0))
            job.updated = int(report.get("updated", job.updated or 0))
            job.skipped = int(report.get("skipped", job.skipped or 0))
            job.deleted = int(report.get("deleted", job.deleted or 0))
            job.failed = int(report.get("failed", job.failed or 0))
            if report.get("scanned") is not None:
                job.total_files = int(report.get("scanned", job.total_files or 0))
                job.current_index = int(report.get("scanned", job.current_index or 0))
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level="error" if status == "failed" else "info",
                    phase=phase,
                    message=message,
                    file_path=job.current_file,
                )
            )

    def _add_event(
        self,
        job_id: str,
        phase: str,
        message: str,
        file_path: str | None = None,
        level: str = "info",
    ) -> None:
        """Add a progress event."""
        with session_scope(self.settings) as session:
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level=level,
                    phase=phase,
                    message=message,
                    file_path=file_path,
                )
            )

    def _find_resumable_job(self, source_path: str) -> str | None:
        """Find the latest resumable job for a source path."""
        with session_scope(self.settings) as session:
            job = session.execute(
                select(ImportJob)
                .where(
                    ImportJob.source_path == source_path,
                    ImportJob.status.in_({"queued", "running", "failed", "completed_with_errors"}),
                )
                .order_by(desc(ImportJob.created_at))
                .limit(1)
            ).scalar_one_or_none()
            if not job:
                return None
            with _THREAD_LOCK:
                thread = _RUNNING_THREADS.get(job.id)
                if thread and thread.is_alive():
                    return job.id
            return job.id

    @staticmethod
    def _snapshot(job: ImportJob) -> ImportJobSnapshot:
        """Create a UI snapshot from a database row."""
        with _THREAD_LOCK:
            thread = _RUNNING_THREADS.get(job.id)
            active = bool(thread and thread.is_alive())
        return ImportJobSnapshot(
            id=job.id,
            source_path=job.source_path,
            status=job.status,
            phase=job.phase,
            message=job.message,
            current_file=job.current_file,
            total_files=job.total_files,
            current_index=job.current_index,
            imported=job.imported,
            updated=job.updated,
            skipped=job.skipped,
            deleted=job.deleted,
            failed=job.failed,
            created_at=job.created_at,
            started_at=job.started_at,
            updated_at=job.updated_at,
            finished_at=job.finished_at,
            is_thread_active=active,
        )


def classify_error(exc: Exception, phase: str = "") -> str:
    """Return a user-facing error category and action hint."""
    text = str(exc)
    lowered = text.lower()
    if any(key in lowered for key in ("api key", "unauthorized", "401", "invalid token", "deepseek")):
        return f"DeepSeek token 或鉴权错误：{text}"
    if any(key in lowered for key in ("timeout", "connection", "connect", "network", "503", "502", "504")):
        return f"服务器连接或响应失败：{text}"
    if "ocr" in lowered or "paddle" in lowered:
        return f"OCR 处理失败：{text}"
    if any(key in lowered for key in ("embedding", "flagembedding", "bge", "huggingface", "torch")):
        return f"向量化失败：{text}"
    if any(key in lowered for key in ("chroma", "vector")):
        return f"向量库写入或检索失败：{text}"
    if any(key in lowered for key in ("no indexable text", "pdf", "docx", "parse")) or "解析" in phase:
        return f"文档解析失败：{text}"
    if any(key in lowered for key in ("permission", "not found", "no such file", "access")):
        return f"本地文件访问失败：{text}"
    return f"{phase or '处理'}失败：{text}"


def _success_message(status: str) -> str:
    """Return a localized success status message."""
    return {
        "imported": "新增文件已完成向量化",
        "updated": "已更新文件向量",
        "skipped": "文件未变化，已跳过",
    }.get(status, status)


def _is_under_root(file_path: str, root: str) -> bool:
    """Return whether file_path is under root, using resolved paths."""
    try:
        Path(file_path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False
