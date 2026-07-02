"""Persistent background import job orchestration."""

from __future__ import annotations

import atexit
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import threading
from typing import Any

from sqlalchemy import case, desc, func, select, update

from sensor_vector_db.config.settings import Settings, get_settings
from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.core.index_profile import build_index_profile, profile_satisfies
from sensor_vector_db.core.types import OperationCancelled
from sensor_vector_db.models.database import (
    Document,
    ImportJob,
    ImportJobEvent,
    ImportJobFile,
    init_database,
    session_scope,
    utc_now,
)
from sensor_vector_db.utils.file_utils import get_file_info, iter_supported_files
from sensor_vector_db.utils.hash_utils import calculate_file_md5
from sensor_vector_db.utils.logger import get_logger


logger = get_logger(__name__)
_RUNNING_THREADS: dict[str, threading.Thread] = {}
_STOP_EVENTS: dict[str, threading.Event] = {}
_JOB_SETTINGS: dict[str, Settings] = {}
_THREAD_LOCK = threading.Lock()
RETRYABLE_FILE_STATUSES = {"pending", "processing", "failed"}
STOP_REQUESTED_STATUS = "stopping"
RUNNING_OR_STOPPING_STATUSES = {"running", STOP_REQUESTED_STATUS}
RESUMABLE_JOB_STATUSES = {
    "queued",
    "running",
    "interrupted",
    "failed",
    "completed_with_errors",
}


class ImportCancelled(OperationCancelled):
    """Raised when an import job has been asked to stop."""


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
        return min(1.0, max(0.0, self.completed_files / self.total_files))

    @property
    def completed_files(self) -> int:
        """Return files that reached a final file-level state."""
        return min(
            self.total_files,
            max(0, self.imported + self.updated + self.skipped + self.failed),
        )

    @property
    def pending_files(self) -> int:
        """Return files not yet in a final file-level state."""
        return max(0, self.total_files - self.completed_files)

    @property
    def can_resume(self) -> bool:
        """Return whether this job can be resumed by the UI."""
        return self.status in RESUMABLE_JOB_STATUSES and not self.is_thread_active


class ImportJobManager:
    """Create, run, resume, and inspect persistent import jobs."""

    def __init__(self, settings: Settings | None = None) -> None:
        """Initialize manager."""
        self.settings = settings or get_settings()
        init_database(self.settings)
        self._recover_orphaned_running_jobs()

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
            stop_event = threading.Event()
            new_thread = threading.Thread(
                target=self._run_import,
                args=(job_id, stop_event),
                name=f"import-job-{job_id[:8]}",
                daemon=True,
            )
            _STOP_EVENTS[job_id] = stop_event
            _RUNNING_THREADS[job_id] = new_thread
            _JOB_SETTINGS[job_id] = self.settings
            new_thread.start()

    def request_stop(self, job_id: str) -> None:
        """Ask a running import job to stop at the next safe checkpoint."""
        with _THREAD_LOCK:
            stop_event = _STOP_EVENTS.get(job_id)
            if stop_event:
                stop_event.set()
        _persist_stop_request(self.settings, [job_id])

    def request_stop_all(self) -> None:
        """Ask all running import jobs to stop at the next safe checkpoint."""
        request_stop_all_running_jobs()

    def _recover_orphaned_running_jobs(self) -> None:
        """Mark persisted running jobs without a live worker as resumable."""
        with _THREAD_LOCK:
            active_job_ids = {
                job_id
                for job_id, thread in _RUNNING_THREADS.items()
                if thread.is_alive()
            }

        recovered_jobs = 0
        recovered_files = 0
        with session_scope(self.settings) as session:
            jobs = session.execute(
                select(ImportJob).where(ImportJob.status.in_(RUNNING_OR_STOPPING_STATUSES))
            ).scalars().all()
            for job in jobs:
                if job.id in active_job_ids:
                    continue
                recovered_jobs += 1
                now = utc_now()
                job.status = "interrupted"
                job.phase = "上次运行已中断"
                job.message = "检测到后台导入线程已不存在；任务已标记为可恢复，未完成文件会重新排队。"
                job.finished_at = now
                job.updated_at = now

                processing_rows = session.execute(
                    select(ImportJobFile).where(
                        ImportJobFile.job_id == job.id,
                        ImportJobFile.status == "processing",
                    )
                ).scalars().all()
                for row in processing_rows:
                    recovered_files += 1
                    row.status = "pending"
                    row.phase = "恢复排队"
                    row.message = "上次运行退出时正在处理该文件，已放回待处理队列"
                    row.error_message = None
                    row.updated_at = now

                session.add(
                    ImportJobEvent(
                        job_id=job.id,
                        level="warning",
                        phase="上次运行已中断",
                        message="启动时发现 running 任务没有对应后台线程，已转为 interrupted，可手动恢复。",
                        file_path=job.current_file,
                    )
                )

        if recovered_jobs:
            logger.info(
                "孤儿导入任务恢复：%d 个任务中断，%d 个文件重新排队（可手动恢复）",
                recovered_jobs,
                recovered_files,
            )

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

    def get_file_status_counts(self, job_id: str) -> dict[str, int]:
        """Return exact per-file status counts for a job."""
        with session_scope(self.settings) as session:
            rows = session.execute(
                select(ImportJobFile.status, func.count())
                .where(ImportJobFile.job_id == job_id)
                .group_by(ImportJobFile.status)
            ).all()
            return {str(status): int(count) for status, count in rows}

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

    def get_failed_rows(self, job_id: str) -> list[dict[str, Any]]:
        """Return ALL failed file rows for the job — no limit applied."""
        with session_scope(self.settings) as session:
            rows = session.execute(
                select(ImportJobFile)
                .where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.status == "failed",
                )
                .order_by(ImportJobFile.file_path)
            ).scalars().all()
            return [
                {
                    "file_path": row.file_path,
                    "error": row.error_message,
                    "message": row.message,
                }
                for row in rows
            ]

    def _run_import(self, job_id: str, stop_event: threading.Event) -> None:
        """Background thread target."""
        try:
            source_path = self._mark_started(job_id)
            manager = DocumentManager(self.settings)
            self._raise_if_cancelled(job_id, stop_event)
            self._prepare_plan(job_id, source_path, manager)
            pending_files = self._get_retryable_files(job_id)

            for sequence, file_row_id in enumerate(pending_files, start=1):
                self._raise_if_cancelled(job_id, stop_event)
                self._process_one_file(
                    job_id,
                    file_row_id,
                    sequence,
                    len(pending_files),
                    manager,
                    stop_event,
                )

            self._finish_from_file_rows(job_id)
        except ImportCancelled:
            counts = self._update_counts(job_id)
            self._mark_finished(
                job_id,
                status="interrupted",
                phase="已停止",
                message="收到停止请求，导入任务已在安全检查点中断，可稍后恢复。",
                report=counts,
            )
        except Exception as exc:
            logger.exception("Import job failed: %s", job_id)
            counts = self._update_counts(job_id)
            counts["error"] = str(exc)
            self._mark_finished(
                job_id,
                status="failed",
                phase="任务失败",
                message=classify_error(exc, "任务"),
                report=counts,
            )
        finally:
            with _THREAD_LOCK:
                _RUNNING_THREADS.pop(job_id, None)
                _STOP_EVENTS.pop(job_id, None)
                _JOB_SETTINGS.pop(job_id, None)

    def _prepare_plan(self, job_id: str, source_path: str, manager: DocumentManager) -> None:
        """Scan source path, plan retries/skips, and remove deleted documents."""
        self._update_job(job_id, "扫描目录", "正在扫描目录并生成导入计划", source_path)
        files = iter_supported_files(source_path)

        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if job:
                # Reset counters so the UI shows zero until scanning determines the real values.
                job.total_files = len(files)
                job.current_index = 0
                job.imported = 0
                job.updated = 0
                job.skipped = 0
                job.failed = 0
                job.updated_at = utc_now()
            # Reset ALL file rows to pending so this run's counts start from zero.
            # _upsert_file_plan will re-evaluate every file and assign the correct status.
            session.execute(
                update(ImportJobFile)
                .where(ImportJobFile.job_id == job_id)
                .values(
                    status="pending",
                    phase="待重新评估",
                    error_message=None,
                    updated_at=utc_now(),
                )
            )
            # Remove rows for files that are no longer in the supported file list.
            # These are excluded file types (CSV, large TXT, db files, etc.) that
            # should never appear as failures — they are simply not our concern.
            supported_paths = {str(p.resolve()) for p in files}
            session.execute(
                ImportJobFile.__table__.delete().where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.file_path.not_in(supported_paths),
                )
            )

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

        # Reconcile runs AFTER all files are hashed so we can tell moves from deletions.
        deleted_count = self._reconcile_deleted_documents_post_plan(job_id, source_path, manager)
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if job:
                job.deleted = deleted_count
                job.updated_at = utc_now()

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
        stop_event: threading.Event,
    ) -> None:
        """Process one planned file and persist its final state."""
        self._raise_if_cancelled(job_id, stop_event)
        with session_scope(self.settings) as session:
            row = session.get(ImportJobFile, file_row_id)
            if not row:
                return
            file_path = row.file_path
            row.status = "processing"
            row.phase = "开始处理"
            row.message = f"正在处理待办 {sequence}/{total_pending}"
            row.started_at = row.started_at or utc_now()
            row.updated_at = utc_now()

        def progress(
            current: int,
            total: int,
            path: str,
            phase: str,
            message: str,
            level: str = "info",
        ) -> None:
            del current, total
            self._raise_if_cancelled(job_id, stop_event)
            self._update_file_progress(job_id, file_row_id, path, phase, message, level)

        try:
            status = manager.import_file(
                file_path,
                progress_callback=progress,
                cancel_callback=lambda: self._raise_if_cancelled(job_id, stop_event),
                current_index=sequence,
                total_files=total_pending,
            )
            self._mark_file_success(job_id, file_row_id, status)
        except ImportCancelled:
            self._mark_file_interrupted(job_id, file_row_id)
            raise
        except Exception as exc:
            self._mark_file_failed(job_id, file_row_id, exc)
        finally:
            self._update_counts(job_id)

    def _raise_if_cancelled(self, job_id: str, stop_event: threading.Event) -> None:
        """Raise when a running job has been asked to stop."""
        if stop_event.is_set() or self._is_stop_requested(job_id):
            raise ImportCancelled("Import job was cancelled.")

    def _is_stop_requested(self, job_id: str) -> bool:
        """Return whether this job has a persisted stop request."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            return bool(job and job.status == STOP_REQUESTED_STATUS)

    def _upsert_file_plan(self, job_id: str, file_path: Path) -> None:
        """Create or update one file row based on current hash and document state."""
        resolved = str(file_path.resolve())
        try:
            file_hash = calculate_file_md5(file_path)
            info = get_file_info(file_path)
            target_profile = build_index_profile(self.settings, info.file_type)
        except Exception as exc:
            self._upsert_error_file(job_id, resolved, exc)
            return

        with session_scope(self.settings) as session:
            existing_doc = session.execute(
                select(Document).where(Document.file_path == resolved)
            ).scalar_one_or_none()
            reusable_candidates = session.execute(
                select(Document)
                .where(Document.file_hash == file_hash, Document.status == "imported")
                .order_by(Document.imported_at)
            ).scalars().all()
            reusable_doc = next(
                (
                    document
                    for document in reusable_candidates
                    if profile_satisfies(document.index_profile, target_profile)
                ),
                None,
            )
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
            row.updated_at = utc_now()

            if (
                existing_doc
                and existing_doc.file_hash == file_hash
                and existing_doc.status == "imported"
                and profile_satisfies(existing_doc.index_profile, target_profile)
            ):
                row.status = "skipped"
                row.phase = "已存在"
                row.message = "文件未变化，且索引配置已满足目标，跳过"
                row.document_id = existing_doc.id
                row.error_message = None
                row.finished_at = utc_now()
            else:
                row.status = "pending"
                if reusable_doc:
                    row.phase = "等待复用"
                    row.message = "文件哈希已入库，等待复用已有向量"
                    row.document_id = existing_doc.id if existing_doc else reusable_doc.id
                elif existing_doc and existing_doc.file_hash == file_hash:
                    row.phase = "等待重建"
                    row.message = "文件未变化，但索引配置未满足目标，等待重建向量"
                    row.document_id = existing_doc.id
                elif existing_doc and existing_doc.status != "imported":
                    row.phase = "等待重建"
                    row.message = "上次索引未完整完成，等待重建向量"
                    row.document_id = existing_doc.id
                else:
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
            row.updated_at = utc_now()
            row.finished_at = utc_now()
        self._add_event(job_id, "扫描失败", error, file_path, level="error")

    def _reconcile_deleted_documents_post_plan(
        self,
        job_id: str,
        source_path: str,
        manager: DocumentManager,
    ) -> int:
        """区分"文件移动"与"文件真正删除"，在 _upsert_file_plan 全部完成后调用。

        此时 ImportJobFile 表已包含本次扫描到的所有文件的 file_hash。
        通过哈希匹配判断：
          - 旧路径消失但哈希在新路径存在 → 文件移动，原地更新路径，向量保留不变
          - 哈希在当前目录中完全找不到 → 文件真正删除，清理向量
        无论哪种情况均不产生 failed 记录。
        """
        source = Path(source_path)
        if source.is_file():
            return 0
        root = str(source.resolve())
        deleted = 0

        with session_scope(self.settings) as session:
            # 当前任务所有成功计划的文件：hash → 新路径（有 hash 才有意义）
            job_file_rows = session.execute(
                select(ImportJobFile.file_hash, ImportJobFile.file_path).where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.file_hash.is_not(None),
                )
            ).all()
            current_hash_to_path: dict[str, str] = {
                row.file_hash: row.file_path for row in job_file_rows
            }
            current_planned_paths = {row.file_path for row in job_file_rows}

            # 找出路径已不在当前磁盘扫描结果中的旧文档
            documents = session.execute(select(Document)).scalars().all()
            stale = [
                doc for doc in documents
                if _is_under_root(doc.file_path, root)
                and doc.file_path not in current_planned_paths
            ]

            # Pre-build a set of file_paths already occupied by non-stale documents,
            # so we never try to move a stale doc onto a path that is already taken.
            stale_ids = {doc.id for doc in stale}
            occupied_paths: set[str] = {
                d.file_path for d in documents if d.id not in stale_ids
            }

            truly_gone_ids: list[tuple[str, str, str]] = []
            claimed_new_paths: set[str] = set()  # paths claimed by a previous stale doc this loop
            for doc in stale:
                new_path = current_hash_to_path.get(doc.file_hash) if doc.file_hash else None
                if (
                    new_path
                    and new_path not in occupied_paths
                    and new_path not in claimed_new_paths
                ):
                    # 文件已移动：原地更新路径，向量和 document_id 均保留
                    claimed_new_paths.add(new_path)
                    old_path = doc.file_path
                    doc.file_path = new_path
                    doc.filename = Path(new_path).name
                    self._add_event(
                        job_id,
                        "路径更新",
                        f"文件已移动，更新路径：{Path(old_path).name} → {Path(new_path).name}",
                        new_path,
                    )
                else:
                    truly_gone_ids.append((doc.id, doc.file_path, doc.filename))
            # session commit → 路径更新持久化到 SQLite

        for doc_id, file_path, filename in truly_gone_ids:
            manager.delete_document(doc_id)
            deleted += 1
            self._add_event(
                job_id,
                "删除同步",
                f"源文件已不存在，已清理旧向量：{filename}",
                file_path,
            )
        return deleted

    def _get_retryable_files(self, job_id: str) -> list[str]:
        """Return file row IDs that need import or retry, with previously-failed files first."""
        with session_scope(self.settings) as session:
            rows = session.execute(
                select(ImportJobFile).where(
                    ImportJobFile.job_id == job_id,
                    ImportJobFile.status.in_(RETRYABLE_FILE_STATUSES),
                ).order_by(
                    # Previously-failed files (phase = "重新尝试") are prioritized first.
                    case((ImportJobFile.phase == "重新尝试", 0), else_=1),
                    ImportJobFile.file_path,
                )
            ).scalars().all()
            return [row.id for row in rows]

    def _mark_started(self, job_id: str) -> str:
        """Mark a job as running and return its source path."""
        with session_scope(self.settings) as session:
            job = session.get(ImportJob, job_id)
            if not job:
                raise ValueError(f"Import job not found: {job_id}")
            source_path = job.source_path
            now = utc_now()
            session.execute(
                update(ImportJob)
                .where(
                    ImportJob.id == job_id,
                    ImportJob.status != STOP_REQUESTED_STATUS,
                )
                .values(
                    status="running",
                    phase="启动",
                    message="后台导入线程已启动",
                    started_at=job.started_at or now,
                    finished_at=None,
                    updated_at=now,
                    total_files=0,
                    current_index=0,
                    imported=0,
                    updated=0,
                    skipped=0,
                    failed=0,
                    deleted=0,
                    current_file=None,
                )
            )
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    phase="启动",
                    message="后台导入线程已启动或已恢复",
                    file_path=source_path,
                )
            )
            return source_path

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
                job.phase = phase
                job.message = message
                job.current_file = file_path
                job.updated_at = utc_now()
            if row:
                row.status = "processing"
                row.phase = phase
                row.message = message
                row.updated_at = utc_now()
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
            row.updated_at = utc_now()
            row.finished_at = utc_now()
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
            row.updated_at = utc_now()
            row.finished_at = utc_now()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level="error",
                    phase="文件失败",
                    message=f"{Path(row.file_path).name}：{error}",
                    file_path=row.file_path,
                )
            )

    def _mark_file_interrupted(self, job_id: str, file_row_id: str) -> None:
        """Return an interrupted file row to pending so it can be resumed."""
        with session_scope(self.settings) as session:
            row = session.get(ImportJobFile, file_row_id)
            if not row:
                return
            row.status = "pending"
            row.phase = "已停止"
            row.message = "收到停止请求，稍后可恢复处理"
            row.error_message = None
            row.updated_at = utc_now()
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level="info",
                    phase="已停止",
                    message=f"{Path(row.file_path).name}：收到停止请求，稍后可恢复处理",
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
        self._write_session_log(job_id, counts)

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
                job.imported = counts["imported"]
                job.updated = counts["updated"]
                job.skipped = counts["skipped"]
                job.failed = counts["failed"]
                completed = counts["imported"] + counts["updated"] + counts["skipped"] + counts["failed"]
                # current_index tracks progress against the scanned directory total, not row count.
                job.current_index = min(completed, job.total_files) if job.total_files else completed
                job.updated_at = utc_now()
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
            job.phase = phase
            job.message = message
            job.current_file = file_path
            if current_index is not None:
                job.current_index = current_index
            if total_files is not None:
                job.total_files = total_files
            job.updated_at = utc_now()
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
            job.finished_at = utc_now()
            job.updated_at = utc_now()
            job.report_json = json.dumps(report, ensure_ascii=False, default=str)
            # Use 0 as fallback — never preserve a stale value from a previous run.
            job.imported = int(report.get("imported", 0))
            job.updated = int(report.get("updated", 0))
            job.skipped = int(report.get("skipped", 0))
            job.deleted = int(report.get("deleted", 0))
            job.failed = int(report.get("failed", 0))
            # total_files is owned by the directory scan in _prepare_plan; do not overwrite here.
            session.add(
                ImportJobEvent(
                    job_id=job_id,
                    level="error" if status == "failed" else "info",
                    phase=phase,
                    message=message,
                    file_path=job.current_file,
                )
            )

    def _write_session_log(self, job_id: str, counts: dict[str, int]) -> None:
        """Append a one-line summary for this completed job to logs/import_sessions.log."""
        try:
            with session_scope(self.settings) as session:
                job = session.get(ImportJob, job_id)
                if not job:
                    return
                source = job.source_path
                started = job.started_at.strftime("%Y-%m-%d %H:%M:%S") if job.started_at else "-"
                finished = job.finished_at.strftime("%Y-%m-%d %H:%M:%S") if job.finished_at else "-"
                status = job.status

                failed_rows = session.execute(
                    select(ImportJobFile).where(
                        ImportJobFile.job_id == job_id,
                        ImportJobFile.status == "failed",
                    ).order_by(ImportJobFile.file_path)
                ).scalars().all()
                failed_details = [
                    f"    [{Path(row.file_path).name}] {row.file_path}\n      原因: {row.error_message or row.message or '未知'}"
                    for row in failed_rows
                ]

            log_path = Path(self.settings.log_file).parent / "import_sessions.log"
            log_path.parent.mkdir(parents=True, exist_ok=True)
            lines = [
                f"\n{'='*72}",
                f"任务ID   : {job_id}",
                f"目录     : {source}",
                f"开始时间 : {started}",
                f"结束时间 : {finished}",
                f"状态     : {status}",
                f"新增     : {counts.get('imported', 0)}",
                f"更新     : {counts.get('updated', 0)}",
                f"跳过     : {counts.get('skipped', 0)}",
                f"删除     : {counts.get('deleted', 0)}",
                f"失败     : {counts.get('failed', 0)}",
            ]
            if failed_details:
                lines.append("失败文件 :")
                lines.extend(failed_details)
            lines.append(f"{'='*72}")
            with open(log_path, "a", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
        except Exception as exc:
            logger.warning("Failed to write session log: %s", exc)

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
                    ImportJob.status.in_(
                        RESUMABLE_JOB_STATUSES
                    ),
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
    if any(key in lowered for key in ("api key", "unauthorized", "401", "403", "invalid token", "deepseek", "crs")):
        return f"大模型 API token 或鉴权错误：{text}"
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
        "skipped": "文件未变化或哈希已入库，已跳过重复向量化",
    }.get(status, status)


def _is_under_root(file_path: str, root: str) -> bool:
    """Return whether file_path is under root, using resolved paths."""
    try:
        Path(file_path).resolve().relative_to(Path(root).resolve())
        return True
    except ValueError:
        return False


def _persist_stop_request(
    settings: Settings | None = None,
    job_ids: list[str] | None = None,
) -> None:
    """Persist stop requests so workers in other UI sessions can observe them."""
    try:
        runtime_settings = settings or get_settings()
        init_database(runtime_settings)
        with session_scope(runtime_settings) as session:
            statement = select(ImportJob)
            if job_ids is None:
                statement = statement.where(ImportJob.status == "running")
            else:
                statement = statement.where(
                    ImportJob.id.in_(job_ids),
                    ImportJob.status.in_(RUNNING_OR_STOPPING_STATUSES),
                )
            jobs = session.execute(statement).scalars().all()
            now = utc_now()
            for job in jobs:
                if job.status != STOP_REQUESTED_STATUS:
                    job.status = STOP_REQUESTED_STATUS
                    job.phase = "停止请求"
                    job.message = "已收到停止请求，当前批次完成后会中断，可稍后恢复。"
                    job.updated_at = now
                    session.add(
                        ImportJobEvent(
                            job_id=job.id,
                            level="warning",
                            phase="停止请求",
                            message="已收到停止请求，当前批次完成后会中断，可稍后恢复。",
                            file_path=job.current_file,
                        )
                    )
    except Exception as exc:  # pragma: no cover - best-effort shutdown path
        logger.warning("Failed to persist import stop request: %s", exc)


def request_stop_all_running_jobs(settings: Settings | None = None) -> None:
    """Ask all import worker threads to stop without blocking interpreter shutdown."""
    with _THREAD_LOCK:
        job_ids = list(_STOP_EVENTS)
        job_settings = {job_id: _JOB_SETTINGS.get(job_id) for job_id in job_ids}
        for stop_event in _STOP_EVENTS.values():
            stop_event.set()
    if settings is not None:
        _persist_stop_request(settings)
        return
    for job_id, job_settings_item in job_settings.items():
        if job_settings_item is not None:
            _persist_stop_request(job_settings_item, [job_id])


def signal_stop_all_running_jobs_nonblocking(settings: Settings | None = None) -> None:
    """Signal all import workers to stop without blocking on SQLite writes.

    Sets the in-memory stop events immediately (microseconds), then
    attempts DB persistence in a **daemon thread** so that a busy WAL
    lock during Ctrl+C shutdown doesn't freeze the launcher process.

    The in-memory events are what actually stop the workers — the DB
    write is only for cross-process visibility and is redundant on
    single-process shutdown.  On next startup,
    ``_recover_orphaned_running_jobs`` handles any leftover state.
    """
    with _THREAD_LOCK:
        for stop_event in _STOP_EVENTS.values():
            stop_event.set()

    def _persist_in_background() -> None:
        try:
            if settings is not None:
                _persist_stop_request(settings)
            else:
                for job_id in list(_JOB_SETTINGS):
                    job_settings_item = _JOB_SETTINGS.get(job_id)
                    if job_settings_item is not None:
                        _persist_stop_request(job_settings_item, [job_id])
        except Exception:
            pass  # best-effort; recovery handles it on next startup

    import threading as _threading

    bg = _threading.Thread(target=_persist_in_background, daemon=True)
    bg.start()


def _request_stop_all_at_exit() -> None:
    """Stop local worker threads at interpreter exit — signal-only, no DB writes."""
    with _THREAD_LOCK:
        for stop_event in _STOP_EVENTS.values():
            stop_event.set()


atexit.register(_request_stop_all_at_exit)
