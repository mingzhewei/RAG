"""Tests for persistent resumable import jobs."""

from pathlib import Path
import threading
import time

from sensor_vector_db.core import import_jobs as import_jobs_module
from sensor_vector_db.core.import_jobs import (
    STOP_REQUESTED_STATUS,
    ImportJobManager,
    classify_error,
)
from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.models.database import ImportJob, ImportJobFile, session_scope


def wait_for_job(manager: ImportJobManager, job_id: str, timeout: float = 20.0):
    """Wait until a background import job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job and job.status in {"completed", "completed_with_errors", "failed", "interrupted"}:
            return job
        time.sleep(0.2)
    raise AssertionError(f"Import job did not finish: {job_id}")


def test_import_job_sync_skip_and_delete(test_settings, tmp_path: Path) -> None:
    """Import jobs should persist state, skip unchanged files, and clean deleted files."""
    source_dir = tmp_path / "sensor"
    source_dir.mkdir()
    source_file = source_dir / "sensor.txt"
    source_file.write_text(
        "型号: LDR-100\n厂商: ACME Sensors\n测距范围: 100 m",
        encoding="utf-8",
    )

    manager = ImportJobManager(test_settings)
    first_job_id = manager.start_import(source_dir)
    first = wait_for_job(manager, first_job_id)
    assert first.status == "completed"
    assert first.imported == 1
    assert DocumentManager(test_settings).stats()["documents"] == 1

    second_job_id = manager.start_import(source_dir)
    second = wait_for_job(manager, second_job_id)
    assert second.status == "completed"
    assert second.skipped == 1

    source_file.unlink()
    third_job_id = manager.start_import(source_dir)
    third = wait_for_job(manager, third_job_id)
    assert third.status == "completed"
    assert third.deleted == 1
    assert DocumentManager(test_settings).stats()["documents"] == 0


def test_import_job_reuses_hash_for_moved_directory(test_settings, tmp_path: Path) -> None:
    """Importing the same file content from another directory should reuse indexed vectors."""
    first_dir = tmp_path / "original"
    moved_dir = tmp_path / "moved"
    first_dir.mkdir()
    moved_dir.mkdir()
    content = "Model: LDR-100\nManufacturer: ACME Sensors\nRange: 100 m"
    (first_dir / "sensor.txt").write_text(content, encoding="utf-8")
    (moved_dir / "sensor.txt").write_text(content, encoding="utf-8")

    manager = ImportJobManager(test_settings)
    first_job_id = manager.start_import(first_dir)
    first = wait_for_job(manager, first_job_id)
    assert first.status == "completed"
    assert first.imported == 1

    moved_job_id = manager.start_import(moved_dir)
    moved = wait_for_job(manager, moved_job_id)
    assert moved.status == "completed"
    assert moved.skipped == 1

    stats = DocumentManager(test_settings).stats()
    assert stats["documents"] == 2
    assert stats["chunks"] == 2
    assert stats["vectors"] == 2
    events = manager.get_events(moved_job_id)
    assert any(event["phase"] == "复用文件哈希" for event in events)


def test_import_job_can_be_interrupted(test_settings, tmp_path: Path, monkeypatch) -> None:
    """A running import job should honor stop requests at progress checkpoints."""
    source_dir = tmp_path / "sensor"
    source_dir.mkdir()
    (source_dir / "sensor.txt").write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")

    class SlowDocumentManager:
        def __init__(self, settings):
            self.settings = settings

        def delete_document(self, document_id: str) -> None:
            return None

        def import_file(
            self,
            file_path,
            progress_callback=None,
            cancel_callback=None,
            current_index=1,
            total_files=1,
        ):
            while True:
                if cancel_callback:
                    cancel_callback()
                if progress_callback:
                    progress_callback(
                        current_index,
                        total_files,
                        str(file_path),
                        "测试慢导入",
                        "等待停止请求",
                    )
                time.sleep(0.05)

    monkeypatch.setattr(import_jobs_module, "DocumentManager", SlowDocumentManager)
    manager = ImportJobManager(test_settings)
    job_id = manager.start_import(source_dir)
    deadline = time.time() + 5
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job and job.is_thread_active:
            break
        time.sleep(0.05)
    manager.request_stop(job_id)

    interrupted = wait_for_job(manager, job_id)
    assert interrupted.status == "interrupted"
    assert interrupted.can_resume


def test_import_job_honors_persisted_stop_request(test_settings, tmp_path: Path, monkeypatch) -> None:
    """A worker should stop even when the stop request arrives via SQLite only."""
    source_dir = tmp_path / "sensor"
    source_dir.mkdir()
    (source_dir / "sensor.txt").write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")
    entered_import = threading.Event()

    class SlowDocumentManager:
        def __init__(self, settings):
            self.settings = settings

        def delete_document(self, document_id: str) -> None:
            return None

        def import_file(
            self,
            file_path,
            progress_callback=None,
            cancel_callback=None,
            current_index=1,
            total_files=1,
        ):
            entered_import.set()
            while True:
                if cancel_callback:
                    cancel_callback()
                time.sleep(0.05)

    monkeypatch.setattr(import_jobs_module, "DocumentManager", SlowDocumentManager)
    manager = ImportJobManager(test_settings)
    job_id = manager.start_import(source_dir)
    deadline = time.time() + 5
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job and job.is_thread_active:
            break
        time.sleep(0.05)
    assert entered_import.wait(timeout=5)

    with session_scope(test_settings) as session:
        job = session.get(ImportJob, job_id)
        assert job is not None
        job.status = STOP_REQUESTED_STATUS
        job.phase = "停止请求"
        job.message = "测试持久化停止请求"
        session.add(job)

    interrupted = wait_for_job(manager, job_id)
    assert interrupted.status == "interrupted"
    assert interrupted.can_resume


def test_progress_updates_do_not_clear_stop_request(test_settings, tmp_path: Path) -> None:
    """Progress writes must not overwrite a persisted stop request."""
    source_dir = tmp_path / "sensor"
    source_dir.mkdir()
    source_file = source_dir / "sensor.txt"
    source_file.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")
    manager = ImportJobManager(test_settings)

    with session_scope(test_settings) as session:
        job = ImportJob(
            source_path=str(source_dir.resolve()),
            status=STOP_REQUESTED_STATUS,
            phase="停止请求",
            message="用户要求停止",
            current_file=str(source_file.resolve()),
            total_files=1,
        )
        session.add(job)
        session.flush()
        job_id = job.id
        row = ImportJobFile(
            job_id=job_id,
            file_path=str(source_file.resolve()),
            status="processing",
            phase="向量化",
            message="正在生成向量",
        )
        session.add(row)
        session.flush()
        row_id = row.id

    manager._update_file_progress(
        job_id,
        row_id,
        str(source_file.resolve()),
        "写入向量库",
        "正在写入 ChromaDB batch",
        "info",
    )

    with session_scope(test_settings) as session:
        job = session.get(ImportJob, job_id)
        assert job is not None
        assert job.status == STOP_REQUESTED_STATUS
        assert job.phase == "写入向量库"


def test_atexit_without_local_workers_does_not_persist_stop(monkeypatch) -> None:
    """Interpreter exit should not touch unrelated persisted jobs."""
    calls = []

    def fake_persist(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(import_jobs_module, "_persist_stop_request", fake_persist)
    with import_jobs_module._THREAD_LOCK:
        old_events = dict(import_jobs_module._STOP_EVENTS)
        import_jobs_module._STOP_EVENTS.clear()
    try:
        import_jobs_module._request_stop_all_at_exit()
    finally:
        with import_jobs_module._THREAD_LOCK:
            import_jobs_module._STOP_EVENTS.clear()
            import_jobs_module._STOP_EVENTS.update(old_events)

    assert calls == []


def test_atexit_with_local_workers_uses_worker_settings(monkeypatch, test_settings) -> None:
    """Interpreter exit should persist stop requests only for local workers."""
    calls = []
    stop_event = threading.Event()

    def fake_persist(*args, **kwargs):
        calls.append((args, kwargs))

    monkeypatch.setattr(import_jobs_module, "_persist_stop_request", fake_persist)
    with import_jobs_module._THREAD_LOCK:
        old_events = dict(import_jobs_module._STOP_EVENTS)
        old_settings = dict(import_jobs_module._JOB_SETTINGS)
        import_jobs_module._STOP_EVENTS.clear()
        import_jobs_module._JOB_SETTINGS.clear()
        import_jobs_module._STOP_EVENTS["job-1"] = stop_event
        import_jobs_module._JOB_SETTINGS["job-1"] = test_settings
    try:
        import_jobs_module._request_stop_all_at_exit()
    finally:
        with import_jobs_module._THREAD_LOCK:
            import_jobs_module._STOP_EVENTS.clear()
            import_jobs_module._STOP_EVENTS.update(old_events)
            import_jobs_module._JOB_SETTINGS.clear()
            import_jobs_module._JOB_SETTINGS.update(old_settings)

    assert stop_event.is_set()
    assert calls == [((test_settings, ["job-1"]), {})]


def test_manager_recovers_orphaned_running_job(test_settings, tmp_path: Path) -> None:
    """A persisted running job without a live worker should become resumable."""
    source_dir = tmp_path / "sensor"
    source_dir.mkdir()
    source_file = source_dir / "sensor.txt"
    source_file.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")

    with session_scope(test_settings) as session:
        job = ImportJob(
            source_path=str(source_dir.resolve()),
            status="running",
            phase="解析文档",
            message="正在解析正文、表格或 OCR 文本",
            current_file=str(source_file.resolve()),
            total_files=1,
        )
        session.add(job)
        session.flush()
        job_id = job.id
        session.add(
            ImportJobFile(
                job_id=job_id,
                file_path=str(source_file.resolve()),
                status="processing",
                phase="解析文档",
                message="正在解析正文、表格或 OCR 文本",
            )
        )

    manager = ImportJobManager(test_settings)
    recovered = manager.get_job(job_id)
    assert recovered is not None
    assert recovered.status == "interrupted"
    assert recovered.can_resume

    rows = manager.get_file_rows(job_id)
    assert rows[0]["status"] == "pending"
    assert rows[0]["phase"] == "恢复排队"
    assert any(event["phase"] == "上次运行已中断" for event in manager.get_events(job_id))


def test_error_classification_for_user_messages() -> None:
    """Known failure classes should produce actionable user-facing messages."""
    assert "大模型 API token" in classify_error(RuntimeError("401 invalid token"))
    assert "服务器连接" in classify_error(RuntimeError("connection timeout"))
    assert "向量化失败" in classify_error(RuntimeError("BGE embedding failed"))
