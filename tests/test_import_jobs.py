"""Tests for persistent resumable import jobs."""

from pathlib import Path
import time

from sensor_vector_db.core import import_jobs as import_jobs_module
from sensor_vector_db.core.import_jobs import ImportJobManager, classify_error
from sensor_vector_db.core.document_manager import DocumentManager


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

        def import_file(self, file_path, progress_callback=None, current_index=1, total_files=1):
            while True:
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


def test_error_classification_for_user_messages() -> None:
    """Known failure classes should produce actionable user-facing messages."""
    assert "DeepSeek token" in classify_error(RuntimeError("401 invalid token"))
    assert "服务器连接" in classify_error(RuntimeError("connection timeout"))
    assert "向量化失败" in classify_error(RuntimeError("BGE embedding failed"))
