"""Tests for persistent resumable import jobs."""

from pathlib import Path
import time

from sensor_vector_db.core.import_jobs import ImportJobManager, classify_error
from sensor_vector_db.core.document_manager import DocumentManager


def wait_for_job(manager: ImportJobManager, job_id: str, timeout: float = 20.0):
    """Wait until a background import job reaches a terminal state."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = manager.get_job(job_id)
        if job and job.status in {"completed", "completed_with_errors", "failed"}:
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


def test_error_classification_for_user_messages() -> None:
    """Known failure classes should produce actionable user-facing messages."""
    assert "DeepSeek token" in classify_error(RuntimeError("401 invalid token"))
    assert "服务器连接" in classify_error(RuntimeError("connection timeout"))
    assert "向量化失败" in classify_error(RuntimeError("BGE embedding failed"))

