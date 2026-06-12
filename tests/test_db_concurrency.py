"""Concurrency regression test for the SQLite WAL configuration.

Reproduces the exact "database is locked" failure: a background writer thread
hammering import_jobs while several reader threads poll job status, mirroring
the import worker plus the Streamlit `run_every` fragment.
"""

from __future__ import annotations

from pathlib import Path
import threading
import time

from sqlalchemy import select, text

from sensor_vector_db.config.settings import Settings
from sensor_vector_db.models.database import (
    ImportJob,
    ImportJobEvent,
    get_engine,
    init_database,
    session_scope,
    utc_now,
)


def _make_settings(tmp_path: Path) -> Settings:
    """Return isolated settings backed by a temp SQLite database."""
    settings = Settings(
        embedding_backend="fake",
        embedding_dimension=64,
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "sensor_rag.db",
        log_file=tmp_path / "log.log",
        chroma_collection="test_concurrency",
    )
    settings.ensure_directories()
    init_database(settings)
    return settings


def test_wal_mode_is_set_on_new_database(tmp_path: Path) -> None:
    """A freshly created database must use WAL journal mode."""
    settings = _make_settings(tmp_path)
    engine = get_engine(settings)
    with engine.connect() as conn:
        mode = conn.execute(text("PRAGMA journal_mode")).scalar()
        timeout = conn.execute(text("PRAGMA busy_timeout")).scalar()
    assert mode == "wal", f"Expected WAL, got {mode!r}"
    assert int(timeout) >= 10000, f"Expected busy_timeout >= 10000ms, got {timeout}"


def test_concurrent_read_write_does_not_lock(tmp_path: Path) -> None:
    """A writer thread and several reader threads must not deadlock or lock.

    Without WAL mode the default journal lock blocks all readers while the
    writer holds a transaction. This test uses a 100 ms sleep inside the
    write transaction to guarantee overlap with reader polls, making the
    failure deterministic on non-WAL databases.
    """
    settings = _make_settings(tmp_path)
    with session_scope(settings) as session:
        job = ImportJob(
            source_path="x",
            status="running",
            phase="p",
            message="m",
            current_file="x",
        )
        session.add(job)
        session.flush()
        job_id = job.id

    errors: list[Exception] = []
    stop = threading.Event()

    def writer() -> None:
        try:
            for index in range(60):
                with session_scope(settings) as session:
                    row = session.get(ImportJob, job_id)
                    row.imported = index
                    row.updated_at = utc_now()
                    session.add(
                        ImportJobEvent(job_id=job_id, phase="p", message=f"e{index}")
                    )
                    # Hold the write transaction open briefly to force overlap
                    time.sleep(0.02)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)
        finally:
            stop.set()

    def reader() -> None:
        try:
            while not stop.is_set():
                with session_scope(settings) as session:
                    session.get(ImportJob, job_id)
                    list(
                        session.execute(
                            select(ImportJobEvent).where(
                                ImportJobEvent.job_id == job_id
                            )
                        ).scalars()
                    )
                time.sleep(0.03)
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    writer_thread = threading.Thread(target=writer)
    reader_threads = [threading.Thread(target=reader) for _ in range(4)]
    for thread in reader_threads:
        thread.start()
    writer_thread.start()
    writer_thread.join(timeout=30)
    stop.set()
    for thread in reader_threads:
        thread.join(timeout=5)

    assert not errors, f"Concurrent access raised: {errors[:3]}"

    with session_scope(settings) as session:
        final = session.get(ImportJob, job_id)
        assert final.imported == 59
