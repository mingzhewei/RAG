"""Pytest fixtures for the local sensor RAG system."""

from pathlib import Path

import pytest

from sensor_vector_db.config.settings import Settings
from sensor_vector_db.models.database import init_database


@pytest.fixture()
def test_settings(tmp_path: Path) -> Settings:
    """Return isolated settings for tests."""
    settings = Settings(
        embedding_backend="fake",
        embedding_dimension=64,
        ocr_enabled=False,
        chunk_size=48,
        chunk_overlap=8,
        chroma_path=tmp_path / "chroma",
        sqlite_path=tmp_path / "sensor_rag.db",
        log_file=tmp_path / "sensor_rag.log",
        chroma_collection="test_collection",
    )
    settings.ensure_directories()
    init_database(settings)
    return settings

