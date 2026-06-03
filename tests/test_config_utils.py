"""Tests for configuration and utility helpers."""

from pathlib import Path

from sensor_vector_db.config.settings import Settings
from sensor_vector_db.utils.file_utils import detect_file_type, iter_supported_files
from sensor_vector_db.utils.hash_utils import calculate_file_md5


def test_settings_create_directories(tmp_path: Path) -> None:
    """Settings should create runtime directories."""
    settings = Settings(
        chroma_path=tmp_path / "data" / "chroma",
        sqlite_path=tmp_path / "data" / "db.sqlite",
        log_file=tmp_path / "logs" / "app.log",
    )
    settings.ensure_directories()
    assert settings.chroma_path.exists()
    assert settings.sqlite_path.parent.exists()
    assert settings.log_file.parent.exists()


def test_file_type_and_hash(tmp_path: Path) -> None:
    """Supported file types and MD5 hashes should be deterministic."""
    txt = tmp_path / "sensor.txt"
    txt.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")
    assert detect_file_type(txt) == "text"
    assert calculate_file_md5(txt) == calculate_file_md5(txt)
    assert iter_supported_files(tmp_path) == [txt]

