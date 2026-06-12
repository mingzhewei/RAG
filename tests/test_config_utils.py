"""Tests for configuration and utility helpers."""

from pathlib import Path

from sensor_vector_db.config.settings import Settings
from sensor_vector_db.utils.file_utils import (
    MAX_TXT_BYTES,
    detect_file_type,
    get_file_exclusion_reason,
    is_supported_file,
    iter_supported_files,
)
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


def test_import_filter_excludes_large_txt_csv_and_database_files(tmp_path: Path) -> None:
    """Import discovery should exclude large TXT, CSV, and database artifacts."""
    small_txt = tmp_path / "sensor.txt"
    limit_txt = tmp_path / "limit.txt"
    large_txt = tmp_path / "large.txt"
    csv_file = tmp_path / "table.csv"
    sqlite_file = tmp_path / "sensor_rag.db"
    sql_dump = tmp_path / "dump.sql"
    chroma_file = tmp_path / "data" / "chroma" / "metadata.json"
    normal_chroma_doc = tmp_path / "manuals" / "chroma" / "sensor.json"

    small_txt.write_text("Model: LDR-100", encoding="utf-8")
    limit_txt.write_bytes(b"x" * MAX_TXT_BYTES)
    large_txt.write_bytes(b"x" * (MAX_TXT_BYTES + 1))
    csv_file.write_text("model,range\nLDR-100,100 m", encoding="utf-8")
    sqlite_file.write_bytes(b"SQLite format 3")
    sql_dump.write_text("CREATE TABLE sensor(id INTEGER);", encoding="utf-8")
    chroma_file.parent.mkdir(parents=True)
    chroma_file.write_text('{"collection": "sensor_documents"}', encoding="utf-8")
    normal_chroma_doc.parent.mkdir(parents=True)
    normal_chroma_doc.write_text('{"model": "LDR-100"}', encoding="utf-8")

    assert is_supported_file(small_txt)
    assert is_supported_file(limit_txt)
    assert not is_supported_file(large_txt)
    assert not is_supported_file(csv_file)
    assert not is_supported_file(sqlite_file)
    assert not is_supported_file(sql_dump)
    assert not is_supported_file(chroma_file)
    assert is_supported_file(normal_chroma_doc)
    assert "larger" in (get_file_exclusion_reason(large_txt) or "")
    assert set(iter_supported_files(tmp_path)) == {limit_txt, normal_chroma_doc, small_txt}
