"""Lightweight integration test for import and hybrid search."""

from pathlib import Path

from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.core.embedding import DeterministicEmbedding
from sensor_vector_db.core.search_engine import SearchEngine
from sensor_vector_db.core.vector_store import VectorStore


def test_import_and_hybrid_search(test_settings, tmp_path: Path) -> None:
    """A text file should import into SQLite/Chroma and be searchable."""
    source = tmp_path / "lidar.txt"
    source.write_text(
        "型号: LDR-100\n厂商: ACME Sensors\n测距范围: 100 m\n精度: ±2 cm",
        encoding="utf-8",
    )
    embedding = DeterministicEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)
    report = manager.import_path(source)
    assert report.imported == 1
    assert manager.stats()["chunks"] >= 1

    search = SearchEngine(test_settings, embedding=embedding, vector_store=vector_store)
    results = search.search("LDR-100 100 m", mode="hybrid", top_k=3)
    assert results
    assert "LDR-100" in results[0].content

