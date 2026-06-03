"""Lightweight integration test for import and hybrid search."""

from pathlib import Path

import pytest
from sqlalchemy import select

from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.core.embedding import DeterministicEmbedding
from sensor_vector_db.core.search_engine import SearchEngine
from sensor_vector_db.core.vector_store import VectorStore
from sensor_vector_db.models.database import Document, DocumentChunk, session_scope


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


class FailingAddVectorStore:
    """Vector store wrapper that fails only when adding new chunks."""

    def __init__(self, backing: VectorStore) -> None:
        self.backing = backing

    def add_chunks(self, *args, **kwargs) -> None:
        raise RuntimeError("simulated vector write failure")

    def delete_by_document_id(self, document_id: str) -> None:
        self.backing.delete_by_document_id(document_id)

    def count(self) -> int:
        return self.backing.count()


class CountingEmbedding(DeterministicEmbedding):
    """Deterministic embedding provider that records expensive embedding calls."""

    def __init__(self, dimension: int = 64) -> None:
        super().__init__(dimension=dimension)
        self.calls = 0
        self.text_count = 0

    def embed_texts(self, texts: list[str]) -> list[list[float]]:
        self.calls += 1
        self.text_count += len(texts)
        return super().embed_texts(texts)


def test_failed_update_restores_previous_document(test_settings, tmp_path: Path) -> None:
    """A failed update should not remove the previously indexed evidence."""
    source = tmp_path / "lidar.txt"
    source.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")
    embedding = DeterministicEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)
    assert manager.import_file(source) == "imported"

    source.write_text("Model: LDR-200\nRange: 200 m", encoding="utf-8")
    failing_manager = DocumentManager(
        test_settings,
        embedding=embedding,
        vector_store=FailingAddVectorStore(vector_store),
    )
    with pytest.raises(RuntimeError, match="simulated vector write failure"):
        failing_manager.import_file(source)

    with session_scope(test_settings) as session:
        documents = session.execute(select(Document)).scalars().all()
        chunks = session.execute(select(DocumentChunk)).scalars().all()

    assert len(documents) == 1
    assert documents[0].file_hash != ""
    assert any("LDR-100" in chunk.content for chunk in chunks)
    assert not any("LDR-200" in chunk.content for chunk in chunks)


def test_duplicate_hash_reuses_existing_vectors(test_settings, tmp_path: Path) -> None:
    """The same file content at a new path should reuse chunks and embeddings."""
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_dir.mkdir()
    second_dir.mkdir()
    content = "Model: LDR-100\nManufacturer: ACME Sensors\nRange: 100 m"
    first = first_dir / "sensor.txt"
    duplicate = second_dir / "renamed_sensor.txt"
    first.write_text(content, encoding="utf-8")
    duplicate.write_text(content, encoding="utf-8")

    embedding = CountingEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)

    assert manager.import_file(first) == "imported"
    first_call_count = embedding.calls
    first_text_count = embedding.text_count
    assert first_call_count == 1

    assert manager.import_file(duplicate) == "skipped"
    assert embedding.calls == first_call_count
    assert embedding.text_count == first_text_count

    with session_scope(test_settings) as session:
        documents = session.execute(select(Document).order_by(Document.file_path)).scalars().all()
        chunks = session.execute(select(DocumentChunk).order_by(DocumentChunk.source_label)).scalars().all()

    assert len(documents) == 2
    assert {document.file_hash for document in documents} == {documents[0].file_hash}
    assert any(document.file_path == str(duplicate.resolve()) for document in documents)
    assert len(chunks) == 2
    assert vector_store.count() == 2

    search = SearchEngine(test_settings, embedding=embedding, vector_store=vector_store)
    results = search.search("LDR-100 100 m", mode="semantic", top_k=5)
    assert any(result.file_path == str(duplicate.resolve()) for result in results)
