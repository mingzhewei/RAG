"""Lightweight integration test for import and hybrid search."""

from pathlib import Path

import pytest
from sqlalchemy import select

from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.core.embedding import DeterministicEmbedding
from sensor_vector_db.core.search_engine import SearchEngine
from sensor_vector_db.core.vector_store import VectorStore, _build_where_clause
from sensor_vector_db.models.database import Document, DocumentChunk, session_scope


def test_build_where_clause_handles_filter_counts() -> None:
    """Empty, single, and multi-key filters must match ChromaDB's where rules."""
    assert _build_where_clause(None) is None
    assert _build_where_clause({}) is None
    assert _build_where_clause({"file_type": "pdf"}) == {"file_type": "pdf"}
    assert _build_where_clause({"file_type": "pdf", "sensor_model": "X"}) == {
        "$and": [{"file_type": "pdf"}, {"sensor_model": "X"}]
    }


def test_semantic_search_with_two_filters(test_settings, tmp_path: Path) -> None:
    """Combining file_type and sensor_model filters must not raise in ChromaDB."""
    source = tmp_path / "lidar.txt"
    source.write_text("型号: LDR-100\n测距范围: 100 m", encoding="utf-8")
    embedding = DeterministicEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)
    assert manager.import_file(source) == "imported"

    search = SearchEngine(test_settings, embedding=embedding, vector_store=vector_store)
    results = search.search(
        "LDR-100 100 m",
        mode="semantic",
        top_k=5,
        filters={"file_type": "text", "sensor_model": "LDR-100"},
    )
    assert results
    assert all(result.file_type == "text" for result in results)


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


def test_incomplete_indexing_document_is_rebuilt(test_settings, tmp_path: Path) -> None:
    """A half-written document should not be skipped after a hard process kill."""
    source = tmp_path / "lidar.txt"
    source.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")
    embedding = DeterministicEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)
    assert manager.import_file(source) == "imported"

    with session_scope(test_settings) as session:
        document = session.execute(select(Document)).scalar_one()
        document.status = "indexing"

    assert manager.stats()["documents"] == 0
    assert manager.import_file(source) == "updated"
    assert manager.stats()["documents"] == 1
    assert manager.stats()["vectors"] == 1

    search = SearchEngine(test_settings, embedding=embedding, vector_store=vector_store)
    results = search.search("LDR-100", mode="hybrid", top_k=3)
    assert results


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


def test_refresh_sensor_models_updates_metadata_without_reembedding(
    test_settings, tmp_path: Path
) -> None:
    """Refreshing hints must fix metadata without recomputing any embedding."""
    source = tmp_path / "lidar.txt"
    source.write_text("型号: LDR-100\n测距范围: 100 m", encoding="utf-8")
    embedding = CountingEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)
    assert manager.import_file(source) == "imported"
    calls_after_import = embedding.calls

    # Simulate a previously polluted model value from the old extraction rules.
    with session_scope(test_settings) as session:
        document = session.execute(select(Document)).scalar_one()
        document.sensor_model = "ISO9001"
        document_id = document.id

    result = manager.refresh_sensor_models(overwrite=True)
    assert result == {"scanned": 1, "updated": 1}
    # No new embedding work was done.
    assert embedding.calls == calls_after_import

    with session_scope(test_settings) as session:
        refreshed = session.get(Document, document_id)
        assert refreshed.sensor_model == "LDR-100"

    # Chroma chunk metadata reflects the corrected model.
    stored = vector_store.collection.get(
        where={"document_id": document_id}, include=["metadatas"]
    )
    assert stored["metadatas"]
    assert all(item["sensor_model"] == "LDR-100" for item in stored["metadatas"])

    # Filtering by the corrected model now returns the document.
    search = SearchEngine(test_settings, embedding=embedding, vector_store=vector_store)
    results = search.search(
        "LDR-100 100 m",
        mode="semantic",
        top_k=5,
        filters={"sensor_model": "LDR-100"},
    )
    assert results


def test_refresh_sensor_models_can_preserve_existing_values(
    test_settings, tmp_path: Path
) -> None:
    """With overwrite disabled, an existing model value is left untouched."""
    source = tmp_path / "lidar.txt"
    source.write_text("型号: LDR-100\n测距范围: 100 m", encoding="utf-8")
    manager = DocumentManager(test_settings)
    assert manager.import_file(source) == "imported"

    with session_scope(test_settings) as session:
        document = session.execute(select(Document)).scalar_one()
        document.sensor_model = "CUSTOM-TAG"

    result = manager.refresh_sensor_models(overwrite=False)
    assert result == {"scanned": 1, "updated": 0}
    with session_scope(test_settings) as session:
        document = session.execute(select(Document)).scalar_one()
        assert document.sensor_model == "CUSTOM-TAG"


def test_same_hash_rebuilds_when_ocr_profile_is_requested(test_settings, tmp_path: Path) -> None:
    """A text-only index should be rebuilt when OCR indexing is later requested."""
    source = tmp_path / "lidar.txt"
    source.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")

    manager = DocumentManager(test_settings)
    assert manager.import_file(source) == "imported"

    with session_scope(test_settings) as session:
        first_document = session.execute(select(Document)).scalar_one()
        assert '"mode":"text"' in (first_document.index_profile or "")

    ocr_settings = test_settings.model_copy(update={"ocr_enabled": True})
    ocr_manager = DocumentManager(ocr_settings)
    assert ocr_manager.import_file(source) == "updated"

    with session_scope(ocr_settings) as session:
        rebuilt_document = session.execute(select(Document)).scalar_one()
        assert '"mode":"ocr"' in (rebuilt_document.index_profile or "")

    text_manager = DocumentManager(test_settings)
    assert text_manager.import_file(source) == "skipped"
