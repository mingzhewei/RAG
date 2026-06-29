"""Lightweight integration test for import and hybrid search."""

from pathlib import Path

import pytest
from sqlalchemy import select

from sensor_vector_db.core.document_manager import DocumentManager
from sensor_vector_db.core.embedding import DeterministicEmbedding
from sensor_vector_db.core.index_profile import build_index_profile, profile_satisfies
from sensor_vector_db.core.search_engine import SearchEngine
from sensor_vector_db.core.vector_store import VectorStore, _build_where_clause
from sensor_vector_db.models.database import Document, DocumentChunk, session_scope
from sensor_vector_db.utils.file_utils import get_file_info
from sensor_vector_db.utils.hash_utils import calculate_file_md5


def test_build_where_clause_handles_filter_counts() -> None:
    """Empty, single, and multi-key filters must match ChromaDB's where rules."""
    assert _build_where_clause(None) is None
    assert _build_where_clause({}) is None
    assert _build_where_clause({"file_type": "pdf"}) == {"file_type": "pdf"}
    assert _build_where_clause({"file_type": "pdf", "sensor_model": "X"}) == {
        "$and": [{"file_type": "pdf"}, {"sensor_model": "X"}]
    }


def test_missing_profile_does_not_satisfy_pdf_ocr_target(test_settings) -> None:
    """Legacy missing profiles should be rebuilt only for PDF OCR targets."""
    ocr_settings = test_settings.model_copy(update={"ocr_enabled": True})

    assert not profile_satisfies(None, build_index_profile(ocr_settings, "pdf"))
    assert profile_satisfies(None, build_index_profile(ocr_settings, "text"))
    assert profile_satisfies(None, build_index_profile(test_settings, "pdf"))


def test_direct_file_import_rejects_excluded_csv(test_settings, tmp_path: Path) -> None:
    """The single-file import path must enforce the same CSV exclusion rule."""
    source = tmp_path / "sensor_table.csv"
    source.write_text("model,range\nLDR-100,100 m", encoding="utf-8")

    manager = DocumentManager(test_settings)

    with pytest.raises(ValueError, match="CSV files are excluded"):
        manager.import_file(source)


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


class RecordingCollection:
    """Small fake Chroma collection that records batch sizes."""

    def __init__(self) -> None:
        self.upsert_sizes: list[int] = []
        self.update_sizes: list[int] = []

    def upsert(self, *, ids, documents, embeddings, metadatas) -> None:
        assert len(ids) == len(documents) == len(embeddings) == len(metadatas)
        self.upsert_sizes.append(len(ids))

    def update(self, *, ids, metadatas) -> None:
        assert len(ids) == len(metadatas)
        self.update_sizes.append(len(ids))


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
    assert vector_store.count() == 1


def test_vector_store_splits_large_chroma_batches(test_settings) -> None:
    """Large Chroma mutations must be split to the client batch limit."""
    vector_store = VectorStore(test_settings)
    vector_store.collection = RecordingCollection()
    vector_store.client.get_max_batch_size = lambda: 3

    chunk_ids = [f"id-{index}" for index in range(8)]
    documents = [f"doc-{index}" for index in range(8)]
    embeddings = [[float(index)] * 4 for index in range(8)]
    metadatas = [{"file_path": f"/tmp/{index}.txt", "chunk_index": index} for index in range(8)]

    vector_store.add_chunks(chunk_ids, documents, embeddings, metadatas)
    vector_store.update_metadata(chunk_ids, metadatas)

    assert vector_store.collection.upsert_sizes == [3, 3, 2]
    assert vector_store.collection.update_sizes == [3, 3, 2]


def test_incomplete_indexing_document_is_recovered(test_settings, tmp_path: Path) -> None:
    """A half-written document should be recovered after a hard process kill."""
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


def test_incomplete_indexing_document_resumes_missing_vectors(
    test_settings, tmp_path: Path
) -> None:
    """A file-level checkpoint should resume only Chroma chunks that are missing."""
    source = tmp_path / "lidar.txt"
    first_text = "Model: LDR-100\nRange: 100 m"
    second_text = "Accuracy: 5 cm\nInterface: RS485"
    source.write_text(f"{first_text}\n\n{second_text}", encoding="utf-8")
    file_info = get_file_info(source)
    file_hash = calculate_file_md5(source)
    embedding = CountingEmbedding(dimension=64)
    vector_store = VectorStore(test_settings)

    # Create DocumentManager *before* the test document so that
    # cleanup_orphan_indexing_documents() does not delete it.
    manager = DocumentManager(test_settings, embedding=embedding, vector_store=vector_store)

    with session_scope(test_settings) as session:
        document = Document(
            file_path=str(source.resolve()),
            filename=source.name,
            file_type=file_info.file_type,
            file_hash=file_hash,
            size_bytes=file_info.size_bytes,
            created_at=file_info.created_at,
            modified_at=file_info.modified_at,
            status="indexing",
            sensor_model="LDR-100",
            index_profile=build_index_profile(test_settings),
        )
        session.add(document)
        session.flush()
        first_chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=0,
            content=first_text,
            content_type="text",
            source_label=f"{source.name} #0",
        )
        second_chunk = DocumentChunk(
            document_id=document.id,
            chunk_index=1,
            content=second_text,
            content_type="text",
            source_label=f"{source.name} #1",
        )
        session.add_all([first_chunk, second_chunk])
        session.flush()
        document_id = document.id
        first_chunk_id = first_chunk.id
        second_chunk_id = second_chunk.id

    first_embedding = embedding.embed_texts([first_text])[0]
    vector_store.add_chunks(
        [first_chunk_id],
        [first_text],
        [first_embedding],
        [
            {
                "document_id": document_id,
                "source_label": f"{source.name} #0",
                "file_path": str(source.resolve()),
                "file_type": file_info.file_type,
                "file_hash": file_hash,
                "filename": source.name,
                "chunk_index": 0,
                "content_type": "text",
                "sensor_model": "LDR-100",
            }
        ],
    )
    embedding.calls = 0
    embedding.text_count = 0

    def fail_parse(*args, **kwargs):
        raise AssertionError("checkpoint resume should not reparse the file")

    manager.parser_factory.parse = fail_parse

    assert manager.import_file(source) == "updated"
    assert embedding.calls == 1
    assert embedding.text_count == 1
    assert vector_store.get_existing_ids([first_chunk_id, second_chunk_id]) == {
        first_chunk_id,
        second_chunk_id,
    }
    with session_scope(test_settings) as session:
        document = session.get(Document, document_id)
        assert document is not None
        assert document.status == "imported"


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


def test_text_file_is_not_rebuilt_when_ocr_profile_is_requested(
    test_settings, tmp_path: Path
) -> None:
    """OCR settings should not force non-PDF documents to be rebuilt."""
    source = tmp_path / "lidar.txt"
    source.write_text("Model: LDR-100\nRange: 100 m", encoding="utf-8")

    manager = DocumentManager(test_settings)
    assert manager.import_file(source) == "imported"

    with session_scope(test_settings) as session:
        first_document = session.execute(select(Document)).scalar_one()
        assert '"mode":"text"' in (first_document.index_profile or "")

    ocr_settings = test_settings.model_copy(update={"ocr_enabled": True})
    ocr_manager = DocumentManager(ocr_settings)
    assert ocr_manager.import_file(source) == "skipped"

    with session_scope(ocr_settings) as session:
        document = session.execute(select(Document)).scalar_one()
        assert '"mode":"text"' in (document.index_profile or "")


def test_pdf_rebuilds_when_ocr_profile_is_requested(test_settings, tmp_path: Path) -> None:
    """A text-only PDF index should be rebuilt when OCR indexing is requested."""
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    source = tmp_path / "lidar.pdf"
    canvas = reportlab.Canvas(str(source))
    canvas.drawString(72, 720, "Model: LDR-100")
    canvas.drawString(72, 700, "Range: 100 m")
    canvas.save()

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
