"""Tests for strict parameter extraction and comparison."""

from sensor_vector_db.core.parameter_extractor import (
    ComparisonCell,
    ParameterComparer,
    ParameterExtractor,
)
from sensor_vector_db.models.database import (
    Document,
    DocumentChunk,
    ExtractedParameter,
    session_scope,
)


def test_rule_extraction_from_text() -> None:
    """Rules should extract only explicit key-value evidence."""
    chunk = DocumentChunk(
        document_id="doc",
        chunk_index=0,
        content="型号: LDR-100\n测距范围: 100 m\n这句话没有参数。",
        content_type="text",
        source_label="sample p.1",
        page_number=1,
    )
    parameters = ParameterExtractor().extract_from_chunks([chunk])
    values = {item.normalized_name: item.value for item in parameters}
    assert values["model"] == "LDR-100"
    assert values["range"] == "100"
    assert "accuracy" not in values


def test_comparison_marks_missing_as_not_found(test_settings) -> None:
    """Comparison must not infer missing parameter values."""
    with session_scope(test_settings) as session:
        doc_a = Document(
            file_path="a.pdf",
            filename="a.pdf",
            file_type="pdf",
            file_hash="a",
            sensor_model="A100",
        )
        doc_b = Document(
            file_path="b.pdf",
            filename="b.pdf",
            file_type="pdf",
            file_hash="b",
            sensor_model="B200",
        )
        session.add_all([doc_a, doc_b])
        session.flush()
        session.add(
            ExtractedParameter(
                document_id=doc_a.id,
                sensor_model="A100",
                name="测距范围",
                normalized_name="range",
                value="100",
                unit="m",
                source_text="测距范围: 100 m",
                page_number=1,
            )
        )

    table = ParameterComparer(test_settings).compare_models(["A100", "B200"])
    assert table["range"]["A100"].value == "100 m"
    assert table["range"]["B200"] == ComparisonCell(
        "未找到",
        "未在已入库文档中找到依据",
    )

