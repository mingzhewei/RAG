"""Tests for document parsing and chunking."""

from pathlib import Path

import pytest

from sensor_vector_db.core.document_processor import (
    DocumentChunker,
    DocumentParserFactory,
    PDFParser,
)
from sensor_vector_db.core.types import ParsedSegment


def test_text_and_code_parsing(test_settings, tmp_path: Path) -> None:
    """Parser factory should parse text and Python code."""
    text_file = tmp_path / "sensor.txt"
    code_file = tmp_path / "driver.py"
    text_file.write_text("型号: LDR-100\n测距范围: 100 m", encoding="utf-8")
    code_file.write_text(
        "class Driver:\n    pass\n\ndef read_sensor():\n    return 1\n",
        encoding="utf-8",
    )

    factory = DocumentParserFactory(test_settings)
    assert factory.parse(text_file)[0].content_type == "text"
    code_segments = factory.parse(code_file)
    assert any("read_sensor" in segment.content for segment in code_segments)


def test_chunk_overlap_preserves_page(test_settings) -> None:
    """Chunking should preserve page metadata."""
    segment = ParsedSegment(
        content=" ".join(f"token{i}" for i in range(120)),
        content_type="text",
        page_number=3,
    )
    chunks = DocumentChunker(test_settings).chunk([segment])
    assert len(chunks) >= 2
    assert all(chunk.page_number == 3 for chunk in chunks)


def test_pdf_parser_text_page(test_settings, tmp_path: Path) -> None:
    """PDF parser should extract text from a synthetic PDF."""
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf_path = tmp_path / "sensor.pdf"
    canvas = reportlab.Canvas(str(pdf_path))
    canvas.drawString(72, 720, "Model: LDR-100")
    canvas.drawString(72, 700, "Range: 100 m")
    canvas.save()

    segments = PDFParser(test_settings).parse(pdf_path)
    assert any("LDR-100" in segment.content for segment in segments)
    assert all(segment.page_number == 1 for segment in segments)

