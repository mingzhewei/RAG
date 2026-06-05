"""Tests for document parsing and chunking."""

from pathlib import Path

import pytest

from sensor_vector_db.core.document_processor import (
    DocumentChunker,
    DocumentParserFactory,
    MetadataExtractor,
    PDFParser,
)
from sensor_vector_db.core.types import OperationCancelled, ParsedSegment


def test_metadata_extractor_prefers_labelled_model() -> None:
    """An explicit 型号/Model label is the highest-confidence model evidence."""
    extractor = MetadataExtractor()
    assert extractor._extract_model("型号: LDR-100\n测距范围: 100 m") == "LDR-100"
    assert extractor._extract_model("Model: VLP-16 sensor") == "VLP-16"


def test_metadata_extractor_ignores_standards_and_interfaces() -> None:
    """The label-free fallback must not treat standards/interfaces as models."""
    extractor = MetadataExtractor()
    assert extractor._extract_model("本产品符合 ISO9001 标准，IP67 防护") is None
    assert extractor._extract_model("通信接口 USB3 和 RS232，参考 GB2312") is None
    # A genuine designator after a noise token is still recovered.
    assert extractor._extract_model("符合 ISO9001 的型号 ABC-200 传感器") == "ABC-200"


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


def test_parser_factory_reuses_parser_instances(test_settings, tmp_path: Path) -> None:
    """Parser factory should keep heavyweight parser clients alive across files."""
    factory = DocumentParserFactory(test_settings)

    first_pdf_parser = factory.get_parser(tmp_path / "first.pdf")
    second_pdf_parser = factory.get_parser(tmp_path / "second.pdf")
    first_text_parser = factory.get_parser(tmp_path / "first.txt")
    second_text_parser = factory.get_parser(tmp_path / "second.txt")

    assert isinstance(first_pdf_parser, PDFParser)
    assert first_pdf_parser is second_pdf_parser
    assert first_text_parser is second_text_parser


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


def test_pdf_parser_respects_ocr_page_limit(test_settings, tmp_path: Path, monkeypatch) -> None:
    """PDF OCR should be capped when ocr_max_pages_per_file is configured."""
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf_path = tmp_path / "blank.pdf"
    canvas = reportlab.Canvas(str(pdf_path))
    for _ in range(3):
        canvas.showPage()
    canvas.save()

    settings = test_settings.model_copy(
        update={
            "ocr_enabled": True,
            "ocr_min_text_chars": 80,
            "ocr_max_pages_per_file": 1,
        }
    )
    opened_documents = []
    closed_documents = []
    ocr_pages = []

    def fake_open(path):
        document = object()
        opened_documents.append(path)
        return document

    def fake_close(document):
        closed_documents.append(document)

    def fake_ocr(self, document, path, page_index):
        del self, document, path
        ocr_pages.append(page_index)
        return f"OCR page {page_index + 1}"

    monkeypatch.setattr(PDFParser, "_open_pdfium_document", staticmethod(fake_open))
    monkeypatch.setattr(PDFParser, "_close_pdfium_document", staticmethod(fake_close))
    monkeypatch.setattr(PDFParser, "_ocr_pdf_page", fake_ocr)

    segments = PDFParser(settings).parse(pdf_path)

    assert opened_documents == [pdf_path]
    assert len(closed_documents) == 1
    assert ocr_pages == [0]
    assert [segment.content for segment in segments] == ["OCR page 1"]


def test_pdf_parser_propagates_cancellation(test_settings, tmp_path: Path) -> None:
    """PDF parser cancellation should not be wrapped as a parse failure."""
    reportlab = pytest.importorskip("reportlab.pdfgen.canvas")
    pdf_path = tmp_path / "sensor.pdf"
    canvas = reportlab.Canvas(str(pdf_path))
    canvas.drawString(72, 720, "Model: LDR-100")
    canvas.save()

    def cancel() -> None:
        raise OperationCancelled("stop requested")

    with pytest.raises(OperationCancelled):
        PDFParser(test_settings).parse(pdf_path, cancel_callback=cancel)
