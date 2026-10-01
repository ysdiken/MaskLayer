"""
Unit tests for the OCR extractor.

Tests are structured so they run without Tesseract installed:
  - Logic tests (unsupported type, empty file, method dispatch) use mocks.
  - pdfplumber path is tested with a minimal synthetic PDF byte string.
  - DOCX path is tested with a real in-memory DOCX.
  - Tesseract path is mocked (CI machines may not have the tur pack).

Run with:
    cd backend
    pytest tests/test_ocr_extractor.py -v
"""

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from app.ocr.extractor import DocumentExtractor, DocumentResult, PageResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_extractor() -> DocumentExtractor:
    return DocumentExtractor()


# ---------------------------------------------------------------------------
# Unsupported file type
# ---------------------------------------------------------------------------

class TestUnsupportedType:
    def test_returns_error_for_xlsx(self):
        ex = make_extractor()
        result = ex.extract_bytes(b"dummy", "report.xlsx")
        assert result.error is not None
        assert "xlsx" in result.error.lower() or "Unsupported" in result.error

    def test_returns_error_for_no_extension(self):
        ex = make_extractor()
        result = ex.extract_bytes(b"dummy", "noextension")
        assert result.error is not None


# ---------------------------------------------------------------------------
# DOCX extraction
# ---------------------------------------------------------------------------

class TestDocxExtraction:
    def _make_docx_bytes(self, text: str) -> bytes:
        """Create a minimal in-memory DOCX with one paragraph."""
        from docx import Document
        doc = Document()
        doc.add_paragraph(text)
        buf = io.BytesIO()
        doc.save(buf)
        return buf.getvalue()

    def test_extracts_paragraph_text(self):
        content = "Ahmet Yılmaz'ın TC Kimlik No: 10000000146"
        docx_bytes = self._make_docx_bytes(content)
        ex = make_extractor()
        result = ex.extract_bytes(docx_bytes, "test.docx")
        assert result.error is None
        assert content in result.full_text

    def test_method_is_docx(self):
        docx_bytes = self._make_docx_bytes("Merhaba dünya")
        ex = make_extractor()
        result = ex.extract_bytes(docx_bytes, "test.docx")
        assert all(p.method == "docx" for p in result.pages)

    def test_page_number_is_one(self):
        docx_bytes = self._make_docx_bytes("Test")
        ex = make_extractor()
        result = ex.extract_bytes(docx_bytes, "test.docx")
        assert result.pages[0].page_number == 1

    def test_extracts_table_text(self):
        from docx import Document
        doc = Document()
        table = doc.add_table(rows=1, cols=2)
        table.cell(0, 0).text = "İsim"
        table.cell(0, 1).text = "Ahmet"
        buf = io.BytesIO()
        doc.save(buf)
        ex = make_extractor()
        result = ex.extract_bytes(buf.getvalue(), "table.docx")
        assert "Ahmet" in result.full_text


# ---------------------------------------------------------------------------
# PDF extraction — pdfplumber path (mocked)
# ---------------------------------------------------------------------------

class TestPDFNativePath:
    def test_native_text_used_when_long_enough(self):
        """If pdfplumber returns >= MIN_CHARS_NATIVE chars, method is 'native'."""
        long_text = "A" * 100

        mock_page = MagicMock()
        mock_page.extract_text.return_value = long_text

        mock_pdf_instance = MagicMock()
        mock_pdf_instance.__enter__ = MagicMock(return_value=mock_pdf_instance)
        mock_pdf_instance.__exit__ = MagicMock(return_value=False)
        mock_pdf_instance.pages = [mock_page]

        ex = make_extractor()
        # Patch at the import site inside the extractor module
        with patch("app.ocr.extractor.pdfplumber") as mock_pdfplumber:
            mock_pdfplumber.open.return_value = mock_pdf_instance
            result = ex.extract_bytes(b"%PDF-fake", "doc.pdf")

        assert result.error is None
        assert len(result.pages) == 1
        assert result.pages[0].method == "native"
        assert result.pages[0].text == long_text

    def test_ocr_fallback_when_native_too_short(self):
        """If pdfplumber returns < MIN_CHARS_NATIVE chars, method is 'ocr'."""
        short_text = "Hi"  # less than MIN_CHARS_NATIVE (20)

        mock_page = MagicMock()
        mock_page.extract_text.return_value = short_text

        mock_pdf_instance = MagicMock()
        mock_pdf_instance.__enter__ = MagicMock(return_value=mock_pdf_instance)
        mock_pdf_instance.__exit__ = MagicMock(return_value=False)
        mock_pdf_instance.pages = [mock_page]

        ex = make_extractor()
        ocr_page = PageResult(page_number=1, text="OCR result", method="ocr", confidence=85.0)

        with patch("app.ocr.extractor.pdfplumber") as mock_pdfplumber:
            mock_pdfplumber.open.return_value = mock_pdf_instance
            with patch.object(ex, "_ocr_pdf_page", return_value=ocr_page):
                result = ex.extract_bytes(b"%PDF-fake", "scanned.pdf")

        assert result.pages[0].method == "ocr"
        assert result.pages[0].text == "OCR result"


# ---------------------------------------------------------------------------
# Tesseract path (mocked — no actual Tesseract required)
# ---------------------------------------------------------------------------

class TestTesseractPath:
    def test_image_file_calls_tesseract(self):
        ex = make_extractor()
        ocr_page = PageResult(page_number=1, text="Merhaba", method="ocr", confidence=90.0)

        with patch.object(ex, "_run_tesseract", return_value=ocr_page) as mock_tess:
            result = ex.extract_bytes(b"\x89PNG\r\n", "scan.png")

        mock_tess.assert_called_once()
        assert result.pages[0].text == "Merhaba"
        assert result.pages[0].method == "ocr"

    def test_run_tesseract_graceful_degradation_no_pytesseract(self):
        """If pytesseract is not installed, returns empty PageResult without crash."""
        ex = make_extractor()
        with patch.dict("sys.modules", {"pytesseract": None, "PIL": None, "PIL.Image": None}):
            page = ex._run_tesseract(b"\x89PNG\r\n", page_number=1)
        assert page.method == "ocr"
        assert page.text == ""


# ---------------------------------------------------------------------------
# DocumentResult helpers
# ---------------------------------------------------------------------------

class TestDocumentResult:
    def test_full_text_joins_pages(self):
        result = DocumentResult(filename="test.pdf")
        result.pages = [
            PageResult(page_number=1, text="Sayfa bir", method="native"),
            PageResult(page_number=2, text="Sayfa iki", method="native"),
        ]
        assert "Sayfa bir" in result.full_text
        assert "Sayfa iki" in result.full_text

    def test_has_ocr_pages_true(self):
        result = DocumentResult(filename="test.pdf")
        result.pages = [
            PageResult(page_number=1, text="text", method="native"),
            PageResult(page_number=2, text="ocr text", method="ocr"),
        ]
        assert result.has_ocr_pages is True

    def test_has_ocr_pages_false(self):
        result = DocumentResult(filename="test.pdf")
        result.pages = [
            PageResult(page_number=1, text="text", method="native"),
        ]
        assert result.has_ocr_pages is False

    def test_extraction_methods_set(self):
        result = DocumentResult(filename="test.pdf")
        result.pages = [
            PageResult(page_number=1, text="a", method="native"),
            PageResult(page_number=2, text="b", method="ocr"),
        ]
        assert result.extraction_methods == {"native", "ocr"}

    def test_full_text_skips_blank_pages(self):
        result = DocumentResult(filename="test.pdf")
        result.pages = [
            PageResult(page_number=1, text="İçerik", method="native"),
            PageResult(page_number=2, text="   ", method="ocr"),   # blank OCR page
        ]
        assert result.full_text.strip() == "İçerik"
