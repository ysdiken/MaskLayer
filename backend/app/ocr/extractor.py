"""
Document text extractor — pdfplumber first, Tesseract fallback.

Strategy per file type:
  PDF  → pdfplumber (native text layer)
           If a page yields < MIN_CHARS chars (scanned page), fall back
           to Tesseract OCR on that page's raster image.
  Image (PNG/JPG/TIFF/BMP/WEBP) → Tesseract directly.
  DOCX → python-docx paragraph concatenation.

pdfplumber vs. Tesseract trade-offs:
  pdfplumber  — exact character positions, no GPU needed, very fast (~5 ms/page).
                Works on digitally-created PDFs. Returns "" on scanned pages.
  Tesseract 5 — slower (~500 ms/page on CPU, ~150 ms with GPU via pytesseract).
                Required for scanned documents. Configured for Turkish (tur).

The PageResult dataclass carries both the plain text and the method used,
so the API response can flag "ocr" pages for analyst review and the audit
log can record extraction quality.

Thesis note:
  This layer feeds directly into the masking pipeline. For the evaluation
  corpus, record per-document extraction_method so you can separately
  measure masking F1 on native-PDF vs. OCR-extracted text — OCR errors
  (especially on diacritics Ğ/Ş/İ/Ö/Ü/Ç) degrade NER recall.

Dependencies:
  pip install pdfplumber pymupdf pytesseract pillow python-docx
  Tesseract binary: https://github.com/UB-Mannheim/tesseract/wiki (Windows)
  Turkish language pack: tessdata/tur.traineddata
"""

from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Literal

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Tesseract binary path (Windows install default)
# Override with TESSERACT_CMD env var if installed elsewhere.
# ---------------------------------------------------------------------------

import os as _os
_TESSERACT_CMD = _os.getenv(
    "TESSERACT_CMD",
    r"C:\Program Files\Tesseract-OCR\tesseract.exe",
)

# ---------------------------------------------------------------------------
# Optional dependency imports — imported at module level so tests can patch
# them. Each is wrapped in a try/except so the module loads even when the
# library is not installed (graceful degradation path in the extractor methods).
# ---------------------------------------------------------------------------

try:
    import pdfplumber  # type: ignore
except ImportError:
    pdfplumber = None  # type: ignore

try:
    import fitz  # PyMuPDF  # type: ignore
except ImportError:
    fitz = None  # type: ignore

try:
    import pytesseract  # type: ignore
    from PIL import Image as _PILImage  # type: ignore
    pytesseract.pytesseract.tesseract_cmd = _TESSERACT_CMD
    _PIL_AVAILABLE = True
except ImportError:
    pytesseract = None  # type: ignore
    _PILImage = None    # type: ignore
    _PIL_AVAILABLE = False

try:
    from docx import Document as _DocxDocument  # type: ignore
    _DOCX_AVAILABLE = True
except ImportError:
    _DocxDocument = None  # type: ignore
    _DOCX_AVAILABLE = False

# Minimum native-text characters per page before we treat it as a scanned page
MIN_CHARS_NATIVE = 20

# Tesseract language string — Turkish + English fallback
TESSERACT_LANG = "tur+eng"

# DPI for rasterising PDF pages for Tesseract (higher = better OCR, slower)
RASTER_DPI = 200


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

ExtractionMethod = Literal["native", "ocr", "docx"]


@dataclass
class PageResult:
    page_number: int          # 1-indexed
    text: str
    method: ExtractionMethod  # how the text was obtained
    confidence: float | None = None  # Tesseract mean confidence (0–100), None for native/docx


@dataclass
class DocumentResult:
    filename: str
    pages: list[PageResult] = field(default_factory=list)
    error: str | None = None

    # Form-feed character used as page separator in full_text / masked_text.
    # \f is the traditional page break, won't appear in real document text,
    # and passes through the masking pipeline unchanged (not a PII pattern).
    PAGE_SEP = "\f"

    @property
    def full_text(self) -> str:
        """Concatenate pages with a form-feed page separator.

        Using \f (ASCII 12 / form feed) lets the frontend split both
        original_text and masked_text at identical boundaries — placeholder
        substitution preserves non-PII characters, so \f stays in place.
        """
        return self.PAGE_SEP.join(p.text for p in self.pages if p.text.strip())

    @property
    def page_breaks(self) -> list[int]:
        """Character offsets (in full_text) where each new page starts.

        Index 0 is always 0 (start of page 1). Useful for the frontend to
        compute which spans belong to which page card.
        """
        non_empty = [p.text for p in self.pages if p.text.strip()]
        offsets: list[int] = []
        pos = 0
        for text in non_empty:
            offsets.append(pos)
            pos += len(text) + 1  # +1 for the \f separator
        return offsets

    @property
    def extraction_methods(self) -> set[ExtractionMethod]:
        return {p.method for p in self.pages}

    @property
    def has_ocr_pages(self) -> bool:
        return "ocr" in self.extraction_methods


# ---------------------------------------------------------------------------
# Main extractor
# ---------------------------------------------------------------------------

class DocumentExtractor:
    """
    Stateless extractor. Instantiate once at module level and reuse.

    All methods are synchronous (CPU-bound). Callers in async context
    should dispatch via:
        loop.run_in_executor(None, extractor.extract_bytes, data, filename)
    """

    def extract_path(self, path: Path | str) -> DocumentResult:
        """Extract text from a file on disk."""
        path = Path(path)
        try:
            data = path.read_bytes()
        except OSError as e:
            return DocumentResult(filename=path.name, error=str(e))
        return self.extract_bytes(data, path.name)

    def extract_bytes(self, data: bytes, filename: str) -> DocumentResult:
        """
        Extract text from raw bytes. Dispatches based on file extension.
        filename is used only for extension detection and logging.
        """
        suffix = Path(filename).suffix.lower()

        if suffix == ".pdf":
            return self._extract_pdf(data, filename)
        elif suffix in {".docx", ".doc"}:
            return self._extract_docx(data, filename)
        elif suffix in {".png", ".jpg", ".jpeg", ".tiff", ".tif", ".bmp", ".webp"}:
            return self._extract_image(data, filename)
        else:
            return DocumentResult(
                filename=filename,
                error=f"Unsupported file type: '{suffix}'. Supported: pdf, docx, png, jpg, tiff, bmp, webp.",
            )

    # ── PDF ───────────────────────────────────────────────────────────────────

    def _extract_pdf(self, data: bytes, filename: str) -> DocumentResult:
        if pdfplumber is None:
            return DocumentResult(
                filename=filename,
                error="pdfplumber not installed. Run: pip install pdfplumber",
            )

        result = DocumentResult(filename=filename)

        try:
            with pdfplumber.open(io.BytesIO(data)) as pdf:
                for page_num, page in enumerate(pdf.pages, start=1):
                    native_text = (page.extract_text() or "").strip()

                    if len(native_text) >= MIN_CHARS_NATIVE:
                        result.pages.append(PageResult(
                            page_number=page_num,
                            text=native_text,
                            method="native",
                        ))
                        logger.debug("PDF '%s' page %d: native text (%d chars).",
                                     filename, page_num, len(native_text))
                    else:
                        # Scanned page — rasterise and OCR
                        logger.debug("PDF '%s' page %d: native text too short (%d chars), "
                                     "falling back to Tesseract.", filename, page_num, len(native_text))
                        ocr_result = self._ocr_pdf_page(data, page_num, filename)
                        result.pages.append(ocr_result)

        except Exception as e:
            logger.error("PDF extraction failed for '%s': %s", filename, e)
            result.error = str(e)

        return result

    def _ocr_pdf_page(self, pdf_data: bytes, page_num: int, filename: str) -> PageResult:
        """Rasterise a single PDF page with PyMuPDF and OCR with Tesseract."""
        if fitz is None:
            logger.warning("PyMuPDF not installed — cannot rasterise PDF page for OCR. "
                           "Run: pip install pymupdf")
            return PageResult(page_number=page_num, text="", method="ocr")

        try:
            doc = fitz.open(stream=pdf_data, filetype="pdf")
            fitz_page = doc[page_num - 1]  # 0-indexed
            mat = fitz.Matrix(RASTER_DPI / 72, RASTER_DPI / 72)  # 72 DPI is PDF default
            pix = fitz_page.get_pixmap(matrix=mat, colorspace=fitz.csRGB)
            img_bytes = pix.tobytes("png")
            doc.close()
        except Exception as e:
            logger.error("PyMuPDF rasterisation failed for '%s' page %d: %s", filename, page_num, e)
            return PageResult(page_number=page_num, text="", method="ocr")

        return self._run_tesseract(img_bytes, page_num)

    # ── Image ─────────────────────────────────────────────────────────────────

    def _extract_image(self, data: bytes, filename: str) -> DocumentResult:
        result = DocumentResult(filename=filename)
        page = self._run_tesseract(data, page_number=1)
        result.pages.append(page)
        return result

    def _run_tesseract(self, img_bytes: bytes, page_number: int) -> PageResult:
        """Run Tesseract on raw image bytes. Returns PageResult with confidence."""
        if not _PIL_AVAILABLE or pytesseract is None:
            logger.warning("pytesseract or Pillow not installed. "
                           "Run: pip install pytesseract pillow")
            return PageResult(page_number=page_number, text="", method="ocr")

        try:
            image = _PILImage.open(io.BytesIO(img_bytes))

            # Get text
            text = pytesseract.image_to_string(image, lang=TESSERACT_LANG, config="--psm 3")

            # Get per-word confidence data for mean confidence metric
            try:
                data = pytesseract.image_to_data(
                    image, lang=TESSERACT_LANG, config="--psm 3",
                    output_type=pytesseract.Output.DICT,
                )
                confs = [int(c) for c in data["conf"] if str(c).lstrip("-").isdigit() and int(c) >= 0]
                mean_conf = round(sum(confs) / len(confs), 1) if confs else None
            except Exception:
                mean_conf = None

            logger.debug("Tesseract page %d: %d chars, confidence %.1f%%",
                         page_number, len(text), mean_conf or 0)

            return PageResult(
                page_number=page_number,
                text=text.strip(),
                method="ocr",
                confidence=mean_conf,
            )

        except Exception as e:
            logger.error("Tesseract OCR failed on page %d: %s", page_number, e)
            return PageResult(page_number=page_number, text="", method="ocr")

    # ── DOCX ─────────────────────────────────────────────────────────────────

    def _extract_docx(self, data: bytes, filename: str) -> DocumentResult:
        if not _DOCX_AVAILABLE or _DocxDocument is None:
            return DocumentResult(
                filename=filename,
                error="python-docx not installed. Run: pip install python-docx",
            )

        result = DocumentResult(filename=filename)
        try:
            doc = _DocxDocument(io.BytesIO(data))
            # Treat the whole DOCX as a single "page" — DOCX has no page concept
            paragraphs = [p.text for p in doc.paragraphs if p.text.strip()]

            # Also extract text from tables
            for table in doc.tables:
                for row in table.rows:
                    for cell in row.cells:
                        cell_text = cell.text.strip()
                        if cell_text:
                            paragraphs.append(cell_text)

            full_text = "\n".join(paragraphs)
            result.pages.append(PageResult(
                page_number=1,
                text=full_text,
                method="docx",
            ))
            logger.debug("DOCX '%s': extracted %d chars from %d paragraphs.",
                         filename, len(full_text), len(paragraphs))

        except Exception as e:
            logger.error("DOCX extraction failed for '%s': %s", filename, e)
            result.error = str(e)

        return result


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------

extractor = DocumentExtractor()
