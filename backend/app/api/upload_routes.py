"""
File upload → text extraction → masking pipeline endpoint.

POST /api/v1/upload
  Accepts: multipart/form-data with a single file field "file"
  Supported: PDF, DOCX, PNG, JPG, TIFF, BMP, WEBP
  Optional form fields:
    mode   (str)  — "full" | "regex" | "ner"  (default: "full")

  Returns: MaskResponse with an added `extraction` block describing
           which pages used native text vs. Tesseract OCR.

This is the primary entry point for real-document testing and the demo.
The masking pipeline (pipeline.py) is unchanged — this layer just feeds
extracted text into it.

Thesis note: `extraction_method` in the response lets you separate your
evaluation corpus into "native PDF" and "OCR" subsets and report F1 for
each. OCR errors on Turkish diacritics (Ğ Ş İ Ö Ü Ç) are an expected
source of NER recall degradation worth analysing.
"""

import asyncio
import time
import uuid
from typing import Literal

from fastapi import APIRouter, File, Form, HTTPException, Request, UploadFile
from pydantic import BaseModel

from app.db import audit
from app.masking import mask_policy
from app.masking.mapping_store import MappingStore
from app.masking.pipeline import pipeline
from app.models.schemas import Span
from app.ocr.extractor import extractor, ExtractionMethod
from pathlib import Path

router = APIRouter()


# ---------------------------------------------------------------------------
# Response schema — extends MaskResponse with extraction metadata
# ---------------------------------------------------------------------------

class PageExtractionInfo(BaseModel):
    page_number: int
    method: ExtractionMethod
    confidence: float | None  # Tesseract mean confidence, None for native/docx


class UploadMaskResponse(BaseModel):
    job_id: str
    filename: str
    original_text: str
    masked_text: str
    spans: list[Span]
    processing_time_ms: float
    extraction: list[PageExtractionInfo]
    has_ocr_pages: bool
    # Character offset in original_text/masked_text where each page starts.
    # Both strings use the same \f separator, so these offsets work for both.
    # Index 0 is always 0 (start of page 1).
    page_breaks: list[int]
    page_separator: str  # always "\f" — sent explicitly so the frontend never hard-codes it


# ---------------------------------------------------------------------------
# Route
# ---------------------------------------------------------------------------

@router.post("/api/v1/upload", response_model=UploadMaskResponse)
async def upload_and_mask(
    req: Request,
    file: UploadFile = File(...),
    mode: Literal["full", "regex", "ner"] = Form("full"),
) -> UploadMaskResponse:
    start_time = time.perf_counter()

    # ── 1. Read upload ────────────────────────────────────────────────────────
    raw_bytes = await file.read()
    filename = file.filename or "upload"

    if not raw_bytes:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # ── 2. Extract text (CPU-bound — thread pool) ─────────────────────────────
    loop = asyncio.get_event_loop()
    doc_result = await loop.run_in_executor(
        None, extractor.extract_bytes, raw_bytes, filename
    )

    if doc_result.error:
        raise HTTPException(
            status_code=422,
            detail=f"Text extraction failed: {doc_result.error}",
        )

    full_text = doc_result.full_text
    if not full_text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted from the document. "
                   "Check Tesseract is installed and the Turkish language pack (tur.traineddata) is available.",
        )

    # ── 3. Masking pipeline ───────────────────────────────────────────────────
    # disabled_labels applies the deployment's output policy; detection is full.
    pipeline_result = await pipeline.run(
        full_text, mode=mode, disabled_labels=mask_policy.get_disabled_labels()
    )

    # ── 4. Persist mapping to Redis ───────────────────────────────────────────
    job_id = str(uuid.uuid4())
    store = MappingStore(req.app.state.redis)
    await store.save_mapping(job_id, pipeline_result.mapping)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Fire-and-forget audit write
    ip = req.client.host if req.client else None
    ocr_pages = [p for p in doc_result.pages if p.method == "ocr"]
    ocr_conf = (
        round(sum(p.confidence for p in ocr_pages if p.confidence is not None)
              / len(ocr_pages), 1)
        if ocr_pages else None
    )
    await audit.record_mask(
        job_id,
        pipeline_result,
        filename=filename,
        file_type=Path(filename).suffix.lstrip(".").lower(),
        page_count=len(doc_result.pages),
        extraction_methods=doc_result.extraction_methods,
        ocr_mean_confidence=ocr_conf,
        mode=mode,
        processing_time_ms=elapsed_ms,
        ip_address=ip,
    )

    extraction_info = [
        PageExtractionInfo(
            page_number=p.page_number,
            method=p.method,
            confidence=p.confidence,
        )
        for p in doc_result.pages
    ]

    return UploadMaskResponse(
        job_id=job_id,
        filename=filename,
        original_text=full_text,
        masked_text=pipeline_result.masked_text,
        spans=pipeline_result.spans,
        processing_time_ms=elapsed_ms,
        extraction=extraction_info,
        has_ocr_pages=doc_result.has_ocr_pages,
        page_breaks=doc_result.page_breaks,
        page_separator=doc_result.PAGE_SEP,
    )
