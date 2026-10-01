"""
Active-learning correction endpoints.

POST /api/v1/corrections  — log a single analyst correction
GET  /api/v1/corrections/stats — aggregated stats for the learning dashboard
GET  /api/v1/corrections/training-data — export raw flagged examples
     (fine-tuning data + per-engine-version error tracking for the thesis)

Corrections are the thesis's active-learning signal: every time an analyst
flags a false positive or adds a missed span, we record it twice —
hashed in `corrections` (KVKK-safe analytics) and raw in `training_examples`
(on-prem fine-tuning store, stamped with ENGINE_VERSION).
"""

from fastapi import APIRouter, Request
from pydantic import BaseModel

from app.db import audit

router = APIRouter()


class CorrectionRequest(BaseModel):
    job_id: str | None = None
    # false_positive | false_negative | relabel
    event_type: str
    entity_label: str        # original label
    entity_text: str         # original text — hashed in corrections, raw in training_examples
    detector_source: str | None = None   # regex | ner | None (for false negatives)
    corrected_label: str | None = None   # for relabel events
    confidence: float | None = None      # detector confidence at correction time
    context: str | None = None           # surrounding text window — for NER fine-tuning


class CorrectionResponse(BaseModel):
    recorded: bool


@router.post("/api/v1/corrections", response_model=CorrectionResponse)
async def log_correction(request: CorrectionRequest, req: Request) -> CorrectionResponse:
    ip = req.client.host if req.client else None
    await audit.record_correction(
        job_id=request.job_id,
        event_type=request.event_type,
        entity_label=request.entity_label,
        entity_text=request.entity_text,
        detector_source=request.detector_source,
        corrected_label=request.corrected_label,
        confidence=request.confidence,
        context=request.context,
        ip_address=ip,
    )
    return CorrectionResponse(recorded=True)


@router.get("/api/v1/corrections/stats")
async def correction_stats() -> dict:
    return await audit.get_correction_stats()


@router.get("/api/v1/corrections/training-data")
async def training_data(
    limit: int = 1000,
    event_type: str | None = None,
    entity_label: str | None = None,
    engine_version: str | None = None,
) -> dict:
    """
    Export raw flagged examples for fine-tuning and thesis evaluation.
    `by_version` in the response groups error counts per engine version —
    the before/after evidence that each pipeline change reduced errors.
    """
    return await audit.get_training_examples(
        limit=limit,
        event_type=event_type,
        entity_label=entity_label,
        engine_version=engine_version,
    )
