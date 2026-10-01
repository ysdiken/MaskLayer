"""
API routes for MaskLayer.

POST /api/v1/mask    — detect PII, replace with placeholders, persist mapping.
POST /api/v1/unmask  — replace placeholders back with original values.
GET  /health         — liveness check + NER model status.

The masking pipeline (pipeline.py) orchestrates regex + NER detection,
span merging, coreference grouping, and placeholder assignment. Routes are
intentionally thin — just HTTP plumbing around the pipeline.
"""

import time
import uuid

from fastapi import APIRouter, HTTPException, Request

from app.db import audit
from app.masking import mask_policy
from app.masking.mapping_store import MappingStore
from app.masking.ner_engine import engine as ner_engine
from app.masking.pipeline import pipeline
from app.models.schemas import (
    MaskRequest,
    MaskResponse,
    UnmaskRequest,
    UnmaskResponse,
)

router = APIRouter()


@router.get("/health")
async def health() -> dict:
    return {
        "status": "ok",
        "service": "masklayer",
        "ner_available": ner_engine.available,
    }


@router.post("/api/v1/mask", response_model=MaskResponse)
async def mask_text(request: MaskRequest, req: Request) -> MaskResponse:
    start_time = time.perf_counter()

    # Output policy: which entity types this deployment actually masks.
    # Detection is unaffected — result.spans is always the full detection.
    result = await pipeline.run(
        request.text,
        mode=request.mode,
        manual_spans=request.manual_spans or [],
        disabled_labels=mask_policy.get_disabled_labels(),
    )

    job_id = str(uuid.uuid4())
    store = MappingStore(req.app.state.redis)
    await store.save_mapping(job_id, result.mapping)

    elapsed_ms = (time.perf_counter() - start_time) * 1000

    # Await audit write — fast (single INSERT), never raises to caller
    ip = req.client.host if req.client else None
    await audit.record_mask(
        job_id,
        result,
        mode=request.mode,
        processing_time_ms=elapsed_ms,
        ip_address=ip,
    )

    return MaskResponse(
        job_id=job_id,
        original_text=request.text,
        masked_text=result.masked_text,
        spans=result.spans,
        processing_time_ms=elapsed_ms,
    )


@router.post("/api/v1/unmask", response_model=UnmaskResponse)
async def unmask_text(request: UnmaskRequest, req: Request) -> UnmaskResponse:
    """
    Replace placeholders in an LLM response with the original PII values.
    job_id must match an active mapping (default TTL: 24 h).
    Unknown placeholders are left as-is.
    """
    store = MappingStore(req.app.state.redis)

    mapping = await store.get_all(request.job_id)
    if not mapping:
        raise HTTPException(
            status_code=404,
            detail=(
                f"No mapping found for job_id '{request.job_id}'. "
                "It may have expired or never existed."
            ),
        )

    unmasked = await store.unmask_text(request.job_id, request.masked_text)

    # Audit the unmask (PII access) event
    ip = req.client.host if req.client else None
    await audit.record_unmask(
        request.job_id,
        placeholder_count=len(mapping),
        ip_address=ip,
    )

    return UnmaskResponse(job_id=request.job_id, unmasked_text=unmasked)
